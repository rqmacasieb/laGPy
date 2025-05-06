import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict
from .gp import *
from .matrix import get_data_rect
from .covar import *
from .params import *
import time
from .utils.distance import *

class Method(Enum):
    ALC = 1
    ALCOPT = 2 
    ALCRAY = 3
    MSPE = 4
    EFI = 5
    NN = 6


def fullGP(Xref: np.ndarray, 
        X: np.ndarray, 
        Z: np.ndarray, 
        d: Optional[Union[float, Tuple[float, float]]] = None,
        g: float = 1/10000,
        lite: bool = True,
        verb: int = 0) -> Dict:
    """
    GP prediction utilizing full training dataset

    Args:
        Xref: Reference points for prediction
        X: Training inputs
        Z: Training outputs
        d: Lengthscale parameter
        g: Nugget parameter
        lite: Whether to use lite version (only diagonal of covariance)
        verb: Verbosity level
        
    Returns:
        Dictionary with the following keys:
            mean: Predicted means
            var: Predicted variances
            df: Degrees of freedom
            llik: Log likelihood
            d_posterior: Posterior lengthscale parameter
            g_posterior: Posterior nugget parameter
    """

    gp = buildGP(X, Z, d, g, export=False, verb=verb)

    if lite:
        results = gp.predict_lite(Xref)
    else:
        results = gp.predict(Xref)

    return {
        "mean": results["mean"],
        "s2": results["s2"],
        "df": results["df"],
        "llik": results["llik"],
        "d_posterior": gp.d,
        "g_posterior": gp.g,
    }


def _laGP(Xref: np.ndarray, 
         X: np.ndarray, 
         Z: np.ndarray, 
         start: Optional[int] = None, 
         end: Optional[int] = None, 
         d: Optional[Union[float, Tuple[float, float]]] = None,
         g: float = 1/10000,
         method: Method = Method.ALC,
         close: Optional[int] = None,
         numstart: Optional[int] = None,
         rect: Optional[np.ndarray] = None,
         lite: bool = True,
         verb: int = 0) -> Dict:
    """
    Local Approximate GP prediction with parameter estimation
    
    Args:
        Xref: Reference points for prediction
        start: Number of initial points
        end: Number of total points to select
        X: Input points
        Z: Output values
        d: Initial length scale (if None, will be estimated)
        g: Initial nugget (if None, will be estimated)
        method: Method for selecting points
        close: Number of close points to consider
        numstart: Number of starting points for ALCRAY
        rect: Optional rectangle bounds
        lite: Whether to use lite version (only diagonal of covariance)
        verb: Verbosity level
        
    Returns:
        Tuple of:
        - Mean predictions
        - Variance predictions
        - Selected indices
        - Final length scale
        - Final nugget
    """
    n = X.shape[0]
    # Get closest points for initial design
    idx = closest_indices(start, Xref, n, X, close, 
                         method in [Method.ALCRAY, Method.ALCOPT, Method.NN])

    if method == Method.NN:
        if verb > 0:
            print(f"NN method selected. Using {end} nearest points from the training set.")
        gp = newGP(X[idx[:end]], Z[idx[:end]], get_value(d, 'start'), get_value(g, 'start'))
        selected = idx[:end]
    
    else: 
        # Setup candidate points
        cand_idx = idx[start:]
        Xcand = X[cand_idx]
        selected = np.zeros(end, dtype=int)
        selected[:start] = idx[:start]

        # Build initial GP
        X_init = X[idx[:start]]
        Z_init = Z[idx[:start]]
        
        gp = newGP(X_init, Z_init, get_value(d, 'start'), get_value(g, 'start'))
        
        # Get rect bounds if needed
        if method in (Method.ALCRAY, Method.ALCOPT) and rect is None:
            rect = get_data_rect(Xcand)

        # Iteratively select points. Only performs ALC for now.
        for i in range(start, end):
            # Point selection logic based on method
            if method == Method.ALCRAY: #TODO: add funx if needed. placeholder for now
                offset = (i - start + 1) % int(np.sqrt(i - start + 1))
                w = alcray_selection(gp, Xcand, Xref, offset, numstart, rect, verb)
            elif method == Method.ALC:
                scores = alc(gp, Xcand, Xref, verb) #no gpu support for now
                w = np.argmax(scores)
            elif method == Method.MSPE: #TODO: add funx if needed. placeholder for now
                scores = mspe(gp, Xcand, Xref, verb)
                w = np.argmin(scores)
            # else:  # Method.NN
            #     w = i - start
                
            # Record chosen point
            selected[i] = cand_idx[w]
            
            # Update GP with chosen candidate
            gp.update(Xcand[w:w+1], Z[cand_idx[w:w+1]], verb=verb-1)
            
            # Re-estimate parameters periodically if requested TODO: do we need this?
            # if param_est and (i - start + 1) % est_freq == 0:
            #     optimize_parameters(gp, d, g, verb)

            # Update candidate set
            if w != len(cand_idx) - 1:
                if method in ['alcray', 'alcopt']:
                    if w == 0:
                        cand_idx = cand_idx[1:]
                        Xcand = Xcand[1:]
                    else:
                        cand_idx[w:] = cand_idx[w + 1:]
                        Xcand[w:] = Xcand[w + 1:]
                else:
                    cand_idx[w] = cand_idx[-1]
                    Xcand[w] = Xcand[-1]
                cand_idx = cand_idx[:-1]
                Xcand = Xcand[:-1]
            elif w == len(cand_idx) - 1:
                cand_idx = cand_idx[:-1]
                Xcand = Xcand[:-1]
            else:
                raise ValueError("candidate index is out of bounds")

    # If required, obtain parameter posterior by MLE and update gp before prediction
    optimize_parameters(gp, d, g, verb)
    
    # Given the updated gp, predict values and return results
    if lite:
        results = gp.predict_lite(Xref)
    else:
        results = gp.predict(Xref)
    
    return {
        "mean": results["mean"],
        "s2": results["s2"],
        "df": results["df"],
        "llik": results["llik"],
        "selected": selected,
        "d_posterior": gp.d,
        "g_posterior": gp.g,
    }

def laGP(Xref: np.ndarray, 
         X: np.ndarray, 
         Z: np.ndarray, 
         start: Optional[int] = None, 
         end: Optional[int] = None, 
         d: Optional[Union[float, Tuple[float, float]]] = None,
         g: float = 1/10000,
         method: Union[str, Method] = "alc",
         close: Optional[int] = None,
         numstart: Optional[int] = None,
         rect: Optional[np.ndarray] = None,
         lite: bool = True,
         verb: int = 0) -> Dict:
    """
    Local Approximate Gaussian Process Regression.
    Combined Python equivalent of laGP.R and laGP_R.c
    
    Args:
        Xref: Reference points for prediction (n_ref × m)
        X: Training inputs (n × m)
        Z: Training outputs (n,)
        start: Initial design size (must be >= 6; if None, full training design is used)
        end: Final design size (if None, full training design is used)
        d: Lengthscale parameter or tuple of (start, mle)
        g: Nugget parameter
        method: One of "alc", "alcopt", "alcray", "mspe", "nn", "fish"
        close: Number of close points to consider
        numstart: Number of starting points for ray-based methods
        rect: Rectangle bounds for ray-based methods
        lite: Whether to use lite version (only diagonal of covariance)
        verb: Verbosity level
        
    Returns:
        Dictionary containing:
            mean: Predicted means
            s2/Sigma: Predicted variances/covariance matrix
            df: Degrees of freedom
            llik: Log likelihood
            time: Computation time
            method: Method used
            d: Lengthscale parameters
            g: Nugget parameters
            close: Number of close points used
            selected: Selected indices (zero-indexed)
    """
    if isinstance(method, Method):
        method = method.name.lower()
    else:
        method = method.lower()

    # Method mapping
    method_map = {
        "alc": Method.ALC, "alcopt": Method.ALCOPT, "alcray": Method.ALCRAY,
        "mspe": Method.MSPE, "fish": Method.EFI, "nn": Method.NN
    }
    
    # Input processing
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")
    imethod = method_map[method]
    
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    Z = np.asarray(Z)
    Xref = np.atleast_2d(Xref)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    m = X.shape[1]
    n = X.shape[0]
    nref = Xref.shape[0]
    
    # Input validation
    if start is None and end is not None and method != "nn":
        raise ValueError("start must be provided ( <= start < end) if end is provided")
    if start is not None:
        if start < 6 or end <= start:
            raise ValueError("must have 6 <= start < end")
        if end is None:
            print("WARNING: Target design size is not provided. Using full training design for GP (i.e., NOT a local approximate GP!)")
        elif end > n:
            print(f"WARNING: Target design size = {end} is greater than training design size = {n}.\n"
                  f"Setting target design size to {n}. Using full training design for GP (i.e., NOT a local approximate GP!)")
            end = n
        elif end == n:
            print(f"WARNING: Target design size = {end} is equal to training design size = {n}. Using full training design for GP (i.e., NOT a local approximate GP!)")
    if Xref.shape[1] != m:
        raise ValueError(f"Dimension mismatch: Xref.shape = {Xref.shape}, X.shape = {X.shape}")
    if len(Z) != n:
        raise ValueError("Length of Z must match number of rows in X")
    
    # Set defaults
    if close is None:
        mult = 10 if method in ["alcray", "alcopt"] else 1
        close = min((1000 + end) * mult, n)
    if numstart is None:
        numstart = m if method == "alcray" else 1
    
    # Process rect
    if method in ["alcray", "alcopt"]:
        if rect is None:
            rect = np.zeros((2, m))
        if method == "alcray" and nref != 1:
            raise ValueError("alcray only implemented for nrow(Xref) = 1")
    else:
        rect = np.zeros(1)
    
    # Process parameters
    d_prior = darg(d, X)
    g_prior = garg(g, Z)
    
    # Start timing
    tic = time.time()
    
    # Call core implementation
    if (start is not None and end is not None and end < n) or (start is None and method == "nn"):
        results = _laGP(Xref=Xref,
            X=X, Z=Z, start=start, end=end,        
            d=d_prior, g=g_prior,
            method=Method(imethod),
            close=close,
            numstart=numstart,
            rect=rect,
            verb=verb,
            lite=lite
        )

        result = {
            'mean': results['mean'],
            's2': results['s2'],
            'selected': results['selected'],
            'df': results['df'],
            'llik': results['llik'],
            'time': time.time() - tic,
            'method': method,
            'd': results['d_posterior'],
            'g': results['g_posterior'],
            'close': close
        }
    elif (start is None and end is None) or end >= n: #full GP implementation
        results = fullGP(Xref=Xref, X=X, Z=Z, d=d_prior, g=g_prior, lite=lite, verb=verb)
        result = {
            'mean': results['mean'],
            's2': results['s2'],
            'df': results['df'],
            'llik': results['llik'],
            'time': time.time() - tic,
            'd': results['d_posterior'],
            'g': results['g_posterior'],
        }
    else:
        raise ValueError("start and end must be provided if start or end is not None")
    
    # Add s2/Sigma
    if not lite:
        result['Sigma'] = results['s2'].reshape(nref, nref)
    
    # Add ray info if needed
    # if method in ["alcray", "alcopt"]:
    #     result['numstart'] = numstart
    
    return result


def alc(gp, Xcand, Xref, verb=0):
    """
    Python implementation of ALC criterion.
    
    Args:
        gp: GP instance with attributes like m, n, Ki, d, g, phi, X
        Xcand: Candidate points (2D numpy array)
        Xref: Reference points (2D numpy array)
        verb: Verbosity level
        
    Returns:
        ALC scores for each candidate point
    """
    m = gp.m #number of dimensions
    n = gp.n #number of data points
    df = float(n)
    ncand = Xcand.shape[0]
    nref = Xref.shape[0]
    
    # Precompute covariance matrix
    k = covar(Xref, gp.X, gp.d)
    
    # Initialize ALC scores
    alc_scores = np.zeros(ncand)
    
    # Calculate the ALC for each candidate    
    if verb > 0:
        print(f"alc: calculating ALC for {ncand} points")
    mui, gvec, kxy = calc_g_mui_kxy(Xcand, gp.X, gp.Ki, Xref, gp.d, gp.g)

    # Create a mask for valid mui values
    valid_mask = mui > np.finfo(float).eps
    
    # Initialize alc_scores with -inf
    alc_scores = np.full(ncand, -np.inf)
        
    if np.any(valid_mask):
        ktKikx = calc_ktKikx(None, nref, k, gp.X.shape[0], gvec[valid_mask], mui[valid_mask], kxy[valid_mask], None, None)
        alc_scores[valid_mask] = calc_alc(nref, ktKikx, [0, 0], gp.phi, df)
    
    return alc_scores

def calc_ktKikx(ktKik, m, k, n, g, mui, kxy, Gmui=None, ktGmui=None):
    """
    Calculate the ktKikx vector for the IECI calculation.
    
    Args:
        ktKik: Initial ktKik vector (1D numpy array) or None
        m: Number of reference points
        k: Covariance matrix (2D numpy array)
        n: Number of data points
        g: g vector (1D numpy array)
        mui: Scalar value
        kxy: Covariance vector between candidate and reference points (1D numpy array)
        Gmui: Optional precomputed Gmui matrix (2D numpy array)
        ktGmui: Optional precomputed ktGmui vector (1D numpy array)
        
    Returns:
        ktKikx: Calculated ktKikx vector (1D numpy array)
    """
    # Check for dimension mismatch
    if k.shape[1] != g.shape[1]:
        raise ValueError(f"Dimension mismatch: k.shape = {k.shape}, g.shape = {g.shape}")
    # Calculate ktKikx
    if Gmui is not None:
        Gmui = np.outer(g, g) / mui
        assert ktGmui is not None
        ktGmui = np.dot(Gmui, k.T)
        ktKikx = np.dot(ktGmui, k.T)
        if ktKik is not None:
            ktKikx += ktKik
    else:
        dot_products = np.dot(g, k.T)
        squared_dot_products = dot_products ** 2
        ktKikx = squared_dot_products.flatten() * mui
        if ktKik is not None:
            ktKikx += ktKik

    # Add 2*diag(kxy %*% t(g) %*% k)
    ktKikx += 2.0 * dot_products.flatten() * kxy.flatten()

    # Add kxy^2/mui
    ktKikx += (kxy ** 2).flatten() / mui

    return ktKikx

def calc_alc(m, ktKik, s2p, phi, tdf, badj=None, w=None):
    """
    Calculate the Active Learning Criterion (ALC).
    
    Args:
        m: Number of points
        ktKik: Array of ktKik values (2D numpy array)
        s2p: Array of s2p values (1D numpy array)
        phi: Scalar value
        badj: Optional array of adjustment factors (1D numpy array)
        tdf: Degrees of freedom
        w: Optional array of weights (1D numpy array)
        
    Returns:
        ALC values for each candidate point (1D numpy array)
    """
    dfrat = tdf / (tdf - 2.0)
    
    # Calculate zphi for all points
    zphi = (s2p[1] + phi) * ktKik

    # Calculate ts2 based on whether badj is provided
    if badj is not None:
        ts2 = badj[:, np.newaxis] * zphi / (s2p[0] + tdf)
    else:
        ts2 = zphi / (s2p[0] + tdf)

    # Calculate alc based on whether w is provided
    if w is not None:
        alc = w[:, np.newaxis] * dfrat * ts2
    else:
        alc = ts2 * dfrat

    return alc 



