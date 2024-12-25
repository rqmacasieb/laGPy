import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict, NamedTuple
from .gp import *
from .matrix import get_data_rect
from .order import order
from .covar_sep import *
from .covar import *
from .params import *
import time
from .utils.distance import distance_asymm

class Method(Enum):
    ALC = 1
    ALCOPT = 2 
    ALCRAY = 3
    MSPE = 4
    EFI = 5
    NN = 6

# class MLEResult(NamedTuple):
#     """Results from MLE optimization"""
#     lengthscale: float  # Optimized lengthscale
#     nugget: float      # Optimized nugget
#     iterations: int    # Number of iterations
#     success: bool      # Whether optimization succeeded
#     llik: float       # Log likelihood at optimum

# def joint_mle_gp(gp: GP, 
#                 d_range: Tuple[float, float] = (1e-6, 1.0),
#                 g_range: Tuple[float, float] = (1e-6, 1.0),
#                 verb: int = 0) -> MLEResult:
#     """
#     Joint maximum likelihood estimation for GP lengthscale and nugget
    
#     Args:
#         gp: GP instance
#         d_range: (min, max) range for lengthscale
#         g_range: (min, max) range for nugget
#         verb: Verbosity level
        
#     Returns:
#         MLEResult containing optimized values and optimization info
#     """
#     def neg_log_likelihood(theta):
#         """Negative log likelihood function for joint optimization"""
#         d, g = theta
        
#         # Update GP parameters
#         gp.d = d
#         gp.g = g
#         gp.update_covariance()
        
#         try:
#             # Log determinant term
#             sign, logdet = np.linalg.slogdet(gp.K)
#             if sign <= 0:
#                 return np.inf
                
#             # Quadratic term
#             alpha = np.linalg.solve(gp.K, gp.Z)
#             quad = np.dot(gp.Z, alpha)
            
#             # Full negative log likelihood
#             nll = 0.5 * (logdet + quad + len(gp.Z) * np.log(2 * np.pi))
#             return nll
#         except np.linalg.LinAlgError:
#             return np.inf

#     # Initial parameter values - use geometric mean of bounds
#     d0 = np.sqrt(d_range[0] * d_range[1])
#     g0 = np.sqrt(g_range[0] * g_range[1])
    
#     # Optimize
#     result = minimize(
#         neg_log_likelihood,
#         x0=[d0, g0],
#         method='L-BFGS-B',
#         bounds=[d_range, g_range],
#         options={'maxiter': 100, 'disp': verb > 0}
#     )
    
#     # Update GP with optimal parameters
#     if result.success:
#         gp.d = result.x[0]
#         gp.g = result.x[1]
#         gp.update_covariance()
    
#     return MLEResult(
#         lengthscale=result.x[0],
#         nugget=result.x[1],
#         iterations=result.nit,
#         success=result.success,
#         llik=-result.fun
#     )

# def estimate_initial_params(X: np.ndarray, Z: np.ndarray) -> Tuple[float, float]:
#     """
#     Estimate initial lengthscale and nugget parameters
    
#     Args:
#         X: Input locations
#         Z: Output values
        
#     Returns:
#         Tuple of (lengthscale, nugget) estimates
#     """
#     # Estimate lengthscale using median distance
#     dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
#     d = np.median(dists[dists > 0])
    
#     # Estimate nugget using output variance
#     z_std = np.std(Z)
#     g = (0.01 * z_std)**2  # Start with 1% of variance
    
#     return d, g

def closest_indices(start: int, Xref: np.ndarray, n: int, X: np.ndarray, 
                   close: int, sorted: bool = False) -> np.ndarray:
    """
    Returns the close indices into X which are closest to Xref.
    
    Args:
        start: Number of initial points
        Xref: Reference points
        n: Number of total points
        X: Input points
        close: Number of close points to find
        sorted: Whether to sort the indices
        
    Returns:
        Array of indices of closest points
    """
    # Ensure Xref is 2D
    if len(Xref.shape) == 1:
        Xref = Xref.reshape(1, -1)

    # Calculate distances to reference location(s)
    D = distance_asymm(X, Xref)
    # D = np.zeros(n)
    # for i in range(Xref.shape[1]):
    #     diff = Xref[:, i:i+1] - X[:, i].reshape(1, -1)
    #     D += np.min(diff**2, axis=0)  # Take minimum across reference points

    # Get indices of closest points
    if n > close:
        idx = np.argsort(D)[:close]
    else:
        idx = np.arange(n)
        
    # Sort by distance if requested
    if sorted:
        idx = idx[np.argsort(D[idx].reshape(-1))]
    elif start < close:
        # Partially sort to get start closest
        idx = np.argpartition(D[idx].reshape(-1), start)
        
    return idx

def _laGP(Xref: np.ndarray, 
         start: int, 
         end: int, 
         X: np.ndarray, 
         Z: np.ndarray, 
         d: Optional[Union[float, Tuple[float, float]]] = None,
         g: float = 1/10000,
         method: Method = Method.ALC,  # Use Method enum
         close: Optional[int] = None,
         numstart: Optional[int] = None,
         rect: Optional[np.ndarray] = None,
         lite: bool = True,
         verb: int = 0) -> Tuple[float, float, float, float, float]:
    """
    Local Approximate GP prediction with parameter estimation
    
    Args:
        start: Number of initial points
        end: Number of total points to select
        Xref: Reference points for prediction
        n: Number of total input points
        X: Input points
        Z: Output values
        d: Initial length scale (if None, will be estimated)
        g: Initial nugget (if None, will be estimated)
        method: Method for selecting points
        close: Number of close points to consider
        param_est: Whether to estimate parameters using MLE
        d_range: (min, max) range for lengthscale optimization
        g_range: (min, max) range for nugget optimization
        est_freq: How often to re-estimate parameters
        alc_gpu: Whether to use GPU for ALC calculations
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
    m = X.shape[1]
    nref = Xref.shape[0]

    # Get closest points for initial design
    idx = closest_indices(start, Xref, n, X, close, 
                         method in ["alcray", "alcopt"])
    # Setup candidate points
    cand_idx = idx[start:]
    Xcand = X[cand_idx]
    selected = np.zeros(end, dtype=int)
    selected[:start] = idx[:start]

    # Build initial GP
    X_init = X[idx[:start]]
    Z_init = Z[idx[:start]]
    
    gp = new_gp(X_init, Z_init, get_value(d, 'start'), get_value(g, 'start'))
    
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
        else:  # Method.NN
            w = i - start
            
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
    if get_value(d, 'mle') and get_value(g, 'mle'):
        if gp.dK is None:
            gp.new_dK()
        gp.jmle(drange = (get_value(d, 'min'), get_value(d, 'max')), 
                grange = (get_value(g, 'min'), get_value(g, 'max')), 
                dab = get_value(d, 'ab'), 
                gab = get_value(g, 'ab'), 
                verb = verb)
    elif get_value(d, 'mle'):
        if gp.dK is None:
            gp.new_dK()
        gp.mle('lengthscale', get_value(d, 'min'), get_value(d, 'max'), 
               get_value(d, 'ab'), verb)
    elif get_value(g, 'mle'):
        gp.mle('nugget', get_value(g, 'min'), get_value(g, 'max'), 
               get_value(g, 'ab'), verb)

    # Given the updated gp, predict values and return results
    if lite:
        mean_pred, s2_pred, df, llik = gp.predict_lite(Xref)
    else:
        mean_pred, s2_pred, df, llik = gp.predict(Xref)
    
    return {
        "mean": mean_pred,
        "s2": s2_pred,
        "df": df,
        "llik": llik,
        "selected": selected,
        "d_posterior": gp.d,
        "g_posterior": gp.g,
    }

def laGP(Xref: np.ndarray, 
         start: int, 
         end: int, 
         X: np.ndarray, 
         Z: np.ndarray, 
         d: Optional[Union[float, Tuple[float, float]]] = None,
         g: float = 1/10000,
         method: str = "alc",
         Xi_ret: bool = True,
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
        start: Initial design size (must be >= 6)
        end: Final design size
        X: Training inputs (n × m)
        Z: Training outputs (n,)
        d: Lengthscale parameter or tuple of (start, mle)
        g: Nugget parameter
        method: One of "alc", "alcopt", "alcray", "mspe", "nn", "fish"
        Xi_ret: Whether to return selected indices
        close: Number of close points to consider
        alc_gpu: Whether to use GPU for ALC calculations
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
            Xi: Selected indices (if Xi_ret=True)
    """
    # Method mapping
    method_map = {
        "alc": 1, "alcopt": 2, "alcray": 3,
        "mspe": 4, "fish": 5, "nn": 6
    }
    
    # Input processing
    method = method.lower()
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")
    imethod = method_map[method]
    
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    Z = np.asarray(Z)
    Xref = np.atleast_2d(Xref)
    
    # Get dimensions
    m = X.shape[1]
    n = X.shape[0]
    nref = Xref.shape[0]
    
    # Input validation
    if start < 6 or end <= start:
        raise ValueError("must have 6 <= start < end")
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
    
    # Initialize output arrays
    mean = np.zeros(nref)
    s2dim = nref if lite else nref * nref
    s2 = np.zeros(s2dim)
    Xi = np.zeros(end, dtype=int) if Xi_ret else None
    
    # Start timing
    tic = time.time()
    
    # Call core implementation
    results = _laGP(Xref=Xref,
        start=start, end=end, X=X, Z=Z,        
        d=d_prior, g=g_prior,
        method=Method(imethod),
        close=close,
        numstart=numstart,
        rect=rect,
        verb=verb,
        lite=lite
    )
    
    # Assemble results
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
    
    # Add s2/Sigma
    if lite:
        result['s2'] = s2
    else:
        result['Sigma'] = s2.reshape(nref, nref)
    
    # Add Xi if requested
    if Xi_ret:
        result['Xi'] = Xi
    
    # Add ray info if needed
    if method in ["alcray", "alcopt"]:
        result['numstart'] = numstart
    
    return result

def alc(gp, Xcand, Xref, verb=0):
    """
    CPU implementation of ALC criterion.
    
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
    
    # Allocate vectors
    gvec = np.zeros(n)
    kxy = np.zeros(nref)
    kx = np.zeros(n)
    ktKikx = np.zeros(nref)
    
    # k <- covar(X1=X, X2=Xref, d=Zt$d, g=0)
    k = covar_sep(m, Xref, Xref.shape[0], gp.X, gp.X.shape[0], gp.d)
    
    # Initialize ALC scores
    alc_scores = np.zeros(ncand)
    
    # Calculate the ALC for each candidate
    for i in range(ncand):
        if verb > 0:
            print(f"alc: calculating ALC for point {i+1} of {ncand}")
        
        # Calculate the g vector, mui, and kxy
        mui, gvec, kx, kxy = calc_g_mui_kxy(m, Xcand[i], gp.X, gp.X.shape[0], gp.Ki, Xref, Xref.shape[0], gp.d, gp.g)
        
        # Skip if numerical problems
        if mui <= np.finfo(float).eps:
            alc_scores[i] = -np.inf
            continue
        
        # Use g, mu, and kxy to calculate ktKik.x
        ktKikx = calc_ktKikx(None, nref, k, gp.X.shape[0], gvec, mui, kxy, None, None)
        
        # Calculate the ALC
        alc_scores[i] = calc_alc(nref, ktKikx, [0, 0], gp.phi, df)
    
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
    ktKikx = np.zeros(m)

    if Gmui is not None:
        Gmui = np.outer(g, g) / mui
        assert ktGmui is not None

    for i in range(m):

        if k[i].ndim > 1:
            k[i] = k[i].flatten()
        if g[i].ndim > 1:
            g[i] = g[i].flatten()

        if k[i].shape[0] != g.shape[0]:
            raise ValueError(f"Dimension mismatch: k[{i}].shape = {k[i].shape}, g.shape = {g.shape}")

        if Gmui is not None:
            ktGmui = np.dot(Gmui, k[i])
            if ktKik is not None:
                ktKikx[i] = ktKik[i] + np.dot(ktGmui, k[i])
            else:
                ktKikx[i] = np.dot(ktGmui, k[i])
        else:
            if ktKik is not None:
                ktKikx[i] = ktKik[i] + (np.dot(k[i], g) ** 2) * mui
            else:
                ktKikx[i] = (np.dot(k[i], g) ** 2) * mui

        # Add 2*diag(kxy %*% t(g) %*% k)
        ktKikx[i] += 2.0 * np.dot(k[i], g) * kxy[i]

        # Add kxy^2/mui
        ktKikx[i] += (kxy[i] ** 2) / mui

    return ktKikx

def calc_alc(m, ktKik, s2p, phi, tdf, badj=None, w=None):
    """
    Calculate the Active Learning Criterion (ALC).
    
    Args:
        m: Number of points
        ktKik: Array of ktKik values (1D numpy array)
        s2p: Array of s2p values (1D numpy array)
        phi: Scalar value
        badj: Optional array of adjustment factors (1D numpy array)
        tdf: Degrees of freedom
        w: Optional array of weights (1D numpy array)
        
    Returns:
        ALC value
    """
    dfrat = tdf / (tdf - 2.0)
    alc = 0.0
    
    for i in range(m):
        zphi = (s2p[1] + phi) * ktKik[i]
        if badj is not None:
            ts2 = badj[i] * zphi / (s2p[0] + tdf)
        else:
            ts2 = zphi / (s2p[0] + tdf)
        
        if w is not None:
            alc += w[i] * dfrat * ts2
        else:
            alc += ts2 * dfrat
    
    return alc / m



