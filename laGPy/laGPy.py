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

class Method(Enum):
    ALC = 1
    ALCOPT = 2 
    ALCRAY = 3
    MSPE = 4
    EFI = 5
    NN = 6

class MLEResult(NamedTuple):
    """Results from MLE optimization"""
    lengthscale: float  # Optimized lengthscale
    nugget: float      # Optimized nugget
    iterations: int    # Number of iterations
    success: bool      # Whether optimization succeeded
    llik: float       # Log likelihood at optimum

def joint_mle_gp(gp: GP, 
                d_range: Tuple[float, float] = (1e-6, 1.0),
                g_range: Tuple[float, float] = (1e-6, 1.0),
                verb: int = 0) -> MLEResult:
    """
    Joint maximum likelihood estimation for GP lengthscale and nugget
    
    Args:
        gp: GP instance
        d_range: (min, max) range for lengthscale
        g_range: (min, max) range for nugget
        verb: Verbosity level
        
    Returns:
        MLEResult containing optimized values and optimization info
    """
    def neg_log_likelihood(theta):
        """Negative log likelihood function for joint optimization"""
        d, g = theta
        
        # Update GP parameters
        gp.d = d
        gp.g = g
        gp.update_covariance()
        
        try:
            # Log determinant term
            sign, logdet = np.linalg.slogdet(gp.K)
            if sign <= 0:
                return np.inf
                
            # Quadratic term
            alpha = np.linalg.solve(gp.K, gp.Z)
            quad = np.dot(gp.Z, alpha)
            
            # Full negative log likelihood
            nll = 0.5 * (logdet + quad + len(gp.Z) * np.log(2 * np.pi))
            return nll
        except np.linalg.LinAlgError:
            return np.inf

    # Initial parameter values - use geometric mean of bounds
    d0 = np.sqrt(d_range[0] * d_range[1])
    g0 = np.sqrt(g_range[0] * g_range[1])
    
    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0=[d0, g0],
        method='L-BFGS-B',
        bounds=[d_range, g_range],
        options={'maxiter': 100, 'disp': verb > 0}
    )
    
    # Update GP with optimal parameters
    if result.success:
        gp.d = result.x[0]
        gp.g = result.x[1]
        gp.update_covariance()
    
    return MLEResult(
        lengthscale=result.x[0],
        nugget=result.x[1],
        iterations=result.nit,
        success=result.success,
        llik=-result.fun
    )

def estimate_initial_params(X: np.ndarray, Z: np.ndarray) -> Tuple[float, float]:
    """
    Estimate initial lengthscale and nugget parameters
    
    Args:
        X: Input locations
        Z: Output values
        
    Returns:
        Tuple of (lengthscale, nugget) estimates
    """
    # Estimate lengthscale using median distance
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    d = np.median(dists[dists > 0])
    
    # Estimate nugget using output variance
    z_std = np.std(Z)
    g = (0.01 * z_std)**2  # Start with 1% of variance
    
    return d, g

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
    D = np.zeros(n)
    for i in range(Xref.shape[1]):
        diff = Xref[:, i:i+1] - X[:, i].reshape(1, -1)
        D += np.min(diff**2, axis=0)  # Take minimum across reference points

    # Get indices of closest points
    if n > close:
        idx = np.argsort(D)[:close]
    else:
        idx = np.arange(n)
        
    # Sort by distance if requested
    if sorted:
        idx = idx[np.argsort(D[idx])]
    elif start < close:
        # Partially sort to get start closest
        idx = np.argpartition(D[idx], start)
        
    return idx

def laGP(start: int, end: int, Xref: np.ndarray, X: np.ndarray, 
         Z: np.ndarray, n: Optional[int] = None, d: Optional[float] = None, g: Optional[float] = 1/10000, 
         method: Method = Method.ALC, close: Optional[int] = None,
         param_est: bool = True, 
         d_range: Tuple[float, float] = (1e-6, 1.0),
         g_range: Tuple[float, float] = (1e-6, 1.0),
         est_freq: int = 10,  # How often to re-estimate parameters
         alc_gpu: bool = False, numstart: int = 1, 
         rect: Optional[np.ndarray] = None,
         verb: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
        verb: Verbosity level
        
    Returns:
        Tuple of:
        - Mean predictions
        - Variance predictions
        - Selected indices
        - Final length scale
        - Final nugget
    """
    if n is None:
        n = X.shape[0]

    if close is None:
        close = min((1000 + end) * (10 if method in [Method.ALCRAY, Method.ALCOPT] else 1), X.shape[0])

    #check input dimension
    if Xref.shape[1] != X.shape[1]:
        raise ValueError(f"Dimension mismatch: Xref.shape = {Xref.shape}, X.shape = {X.shape}")

    # Get closest points for initial design
    idx = closest_indices(start, Xref, n, X, close, 
                         method in (Method.ALCRAY, Method.ALCOPT))
    
    # Initial data
    X_init = X[idx[:start]]
    Z_init = Z[idx[:start]]
    
    # Initial parameter estimates if not provided
    if d is None or g is None:
        d_est, g_est = estimate_initial_params(X_init, Z_init)
        d = d_est if d is None else d
        g = g_est if g is None else g
        
        if verb > 0:
            print(f"Initial estimates: lengthscale={d:.6f}, nugget={g:.6f}")
    
    # Build initial GP
    gp = new_gp(X_init, Z_init, d, g)
    
    # Initial parameter optimization if requested
    if param_est:
        mle_result = joint_mle_gp(gp, d_range, g_range, verb=verb-1)
        if verb > 0:
            print(f"MLE results: lengthscale={mle_result.lengthscale:.6f}, "
                  f"nugget={mle_result.nugget:.6f} "
                  f"(iterations: {mle_result.iterations})")
    
    # Setup candidate points
    cand_idx = idx[start:]
    Xcand = X[cand_idx]
    
    # Get rect bounds if needed
    if method in (Method.ALCRAY, Method.ALCOPT) and rect is None:
        rect = get_data_rect(Xcand)
        
    # Storage for selected indices
    selected = np.zeros(end, dtype=int)
    selected[:start] = idx[:start]
    
    # Iteratively select points
    for i in range(start, end):
        # Point selection logic based on method
        if method == Method.ALCRAY:
            roundrobin = (i - start + 1) % int(np.sqrt(i - start + 1))
            w = alcray_selection(gp, Xcand, Xref, roundrobin, numstart, rect, verb)
        elif method == Method.ALC:
            scores = alc(gp, Xcand, Xref, verb) #no gpu support for now
            w = np.argmax(scores)
        elif method == Method.MSPE:
            scores = mspe(gp, Xcand, Xref, verb)
            w = np.argmin(scores)
        else:  # Method.NN
            w = i - start
            
        # Record chosen point
        selected[i] = cand_idx[w]
        
        # Update GP
        gp = update_gp(gp, Xcand[w:w+1], Z[cand_idx[w:w+1]])
        
        # Re-estimate parameters periodically if requested
        if param_est and (i - start + 1) % est_freq == 0:
            mle_result = joint_mle_gp(gp, d_range, g_range, verb=verb-1)
            if verb > 0:
                print(f"Update {i}: lengthscale={mle_result.lengthscale:.6f}, "
                      f"nugget={mle_result.nugget:.6f}")
        
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
    
    # Final predictions
    mean, var = pred_gp(gp, Xref)
    
    return mean, var, selected, gp.d, gp.g

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

#these are placeholder functions for now. Will develop these later if needed.
def mspe(gp: GP, Xcand: np.ndarray, Xref: np.ndarray, verb: int = 0) -> np.ndarray:
    """Calculate MSPE criterion"""
    # Implementation of MSPE calculations
    pass

def alcray_selection(gp: GP, Xcand: np.ndarray, Xref: np.ndarray, 
                    roundrobin: int, numstart: int, rect: np.ndarray,
                    verb: int = 0) -> int:
    """ALCRAY point selection"""
    # Implementation of ALCRAY selection
    pass

