# lagp.py
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict, NamedTuple

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

def laGP(m: int, start: int, end: int, Xref: np.ndarray, n: int, X: np.ndarray, 
         Z: np.ndarray, d: Optional[float] = None, g: Optional[float] = None, 
         method: Method = Method.ALC, close: int = 0,
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
        m: Input dimension
        start: Number of initial points
        end: Number of total points to select
        Xref: Reference points for prediction
        n: Number of total points
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
    # Get closest points for initial design
    idx = closest_indices(m, start, Xref, n, X, close, 
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
    ncand = len(cand_idx)
    
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
            scores = alc_gpu(gp, Xcand, Xref, verb) if alc_gpu else alc_cpu(gp, Xcand, Xref, verb)
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
        if w < ncand - 1:
            if method in (Method.ALCRAY, Method.ALCOPT):
                cand_idx = np.delete(cand_idx, w)
                Xcand = np.delete(Xcand, w, axis=0)
            else:
                cand_idx[w] = cand_idx[-1]
                Xcand[w] = Xcand[-1]
        ncand -= 1
    
    # Final predictions
    mean, var = pred_gp(gp, Xref)
    
    return mean, var, selected, gp.d, gp.g