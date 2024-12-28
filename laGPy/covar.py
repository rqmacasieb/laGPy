import numpy as np
from typing import Optional, Tuple
from .utils.distance import distance

def covar(X1: np.ndarray, X2: np.ndarray, d: float) -> np.ndarray:
    """
    Calculate covariance between two sets of points
    
    Args:
        X1: First set of points
        X2: Second set of points
        d: Length scale parameter
        
    Returns:
        Covariance matrix
    """
    D = distance(X1, X2)
    K = np.exp(-D / d)
    return K

def covar_symm(X: np.ndarray, d: float, g: float) -> np.ndarray:
    """
    Calculate symmetric covariance matrix for one set of points
    
    Args:
        X: Input points
        d: Length scale parameter
        g: Nugget parameter
        
    Returns:
        Symmetric covariance matrix
    """
    K = covar(X, X, d)
    # np.fill_diagonal(K, 1.0 + g)
    K.flat[::K.shape[0] + 1] += g
    return K

def calc_g_mui_kxy(col, x, X, n, Ki, Xref, nref, d, g):
    """
    Calculate the g vector, mui, and kxy for the IECI calculation.
    
    Args:
        col: Number of columns (features)
        x: Single input point (1D numpy array)
        X: Input data matrix (2D numpy array)
        n: Number of data points in X
        Ki: Inverse of the covariance matrix of X (2D numpy array)
        Xref: Reference data matrix (2D numpy array)
        nref: Number of reference points
        d: Range parameters (1D numpy array)
        g: Nugget parameter
        
    Returns:
        Tuple of (mui, gvec, kxy)
    """
    # Calculate kx: covariance between x and each point in X
    kx = covar(X, x, d).ravel()
    
    # Calculate kxy: covariance between x and each point in Xref
    kxy = covar(x, Xref, d).ravel() if nref > 0 else None

    # Calculate gvec: Ki * kx
    gvec = Ki @ kx
    
    # Calculate mui: 1 + g - kx' * gvec
    mui = 1.0 + g - kx @ gvec
    
    # Calculate gvec: - Kikx/mui
    gvec *= -1.0 / mui

    return mui, gvec, kxy

def diff_covar_symm(X: np.ndarray, d: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the first and second derivatives of a symmetric covariance matrix.
    
    Args:
        X: Data matrix (2D array).
        d: Lengthscale parameter.
        
    Returns:
        dK: First derivative of the covariance matrix (2D array).
        d2K: Second derivative of the covariance matrix (2D array).
    """
    d2 = d**2
    n = X.shape[0]
    dK = np.zeros((n, n))
    d2K = np.zeros((n, n))
    
    # Calculate pairwise distances using the imported distance function
    D = distance(X, X)
    

    # Calculate the covariance derivatives
    for i in range(n):
        for j in range(i + 1, n):
            dist = D[i, j]
            dK[i, j] = dK[j, i] = dist * np.exp(-dist / d) / d2
            d2K[i, j] = d2K[j, i] = dK[i, j] * (dist - 2.0 * d) / d2

        dK[i, i] = 0.0
        d2K[i, i] = 0.0
            
    return dK, d2K

