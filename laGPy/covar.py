import numpy as np
from typing import Optional, Tuple
from .utils.distance import distance
# def distance(X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
#     """
#     Calculate pairwise distances between points
    
#     Args:
#         X1: First set of points
#         X2: Optional second set of points (if None, use X1)
        
#     Returns:
#         Matrix of pairwise distances
#     """
#     if X2 is None:
#         X2 = X1
        
#     n1, m = X1.shape
#     n2 = X2.shape[0]
    
#     D = np.zeros((n1, n2))
#     for i in range(m):
#         D += (X1[:, i:i+1] - X2[:, i].reshape(1, -1))**2
#     return D

def covar(X1: np.ndarray, X2: np.ndarray, d: float, g: float = 0.0) -> np.ndarray:
    """
    Calculate covariance between two sets of points
    
    Args:
        X1: First set of points
        X2: Second set of points
        d: Length scale parameter
        g: Nugget parameter
        
    Returns:
        Covariance matrix
    """
    D = distance(X1, X2)
    K = np.exp(-D / (2 * d * d))
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
    np.fill_diagonal(K, 1.0 + g)
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
        Tuple of (mui, gvec, kx, kxy)
    """
    # Calculate kx: covariance between x and each point in X
    kx = np.exp(-0.5 * np.sum(((X - x) / d) ** 2, axis=1))
    
    # Calculate kxy: covariance between x and each point in Xref
    kxy = np.exp(-0.5 * np.sum(((Xref - x) / d) ** 2, axis=1))
    
    # Calculate gvec: Ki * kx
    gvec = Ki @ kx
    
    # Calculate mui: 1 + g - kx' * gvec
    mui = 1.0 + g - np.dot(kx, gvec)
    
    # Calculate gvec: - Kikx/mui
    gvec *= -1.0 / mui

    return mui, gvec, kx, kxy

def diff_covar_symm(X: np.ndarray, d: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate first and second derivatives of the symmetric covariance matrix.
    
    Args:
        X: Input matrix (n x m)
        d: Length scale parameter
        
    Returns:
        Tuple of (dK, d2K) - first and second derivatives
    """
    n = X.shape[0]
    dK = np.zeros((n, n))
    d2K = np.zeros((n, n))
    
    # Calculate pairwise distances
    for i in range(n):
        for j in range(i+1):
            dist = np.sum((X[i] - X[j])**2)
            exp_term = np.exp(-0.5 * dist / d**2)
            
            # First derivative
            dK[i,j] = exp_term * dist / d**3
            dK[j,i] = dK[i,j]  # symmetric
            
            # Second derivative
            d2K[i,j] = exp_term * dist * (dist - 3*d**2) / d**6
            d2K[j,i] = d2K[i,j]  # symmetric
            
    return dK, d2K

