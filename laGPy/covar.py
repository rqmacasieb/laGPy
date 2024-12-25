import numpy as np
from typing import Optional

def distance(X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate pairwise distances between points
    
    Args:
        X1: First set of points
        X2: Optional second set of points (if None, use X1)
        
    Returns:
        Matrix of pairwise distances
    """
    if X2 is None:
        X2 = X1
        
    n1, m = X1.shape
    n2 = X2.shape[0]
    
    D = np.zeros((n1, n2))
    for i in range(m):
        D += (X1[:, i:i+1] - X2[:, i].reshape(1, -1))**2
    return D

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