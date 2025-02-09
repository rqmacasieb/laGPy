import numpy as np
from typing import Tuple
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
    #using isotrophic Gaussian by default
    #TODO: perhaps add an option to use other kernel functions if needed
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

def calc_g_mui_kxy(Xcand, X, Ki, Xref, d, g):
    """
    Calculate the g vector, mui, and kxy for all candidate points.
    
    Args:
        Xcand: Candidate points (2D numpy array)
        X: Input data matrix (2D numpy array)
        Ki: Inverse of the covariance matrix of X (2D numpy array)
        Xref: Reference data matrix (2D numpy array)
        d: Range parameters (1D numpy array)
        g: Nugget parameter
        
    Returns:
        Tuple of (mui, gvec, kxy) for all candidate points
    """
    # Calculate kx: covariance between each candidate point and each point in X
    kx = covar(X, Xcand, d)  # Shape: (ncand, n)
    
    # Calculate kxy: covariance between each candidate point and each point in Xref
    kxy = covar(Xcand, Xref, d) if Xref.size > 0 else None  # Shape: (ncand, nref)

    # Calculate gvec: Ki * kx for each candidate point
    gvec = kx.T @ Ki  # Shape: (ncand, n)

    # Calculate mui: 1 + g - diag(kx @ gvec.T)
    mui = 1.0 + g - np.einsum('ij,ij->i', kx.T, gvec)  # Shape: (ncand,)

    # Calculate gvec: - Kikx/mui for each candidate point
    gvec = -gvec / mui[:, np.newaxis]  # Broadcasting to divide each row by corresponding mui

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
    
    # Calculate pairwise distances using the imported distance function
    D = distance(X, X)
    
    # Calculate first derivative
    dK = np.where(~np.eye(n, dtype=bool), 
                 D * np.exp(-D / d) / d2, 
                 0)
    
    # Calculate second derivative
    d2K = np.where(~np.eye(n, dtype=bool), 
                   dK * (D - 2.0 * d) / d2,
                   0)
    
    return dK, d2K

