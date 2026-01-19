import numpy as np
from typing import Tuple, Literal
from .utils.distance import distance

KernelType = Literal['squared_exponential', 'exponential', 'matern32', 'matern52']

def covar(X1: np.ndarray, X2: np.ndarray, d: float, kernel: KernelType = 'squared_exponential') -> np.ndarray:
    """
    Calculate covariance between two sets of points
    
    Args:
        X1: First set of points
        X2: Second set of points
        d: Length scale parameter
        kernel: Kernel type
        
    Returns:
        Covariance matrix
    """
    D_sq = distance(X1, X2) 

    if kernel == 'squared_exponential':
        K = np.exp(-D_sq / d)
    elif kernel == 'exponential':
        D = np.sqrt(np.maximum(D_sq, 0))
        K = np.exp(-D / d)
    elif kernel == 'matern32':
        D = np.sqrt(np.maximum(D_sq, 0))
        sqrt3_D_d = np.sqrt(3) * D / d
        K = (1 + sqrt3_D_d) * np.exp(-sqrt3_D_d)
    elif kernel == 'matern52':
        D = np.sqrt(np.maximum(D_sq, 0))
        sqrt5_D_d = np.sqrt(5) * D / d
        K = (1 + sqrt5_D_d + 5 * D**2 / (3 * d**2)) * np.exp(-sqrt5_D_d)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")
    
    return K

def covar_symm(X: np.ndarray, d: float, g: float, kernel: KernelType = 'squared_exponential') -> np.ndarray:
    """
    Calculate symmetric covariance matrix for one set of points
    
    Args:
        X: Input points
        d: Length scale parameter
        g: Nugget parameter
        kernel: Kernel type
    Returns:
        Symmetric covariance matrix
    """
    K = covar(X, X, d, kernel)
    # np.fill_diagonal(K, 1.0 + g)
    K.flat[::K.shape[0] + 1] += g
    return K

def calc_g_mui_kxy(Xcand, X, Ki, Xref, d, g, kernel: KernelType = 'squared_exponential'):
    """
    Calculate the g vector, mui, and kxy for all candidate points.
    
    Args:
        Xcand: Candidate points (2D numpy array)
        X: Input data matrix (2D numpy array)
        Ki: Inverse of the covariance matrix of X (2D numpy array)
        Xref: Reference data matrix (2D numpy array)
        d: Range parameters (1D numpy array)
        g: Nugget parameter
        kernel: Kernel type
    Returns:
        Tuple of (mui, gvec, kxy) for all candidate points
    """
    # Calculate kx: covariance between each candidate point and each point in X
    kx = covar(X, Xcand, d, kernel)  # Shape: (ncand, n)
    
    # Calculate kxy: covariance between each candidate point and each point in Xref
    kxy = covar(Xcand, Xref, d, kernel) if Xref.size > 0 else None  # Shape: (ncand, nref)

    # Calculate gvec: Ki * kx for each candidate point
    gvec = kx.T @ Ki  # Shape: (ncand, n)

    # Calculate mui: 1 + g - diag(kx @ gvec.T)
    mui = 1.0 + g - np.einsum('ij,ij->i', kx.T, gvec)  # Shape: (ncand,)

    # Calculate gvec: - Kikx/mui for each candidate point
    gvec = -gvec / mui[:, np.newaxis]  # Broadcasting to divide each row by corresponding mui

    return mui, gvec, kxy

def diff_covar_symm(X: np.ndarray, d: float, kernel: KernelType = 'squared_exponential') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the first and second derivatives of a symmetric covariance matrix.
    
    Args:
        X: Data matrix (2D array).
        d: Lengthscale parameter.
        kernel: Kernel type
    Returns:
        dK: First derivative of the covariance matrix (2D array).
        d2K: Second derivative of the covariance matrix (2D array).
    """
    d2 = d**2
    n = X.shape[0]
    
    # Calculate pairwise distances using the imported distance function
    D_sq = distance(X, X)  # Squared distances
    D = np.sqrt(np.maximum(D_sq, 0))  # Euclidean distances
    
    # Avoid division by zero for diagonal elements
    mask = ~np.eye(n, dtype=bool)
    
    if kernel == 'squared_exponential':
        exp_term = np.exp(-D_sq / d)
        dK = np.where(mask, D_sq * exp_term / d2, 0)
        d2K = np.where(mask, dK * (D_sq - 2.0 * d) / d2, 0)
    elif kernel == 'exponential':
        exp_term = np.exp(-D / d)
        dK = np.where(mask, D * exp_term / d2, 0)
        d2K = np.where(mask, dK * (D - 2.0 * d) / d2, 0)
    elif kernel == 'matern32':
        sqrt3 = np.sqrt(3)
        sqrt3_D_d = sqrt3 * D / d
        exp_term = np.exp(-sqrt3_D_d)
        
        dK = np.where(mask, sqrt3 * D * (1 + sqrt3_D_d) * exp_term / d2, 0)
        
        d2K = np.where(mask, sqrt3 * D * exp_term * (sqrt3 * (D - 2*d) + sqrt3_D_d * (sqrt3 * D - 2*d)) / (d2 * d), 0)
    elif kernel == 'matern52':
        sqrt5 = np.sqrt(5)
        sqrt5_D_d = sqrt5 * D / d
        D2_d2 = D**2 / d2
        exp_term = np.exp(-sqrt5_D_d)
        
        coeff1 = 1 + sqrt5_D_d + 5 * D2_d2 / 3
        dK = np.where(mask,
                      sqrt5 * D * exp_term * (sqrt5 / d + 10 * D / (3 * d2)) / d + 
                      coeff1 * sqrt5 * D * exp_term / d2,
                      0)
                      
        dK_term = sqrt5 * D * exp_term / d2
        d2K = np.where(mask,
                       dK_term * (sqrt5 * (D - 2*d) / d + 10 * (D - d) / (3 * d)) +
                       coeff1 * sqrt5 * D * exp_term * (sqrt5 * (D - 2*d)) / (d2 * d),
                       0)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")  
    
    return dK, d2K

