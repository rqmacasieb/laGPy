import numpy as np
from typing import Optional, Tuple, List, Callable
import warnings

def get_data_rect(X: np.ndarray) -> np.ndarray:
    """
    Compute and return the rectangle implied by the X data
    
    Args:
        X: Input matrix of shape (N, d)
        
    Returns:
        Rectangle bounds of shape (2, d) containing min/max for each dimension
    """
    return np.vstack([X.min(axis=0), X.max(axis=0)])

def new_matrix(n1: int, n2: int) -> np.ndarray:
    """Create a new n1 x n2 matrix"""
    if n1 == 0 or n2 == 0:
        return None
    return np.zeros((n1, n2))

def new_zero_matrix(n1: int, n2: int) -> np.ndarray:
    """Create a new zero-filled n1 x n2 matrix"""
    return np.zeros((n1, n2))

def new_id_matrix(n: int) -> np.ndarray:
    """Create a new n x n identity matrix"""
    return np.eye(n)

def wmean_of_columns(M: np.ndarray, weight: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate weighted mean of columns
    
    Args:
        M: Input matrix
        weight: Optional weight vector
        
    Returns:
        Weighted mean of columns
    """
    if weight is None:
        return M.mean(axis=0)
    else:
        return np.average(M, axis=0, weights=weight)

def wcov_of_columns(M: np.ndarray, mean: np.ndarray, 
                   weight: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate weighted covariance of columns
    
    Args:
        M: Input matrix
        mean: Mean vector
        weight: Optional weight vector
        
    Returns:
        Weighted covariance matrix
    """
    n = M.shape[0]
    if weight is None:
        weight = np.ones(n) / n
    else:
        weight = weight / weight.sum()
        
    M_centered = M - mean[None, :]
    return (weight[:, None, None] * 
            M_centered[:, :, None] * 
            M_centered[:, None, :]).sum(axis=0)

def check_means(mean: np.ndarray, q1: np.ndarray, 
                median: np.ndarray, q2: np.ndarray) -> None:
    """
    Enforce that means should lie within the quantiles
    
    Args:
        mean: Mean values
        q1: First quartile values
        median: Median values
        q2: Third quartile values
    """
    mask = (mean > q2) | (mean < q1)
    if mask.any():
        warnings.warn(f"{mask.sum()} predictive means replaced with medians")
        mean[mask] = median[mask]