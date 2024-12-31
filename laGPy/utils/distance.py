from typing import Optional, Union
import numpy as np

def distance(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Calculate the distance matrix between the rows of X1 and X2,
    or between X1 and itself when X2 is None.
    
    Args:
        X1: First input matrix (n1 x m)
        X2: Second input matrix (n2 × m) or None
    
    Returns:
        Distance matrix (n1 × n2) or (n1 × n1) if X2 is None
    
    Raises:
        ValueError: If input dimensions don't match
    """
    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    # Coerce arguments and extract dimensions
    X1_norm = np.sum(X1**2, axis=1)[:, np.newaxis]
    X2_norm = np.sum(X2**2, axis=1)
    
    # Use matrix multiplication for efficient computation
    cross_term = -2 * np.dot(X1, X2.T)
    
    # Combine terms using broadcasting
    D = X1_norm + X2_norm + cross_term
    
    # Ensure non-negative distances due to numerical precision
    D = np.maximum(D, 0)
    return D

# def distance_symm(X: np.ndarray) -> np.ndarray:
#     """
#     Optimized symmetric distance calculation.
#     Used internally when X2 is None.
    
#     Args:
#         X: Input matrix (n × m)
    
#     Returns:
#         Symmetric distance matrix (n × n)
#     """
#     return distance(X)

def distance_asymm(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Optimized asymmetric distance calculation.
    Used internally when X2 is provided.
    
    Args:
        X1: First input matrix (n1 × m)
        X2: Second input matrix (n2 × m)
    
    Returns:
        Distance matrix (n1 × n2)
    """
    return distance(X1, X2)

# Optional: Vectorized versions using NumPy's built-in functions
# def distance_vectorized(X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
#     """
#     Vectorized version of distance calculation using NumPy operations.
#     This version might be faster for large matrices.
    
#     Args:
#         X1: First input matrix (n1 × m)
#         X2: Second input matrix (n2 × m) or None
    
#     Returns:
#         Distance matrix
#     """
#     X1 = np.asarray(X1)
#     if X1.ndim == 1:
#         X1 = X1.reshape(-1, 1)
    
#     if X2 is None:
#         # Symmetric case
#         sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + \
#                    np.sum(X1**2, axis=1) - \
#                    2 * np.dot(X1, X1.T)
#     else:
#         # Asymmetric case
#         X2 = np.asarray(X2)
#         if X2.ndim == 1:
#             X2 = X2.reshape(-1, 1)
#         if X1.shape[1] != X2.shape[1]:
#             raise ValueError("Column dimension mismatch between X1 and X2")
            
#         sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + \
#                    np.sum(X2**2, axis=1) - \
#                    2 * np.dot(X1, X2.T)
    
#     # Ensure non-negative due to floating point errors
#     sq_dists = np.maximum(sq_dists, 0)
#     return np.sqrt(sq_dists)

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