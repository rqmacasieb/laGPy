import numpy as np

def covar_sep_symm(col, X, n, d, g):
    """
    Calculate the symmetric covariance matrix with a separable power exponential correlation function.
    
    Args:
        col: Number of columns (features)
        X: Input data matrix (2D numpy array)
        n: Number of data points
        d: Range parameters (1D numpy array)
        g: Nugget parameter
        
    Returns:
        Symmetric covariance matrix (2D numpy array)
    """
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 1.0 + g
        for j in range(i + 1, n):
            K[i, j] = np.sum((X[i] - X[j]) ** 2 / d)
            K[i, j] = np.exp(-K[i, j])
            K[j, i] = K[i, j]
    return K

def covar_sep(col, X1, n1, X2, n2, d):
    """
    Calculate the covariance matrix between X1 and X2 with a separable power exponential correlation function.
    
    Args:
        col: Number of columns (features)
        X1: First input data matrix (2D numpy array)
        n1: Number of data points in X1
        X2: Second input data matrix (2D numpy array)
        n2: Number of data points in X2
        d: Range parameters (1D numpy array)
        
    Returns:
        Covariance matrix (2D numpy array)
    """
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = np.sum((X1[i] - X2[j]) ** 2 / d)
            K[i, j] = np.exp(-K[i, j])
    return K

def diff_covar_sep(col, X1, n1, X2, n2, d, K):
    """
    Calculate the first derivative of the covariance matrix with respect to d.
    
    Args:
        col: Number of columns (features)
        X1: First input data matrix (2D numpy array)
        n1: Number of data points in X1
        X2: Second input data matrix (2D numpy array)
        n2: Number of data points in X2
        d: Range parameters (1D numpy array)
        K: Covariance matrix (2D numpy array)
        
    Returns:
        Derivative of covariance matrix (3D numpy array)
    """
    dK = np.zeros((col, n1, n2))
    for k in range(col):
        d2k = d[k] ** 2
        for i in range(n1):
            for j in range(n2):
                dK[k, i, j] = K[i, j] * ((X1[i, k] - X2[j, k]) ** 2) / d2k
    return dK

def diff_covar_sep_symm(col, X, n, d, K):
    """
    Calculate the first derivative of the symmetric covariance matrix with respect to d.
    
    Args:
        col: Number of columns (features)
        X: Input data matrix (2D numpy array)
        n: Number of data points
        d: Range parameters (1D numpy array)
        K: Covariance matrix (2D numpy array)
        
    Returns:
        Derivative of covariance matrix (3D numpy array)
    """
    dK = np.zeros((col, n, n))
    for k in range(col):
        d2k = d[k] ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dK[k, i, j] = K[i, j] * ((X[i, k] - X[j, k]) ** 2) / d2k
                dK[k, j, i] = dK[k, i, j]
            dK[k, i, i] = 0.0
    return dK

