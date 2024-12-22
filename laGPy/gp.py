import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .matrix import *
from .covar import covar, covar_symm

@dataclass
class GP:
    """Gaussian Process class"""
    X: np.ndarray           # Design matrix
    K: np.ndarray           # Covariance between design points
    Ki: np.ndarray          # Inverse of K
    dK: Optional[np.ndarray] = None    # First derivative of K (optional)
    d2K: Optional[np.ndarray] = None   # Second derivative of K (optional)
    ldetK: float = 0.0      # Log determinant of K
    Z: np.ndarray = None    # Response vector
    KiZ: np.ndarray = None  # Ki @ Z
    d: float = 1.0         # Lengthscale parameter
    g: float = 0.0         # Nugget parameter
    phi: float = 0.0       # t(Z) @ Ki @ Z
    F: float = 0.0         # Approx Fisher info (optional with dK)

    @property
    def m(self) -> int:
        """Number of columns in X"""
        return self.X.shape[1]

    @property
    def n(self) -> int:
        """Number of rows in X"""
        return self.X.shape[0]

def new_gp(X: np.ndarray, Z: np.ndarray, d: float, g: float, 
           compute_derivs: bool = False) -> GP:
    """
    Create a new Gaussian Process
    
    Args:
        X: Design matrix
        Z: Response vector
        d: Length scale parameter
        g: Nugget parameter
        compute_derivs: Whether to compute derivatives
        
    Returns:
        New GP instance
    """
    n, m = X.shape
    
    # Calculate covariance matrix
    K = covar_symm(X, d, g)
    
    # Calculate inverse and log determinant
    L = np.linalg.cholesky(K)
    Ki = np.linalg.inv(K)
    ldetK = 2 * np.sum(np.log(np.diag(L)))
    
    # Calculate KiZ
    KiZ = Ki @ Z
    
    # Calculate phi
    phi = Z @ KiZ
    
    gp = GP(X=X, K=K, Ki=Ki, Z=Z, KiZ=KiZ, 
            ldetK=ldetK, d=d, g=g, phi=phi)
    
    if compute_derivs:
        # Add derivative calculations here if needed
        pass
        
    return gp

def pred_gp(gp: GP, XX: np.ndarray, include_nugget: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions at new locations
    
    Args:
        gp: GP instance
        XX: Prediction locations
        include_nugget: Whether to include nugget in predictions
        
    Returns:
        Tuple of (mean, variance) predictions
    """
    k = covar(XX, gp.X, gp.d)
    mean = k @ gp.KiZ
    
    if include_nugget:
        var = np.ones(XX.shape[0]) * (1 + gp.g) - np.sum(k @ gp.Ki * k, axis=1)
    else:
        var = np.ones(XX.shape[0]) - np.sum(k @ gp.Ki * k, axis=1)
        
    return mean, var