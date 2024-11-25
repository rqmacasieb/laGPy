import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve, cholesky
from typing import Tuple, Optional, List

class LaGP:
    def __init__(self, kernel="gaussian"):
        """
        Initialize Local Approximate Gaussian Process
        
        Args:
            kernel (str): Kernel function to use ("gaussian" or "matern")
        """
        self.kernel = kernel
        
    def _kernel_fn(self, X1: np.ndarray, X2: np.ndarray, 
                   length_scale: float, nugget: float = 0.0) -> np.ndarray:
        """
        Compute kernel matrix between X1 and X2
        
        Args:
            X1: First set of points (n1 x d)
            X2: Second set of points (n2 x d)
            length_scale: Length scale parameter
            nugget: Nugget parameter for numerical stability
            
        Returns:
            Kernel matrix (n1 x n2)
        """
        dist = cdist(X1, X2)
        if self.kernel == "gaussian":
            K = np.exp(-0.5 * (dist / length_scale) ** 2)
        else:  # matern
            K = (1 + np.sqrt(3) * dist / length_scale) * \
                np.exp(-np.sqrt(3) * dist / length_scale)
        
        # Add nugget to diagonal if X1 and X2 are the same
        if X1 is X2:
            K += nugget * np.eye(X1.shape[0])
        return K
    
    def _find_closest_points(self, X: np.ndarray, Xref: np.ndarray, 
                           n_close: int) -> np.ndarray:
        """
        Find indices of closest points to reference points
        
        Args:
            X: Training points
            Xref: Reference points
            n_close: Number of closest points to find
            
        Returns:
            Indices of closest points
        """
        distances = cdist(Xref, X)
        return np.argsort(distances[0])[:n_close]
    
    def _compute_alc_score(self, X_cand: np.ndarray, X_ref: np.ndarray, 
                          K: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
        """
        Compute Active Learning Cohn (ALC) score for candidate points
        
        Args:
            X_cand: Candidate points
            X_ref: Reference points
            K: Current kernel matrix
            K_inv: Inverse of current kernel matrix
            
        Returns:
            ALC scores for each candidate point
        """
        scores = []
        for x in X_cand:
            k_star = self._kernel_fn(X_ref, x.reshape(1, -1), 
                                   self.length_scale, self.nugget)
            k_star_star = self._kernel_fn(x.reshape(1, -1), x.reshape(1, -1), 
                                        self.length_scale, self.nugget)
            
            # Compute variance reduction
            v1 = k_star_star - k_star.T @ K_inv @ k_star
            scores.append(v1[0, 0])
            
        return np.array(scores)
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, 
                   start: int = 6, end: Optional[int] = None, 
                   length_scale: float = 1.0, nugget: float = 1e-6, 
                   n_close: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit LaGP model and make predictions
        
        Args:
            X: Training input data (n_samples x n_features)
            y: Training target values (n_samples,)
            X_test: Test input data (n_test x n_features)
            start: Number of initial points
            end: Final number of points (default: start + 20)
            length_scale: Kernel length scale parameter
            nugget: Nugget parameter for numerical stability
            n_close: Number of close points to consider
            
        Returns:
            Tuple of (predictions mean, predictions variance)
        """
        self.length_scale = length_scale
        self.nugget = nugget
        
        if end is None:
            end = start + 20
            
        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)
        variances = np.zeros(n_test)
        
        # For each test point
        for i in range(n_test):
            # Find closest points to current test point
            closest_idx = self._find_closest_points(X, X_test[i:i+1], n_close)
            X_close = X[closest_idx]
            y_close = y[closest_idx]
            
            # Initialize with starting points
            active_idx = list(range(start))
            remaining_idx = list(range(start, n_close))
            
            X_active = X_close[active_idx]
            y_active = y_close[active_idx]
            
            # Sequentially add points
            while len(active_idx) < end and remaining_idx:
                # Compute current kernel matrix and its inverse
                K = self._kernel_fn(X_active, X_active, 
                                  self.length_scale, self.nugget)
                K_inv = solve(K, np.eye(K.shape[0]))
                
                # Compute ALC scores for remaining points
                X_cand = X_close[remaining_idx]
                scores = self._compute_alc_score(X_cand, X_active, K, K_inv)
                
                # Select best point
                best_idx = np.argmax(scores)
                new_idx = remaining_idx.pop(best_idx)
                active_idx.append(new_idx)
                
                # Update active set
                X_active = X_close[active_idx]
                y_active = y_close[active_idx]
            
            # Make prediction
            K = self._kernel_fn(X_active, X_active, 
                              self.length_scale, self.nugget)
            k_star = self._kernel_fn(X_active, X_test[i:i+1], 
                                   self.length_scale)
            
            # Compute predictive mean and variance
            K_inv = solve(K, np.eye(K.shape[0]))
            predictions[i] = k_star.T @ K_inv @ y_active
            variances[i] = (1.0 + self.nugget - 
                          k_star.T @ K_inv @ k_star)[0, 0]
            
        return predictions, variances
