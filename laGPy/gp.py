import numpy as np
import pickle
import os
from scipy.stats import gamma
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Optional, Tuple, Union 
from .matrix import *
from .covar import *
from .params import *
from .utils.brent_fmin import brent_fmin
from .covar import KernelType

@dataclass
class OptInfo:
    """Structure to hold optimization information"""
    gp: 'GP'
    theta: str
    ab: Optional[Tuple[float, float]]
    its: int = 0
    verb: int = 0

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
    kernel: KernelType = 'squared_exponential'
    
    @property
    def m(self) -> int:
        """Number of columns in X"""
        return self.X.shape[1]

    @property
    def n(self) -> int:
        """Number of rows in X"""
        return self.X.shape[0]
    
    def update_covariance(self) -> None:
        """
        Update covariance matrix K and its inverse Ki based on current parameters
        Also updates ldetK (log determinant) and KiZ
        """
        # Calculate covariance matrix
        self.K = covar_symm(self.X, self.d, self.g, self.kernel)
        
        try:
            # Calculate Cholesky decomposition
            L = cho_factor(self.K)
        
            # inverse calculation
            self.Ki = cho_solve(L, np.eye(self.K.shape[0]))
            
            # log determinant calculation
            self.ldetK = 2 * np.sum(np.log(np.diag(L[0])))
            
            if self.Z is not None:
                # More efficient solve
                self.KiZ = cho_solve(L, self.Z)
                self.phi = self.Z @ self.KiZ
                
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular or not positive definite")

    def mle(self, theta: str, tmin: float, tmax: float, 
        ab: Optional[Tuple[float, float]], verb: int = 0) -> float:
        """
        Maximum Likelihood Estimation for GP parameters using Newton's method.
        
        Args:
            theta: Parameter to optimize ('lengthscale' or 'nugget')
            tmin: Minimum value for parameter
            tmax: Maximum value for parameter
            ab: Tuple of prior parameters (shape, scale)
            verb: Verbosity level
            
        Returns:
        th : float
            Optimized parameter value.
        its : int
            Number of iterations performed.
        """
        # Set priors based on theta
        dab = ab if theta == 'lengthscale' else None
        gab = ab if theta == 'nugget' else None
        
        # Initialization
        its = 0
        restored_dK = False
        
        # Get current parameter value
        th = self.d if theta == 'lengthscale' else self.g
        
        # Check if too close to tmin for nugget
        if theta == 'nugget' and abs(th - tmin) < np.finfo(float).eps:
            if verb > 0:
                print(f"(g={th}) -- starting too close to min ({tmin})")
            return th, its
        
        # Initial likelihood calculation
        llik_init = self.log_likelihood(dab, gab)
        
        # Initial printing
        if verb > 0:
            param_name = 'd' if theta == 'lengthscale' else 'g'
            print(f"({param_name}={th}, llik={llik_init}) ", end='')
            if verb > 1:
                print()
        
        while True:  # checking for improved llik
            while True:  # Newton step(s)
                llik_new = float('-inf')
                while True:  # Newton proposal
                    # Calculate derivatives
                    if theta == 'lengthscale':
                        dllik, d2llik = self.derivatives(theta, dab)
                    else:
                        dllik, d2llik = self.derivatives(theta, gab)
                    
                    # Check for convergence by root
                    if abs(dllik) < np.finfo(float).eps:
                        if its == 0:
                            if verb > 0:
                                print("-- Newton not needed")
                            return th, its
                        break  # goto newtondone
                    
                    # Newton update
                    rat = dllik/d2llik
                    adj = 1.0
                    its += 1
                    
                    # Check if we're going the right way
                    if (dllik < 0 and rat < 0) or (dllik > 0 and rat > 0):
                        if self.dK is None and not restored_dK:
                            self.delete_dK()
                            restored_dK = True
                        th = self.optimize(theta, tmin, tmax, ab, "[slip]", its, verb)
                        goto_mledone = True
                        break
                    else:
                        tnew = th - adj*rat  # right way: Newton
                    
                    # Check that we haven't proposed a tnew out of range
                    while (tnew <= tmin or tnew >= tmax) and adj > np.finfo(float).eps:
                        adj /= 2.0
                        tnew = th - adj*rat
                    
                    # If still out of range, restart
                    if tnew <= tmin or tnew >= tmax:
                        if self.dK is None and not restored_dK:
                            self.delete_dK()
                            restored_dK = True
                        th = self.optimize(theta, tmin, tmax, ab, "[range]", its, verb)
                        goto_mledone = True
                        break
                    else:
                        break  # exit Newton proposal loop
                
                if 'goto_mledone' in locals():
                    break
                
                # Update parameters
                if theta == 'lengthscale':
                    self.update_params(d=tnew, g=self.g)
                else:
                    if self.dK is None and not restored_dK:
                        self.delete_dK()
                        restored_dK = True
                    self.update_params(d=self.d, g=tnew)
                
                # Print progress
                if verb > 1:
                    print(f"\ti={its} theta={tnew}, c(a,b)=({ab[0]},{ab[1]})")
                
                # Check for convergence
                if abs(tnew - th) < np.finfo(float).eps:
                    break
                else:
                    th = tnew
                
                # Check for max iterations
                if its >= 100:
                    if verb > 0:
                        print("Warning: Newton 100/max iterations")
                    return th, its
            
            if 'goto_mledone' in locals():
                break
                
            # Check that we went in the right direction
            llik_new = self.log_likelihood(dab, gab)
            if llik_new < llik_init - np.finfo(float).eps:
                if verb > 0:
                    print(f"llik_new = {llik_new}")
                llik_new = float('-inf')
                if self.dK is None and not restored_dK:
                    self.delete_dK()
                    restored_dK = True
                th = self.optimize(theta, tmin, tmax, ab, "[dir]", its, verb)
                break
            else:
                break
        
        # Final likelihood calculation if needed
        if not np.isfinite(llik_new):
            llik_new = self.log_likelihood(dab, gab)
        
        # Print final progress
        if verb > 0:
            param_name = 'd' if theta == 'lengthscale' else 'g'
            param_val = self.d if theta == 'lengthscale' else self.g
            print(f"-> {its} Newtons -> ({param_name}={param_val}, llik={llik_new})")
        
        # Restore derivative matrices if needed
        if restored_dK:
            self.new_dK()
        
        return th, its

    def jmle(self, drange: Tuple[float, float], grange: Tuple[float, float], 
         dab: Tuple[float, float], gab: Tuple[float, float], verb: int = 0) -> Tuple[int, int]:
        """
        Joint Maximum Likelihood Estimation for both lengthscale and nugget parameters.
        
        Args:
            drange: Tuple of (min, max) for lengthscale parameter
            grange: Tuple of (min, max) for nugget parameter
            dab: Tuple of prior parameters for lengthscale
            gab: Tuple of prior parameters for nugget
            verb: Verbosity level
            
        Returns:
            Tuple of (dits, gits) - number of iterations for lengthscale and nugget
        """
        # Sanity checks
        assert gab is not None and dab is not None, "Prior parameters must be provided"
        assert grange is not None and drange is not None, "Parameter ranges must be provided"

        dits = gits = 0

        # Loop over coordinate-wise iterations
        for i in range(100):
            # Optimize lengthscale
            th, dit = self.mle('lengthscale', drange[0], drange[1], dab, verb)
            dits += dit

            # Optimize nugget
            th, git = self.mle('nugget', grange[0], grange[1], gab, verb)
            gits += git

            # Check for convergence
            if dit <= 1 and git <= 1:
                break

        if i == 99 and verb > 0:
            print("Warning: max outer its (N=100) reached")

        return dits, gits

    def log_likelihood(self, dab: Optional[Tuple[float, float]] = None, 
                      gab: Optional[Tuple[float, float]] = None) -> float:
        """
        Calculate the log likelihood of the GP model.
        
        Args:
            dab: Optional tuple of (shape, scale) parameters for lengthscale prior
            gab: Optional tuple of (shape, scale) parameters for nugget prior
            
        Returns:
            Log likelihood value
        """
        # Proportional likelihood calculation
        llik = -0.5 * (self.n * np.log(0.5 * self.phi) + self.ldetK)

        # Add lengthscale prior if specified
        if self.d > 0 and dab is not None and dab[0] > 0 and dab[1] > 0:
            llik += gamma.logpdf(self.d, a=dab[0], scale=1.0/dab[1])

        # Add nugget prior if specified
        if self.g > 0 and gab is not None and gab[0] > 0 and gab[1] > 0:
            llik += gamma.logpdf(self.g, a=gab[0], scale=1.0/gab[1])

        return llik
    
    def derivatives(self, theta: str, ab: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Calculate first and second derivatives of the log likelihood.
        
        Args:
            theta: Parameter type ('lengthscale' or 'nugget')
            ab: Optional tuple of (shape, scale) parameters for prior
            
        Returns:
            Tuple of (dllik, d2llik) - first and second derivatives
        """
        # Sanity checks
        assert self.dK is not None and self.d2K is not None, "Derivative matrices must exist"
        
        # Get parameter value
        th = self.d if theta == 'lengthscale' else self.g
        
        # Deal with possible prior
        if ab is not None and ab[0] > 0 and ab[1] > 0:
            dlp = (ab[0] - 1.0) / th - ab[1]
            d2lp = -(ab[0] - 1.0) / (th * th)
        else:
            dlp = d2lp = 0.0
        
        # Initialize derivatives
        dllik = dlp
        d2llik = d2lp
        
        # Calculate dKKi = dK @ Ki
        dKKi = self.dK @ self.Ki
        
        # Calculate dKKidK = dK @ Ki @ dK
        dKKidK = dKKi @ self.dK
        
        # Calculate first part of derivatives
        # dllik = -0.5 * tr(Ki @ dK)
        # d2llik = -0.5 * tr(Ki @ [d2K - dKKidK])
        dllik -= 0.5 * np.trace(self.Ki @ self.dK)
        d2llik -= 0.5 * np.trace(self.Ki @ (self.d2K - dKKidK))
        
        # Calculate two = 2*dKKidK - d2K
        two = 2.0 * dKKidK - self.d2K
        
        # Calculate second part of second derivative
        # d2llik -= 0.5 * KiZ @ two @ KiZ / phi
        KiZtwo = two @ self.KiZ
        
        if self.KiZ.ndim == 1:
            d2llik -= 0.5 * self.n * (self.KiZ.dot(KiZtwo)) / self.phi
        elif self.KiZ.shape[1] == 1:
            d2llik -= 0.5 * self.n * float(self.KiZ.T @ KiZtwo) / self.phi
        else:
            d2llik -= 0.5 * self.n * (self.KiZ @ KiZtwo) / self.phi
        
        # Calculate third part of derivatives
        KiZtwo = self.dK @ self.KiZ
        
        if self.KiZ.ndim == 1:
            phirat = (self.KiZ.dot(KiZtwo)) / self.phi
        elif self.KiZ.shape[1] == 1:
            phirat = float(self.KiZ.T @ KiZtwo) / self.phi
        else:
            phirat = (self.KiZ @ KiZtwo) / self.phi
        
        d2llik += 0.5 * self.n * phirat * phirat
        dllik += 0.5 * self.n * phirat
        
        return dllik, d2llik

    def optimize(self, theta: str, tmin: float, tmax: float, 
        ab: Optional[Tuple[float, float]], msg: str = "", 
        its: int = 0, verb: int = 0) -> float:
        """
        Optimize GP parameters using Brent's method.
        
        Args:
            theta: Parameter to optimize ('lengthscale' or 'nugget')
            tmin: Minimum value for parameter
            tmax: Maximum value for parameter
            ab: Optional tuple of prior parameters
            msg: Message for verbose output
            its: Initial iteration count
            verb: Verbosity level
            
        Returns:
            Optimized parameter value
        """
        # Sanity check
        assert tmin < tmax, "tmin must be less than tmax"
        
        # Get current parameter value
        th = self.d if theta == 'lengthscale' else self.g
        
        # Create optimization info structure
        info = OptInfo(gp=self, theta=theta, ab=ab, verb=verb)
        
        def objective(x: float, info: OptInfo) -> float:
            """Negative log likelihood objective function"""
            # Update GP parameters
            if info.theta == 'lengthscale':
                info.gp.update_params(d=x, g=info.gp.g)
            else:
                info.gp.update_params(d=info.gp.d, g=x)
            
            # Calculate negative log likelihood
            return -info.gp.log_likelihood(
                info.ab if info.theta == 'lengthscale' else None,
                info.ab if info.theta == 'nugget' else None
            )
        
        while True:
            # Use Brent's method for optimization
            tnew = brent_fmin(tmin, tmax, objective, info, np.finfo(float).eps)
            info.its += 1
            
            # Check if solution is on boundary
            if tmin < tnew < tmax:
                break
                
            if tnew == tmin:
                # Left boundary found
                tmin *= 2
                if verb > 0:
                    print(f"opt: tnew=tmin, increasing tmin={tmin}")
            else:
                # Right boundary found
                tmax /= 2.0
                if verb > 0:
                    print(f"opt: tnew=tmax, decreasing tmax={tmax}")
            
            # Check that boundaries are still valid
            if tmin >= tmax:
                raise ValueError("Unable to optimize: tmin >= tmax")
        
        # Update GP parameters with optimal value
        if theta == 'lengthscale':
            if self.d != tnew:
                self.update_params(d=tnew, g=self.g)
        else:
            if self.g != tnew:
                self.update_params(d=self.d, g=tnew)
        
        # Print message if verbose
        if verb > 0:
            print(f"opt {msg}: told={th} -[{info.its}]-> tnew={tnew}")
        
        return tnew

    # def optimize(self, theta: str, tmin: float, tmax: float, 
    #         ab: Optional[Tuple[float, float]], msg: str = "", 
    #         its: int = 0, verb: int = 0) -> float:
    #     """
    #     Optimize GP parameters using bounded optimization via SciPy's minimize_scalar.
    #     Uses Brent's method internally with additional boundary handling.
        
    #     Args:
    #         theta: Parameter to optimize ('lengthscale' or 'nugget')
    #         tmin: Minimum value for parameter
    #         tmax: Maximum value for parameter
    #         ab: Optional tuple of prior parameters
    #         msg: Message for verbose output
    #         its: Initial iteration count
    #         verb: Verbosity level
            
    #     Returns:
    #         Optimized parameter value
    #     """
    #     # Sanity check
    #     assert tmin < tmax, "tmin must be less than tmax"
        
    #     # Get current parameter value
    #     th = self.d if theta == 'lengthscale' else self.g
        
    #     # Create optimization info structure
    #     info = OptInfo(gp=self, theta=theta, ab=ab, verb=verb)
        
    #     def objective(x: float) -> float:
    #         """Negative log likelihood objective function"""
    #         # Update GP parameters
    #         if theta == 'lengthscale':
    #             self.update_params(d=x, g=self.g)
    #         else:
    #             self.update_params(d=self.d, g=x)
            
    #         # Calculate negative log likelihood
    #         return -self.log_likelihood(ab if theta == 'lengthscale' else None,
    #                                 ab if theta == 'nugget' else None)
        
    #     while True:
    #         result = minimize_scalar(objective, 
    #                             bounds=(tmin, tmax),
    #                             method='bounded',
    #                             options={'xatol': np.finfo(float).eps})
            
    #         tnew = result.x
    #         info.its = result.nfev
            
    #         # Check if solution is on boundary
    #         if tmin < tnew < tmax:
    #             break
                
    #         if tnew == tmin:
    #             # Left boundary found
    #             tmin *= 2
    #             if verb > 0:
    #                 print(f"opt: tnew=tmin, increasing tmin={tmin}")
    #         else:
    #             # Right boundary found
    #             tmax /= 2.0
    #             if verb > 0:
    #                 print(f"opt: tnew=tmax, decreasing tmax={tmax}")
            
    #         # Check that boundaries are still valid
    #         if tmin >= tmax:
    #             raise ValueError("Unable to optimize: tmin >= tmax")
        
    #     # Update GP parameters with optimal value
    #     if theta == 'lengthscale':
    #         if self.d != tnew:
    #             self.update_params(d=tnew, g=self.g)
    #     else:
    #         if self.g != tnew:
    #             self.update_params(d=self.d, g=tnew)
        
    #     # Print message if verbose
    #     if verb > 0:
    #         print(f"opt {msg}: told={th} -[{info.its}]-> tnew={tnew}")
        
    #     return tnew

    def update_params(self, d: float, g: float) -> None:
        """
        Update GP parameters and recalculate all dependent matrices.
        
        Args:
            d: New lengthscale parameter
            g: New nugget parameter
        """
        # Sanity checks
        assert d >= 0 and g >= 0, "Parameters must be non-negative"
        if d == 0:
            assert self.dK is None, "dK should be None when d=0"

        # Update parameters
        self.d = d
        self.g = g
        
        # Build covariance matrix
        if d > 0:
            self.K = covar_symm(self.X, d, g, self.kernel)
        else:
            self.K = np.eye(self.n)  # identity matrix when d=0

        # Calculate inverse and log determinant
        self.Ki = np.eye(self.n)  # Start with identity matrix
        if d > 0:
            try:
                # Use Cholesky decomposition for stable inverse
                L = np.linalg.cholesky(self.K)
                self.Ki = np.linalg.inv(self.K)
                self.ldetK = 2 * np.sum(np.log(np.diag(L)))
            except np.linalg.LinAlgError as e:
                raise ValueError(f"Bad Cholesky decomp, d={d}, g={g}") from e
        else:
            self.ldetK = 0.0

        # Update phi = Z^T Ki Z
        if self.Z is not None:
            self.KiZ = self.Ki @ self.Z
            if self.Z.ndim == 1:
                # For 1D arrays, use dot product
                self.phi = self.Z.dot(self.KiZ)
            elif self.Z.shape[1] == 1:
                # For column vectors (n,1), transpose the first one
                self.phi = float(self.Z.T @ self.KiZ)
            else:
                # For other 2D arrays
                self.phi = self.Z @ self.KiZ

        # Calculate derivatives and Fisher info if needed
        if self.dK is not None:
            self.dK, self.d2K = diff_covar_symm(self.X, self.d, self.kernel)
            self.F = self.fisher_info()

    def new_dK(self) -> None:
        """
        Initialize derivative matrices for the GP.
        Calculates first (dK) and second (d2K) derivatives of the covariance matrix,
        and the approximate Fisher information (F).
        """
        assert self.dK is None and self.d2K is None, "Derivative matrices already exist"
        
        # Calculate derivatives of covariance matrix
        self.dK, self.d2K = diff_covar_symm(self.X, self.d, self.kernel)
        
        # Calculate Fisher information
        self.F = self.fisher_info()

    def delete_dK(self) -> None:
        """Delete derivative matrices"""
        self.dK = None
        self.d2K = None
        self.F = 0.0

    def update(self, X_new: np.ndarray, Z_new: np.ndarray, verb: int = 0) -> None:
        """Update GP with new observations"""
        self = updateGP(self, X_new, Z_new)
        if verb > 0:
            print(f"Updated GP with new point(s). New n = {self.n}")

    def predict_lite(self, Xref: np.ndarray, nonug: bool = False, compute_gradients: bool = False) -> Dict:
        """
        Lightweight prediction at reference points
        
        Args:
            Xref: Reference points for prediction
            nonug: If True, use minimal nugget instead of GP nugget
            compute_gradients: If True, include gradients in the output
            
        Returns:
            Dictionary with the following keys:
                "mean": Mean predictions
                "s2": Variance predictions
                "df": Degrees of freedom
                "llik": Log likelihood
                "dmean": Gradients of mean predictions w.r.t. inputs (if compute_gradients=True)
                "ds2": Gradients of variance predictions w.r.t. inputs (if compute_gradients=True)
        """
        g = np.sqrt(np.finfo(float).eps) if nonug else self.g
        
        k, ktKi, ktKik = self.new_predutilGP_lite(len(Xref), Xref)
        
        mean = ktKi @ self.Z
        
        df = float(self.n)
        phidf = self.phi / df
        var = phidf * (1.0 + g - ktKik)
        
        result = {
            "mean": mean,
            "s2": var,
            "df": df,
            "llik": -0.5 * (self.n * np.log(0.5 * self.phi) + self.ldetK)
        }
        
        if compute_gradients:
            nn = len(Xref)
            m = self.m
            
            dmean = np.zeros((nn, m))
            ds2 = np.zeros((nn, m))
            
            for i in range(nn):
                for j in range(m):
                    # Calculate ∂k/∂x_j for point i
                    dk_dx = self.compute_dk_dx(Xref[i:i+1], self.X, j)
                    
                    # ∂μ/∂x = ∂k/∂x @ Ki @ Z
                    dmean[i, j] = dk_dx @ self.Ki @ self.Z
                    
                    # ∂σ²/∂x = -phidf * ∂(ktKik)/∂x
                    # ∂(ktKik)/∂x = 2 * ∂k/∂x @ Ki @ k.T
                    dktKik_dx = 2.0 * dk_dx @ self.Ki @ k[i:i+1].T
                    ds2[i, j] = -phidf * dktKik_dx
            
            result["dmean"] = dmean
            result["ds2"] = ds2
        
        return result

    def predict(self, Xref: np.ndarray, nonug: bool = False, compute_gradients: bool = False) -> Dict:
        """
        Full prediction at reference points
        
        Args:
            Xref: Reference points for prediction
            nonug: If True, use minimal nugget instead of GP nugget
            compute_gradients: If True, include gradients in the output
            
        Returns:
            Dictionary with the following keys:
                "mean": Mean predictions
                "Sigma": Covariance predictions
                "df": Degrees of freedom
                "llik": Log likelihood
                "dmean": Gradients of mean predictions w.r.t. inputs (if compute_gradients=True)
                "ds2": Gradients of variance predictions w.r.t. inputs (if compute_gradients=True)
        """
        nn = len(Xref)
        m = self.m  # number of input dimensions

        g = np.sqrt(np.finfo(float).eps) if nonug else self.g
        
        mean = np.zeros(nn)
        Sigma = covar_symm(Xref, self.d, g, self.kernel)
        
        # Calculate covariance between training and test points
        k = covar(Xref, self.X, self.d, self.kernel)
        
        df = float(self.n)
        phidf = self.phi / df
        
        if compute_gradients:
            dmean = np.zeros((nn, m))
            ds2 = np.zeros((nn, m))
            self.pred_generic(self.n, phidf, self.Z, self.Ki, nn, k, mean, Sigma, 
                             compute_gradients=True, Xref=Xref, dmean=dmean, ds2=ds2)
        else:
            self.pred_generic(self.n, phidf, self.Z, self.Ki, nn, k, mean, Sigma)

        result = {
            "mean": mean,
            "Sigma": Sigma,
            "df": df,
            "llik": -0.5 * (self.n * np.log(0.5 * self.phi) + self.ldetK)
        }

        if compute_gradients:
            result["dmean"] = dmean
            result["ds2"] = ds2
        
        return result

    def pred_generic(self, n: int, phidf: float, Z: np.ndarray, Ki: np.ndarray, 
                 nn: int, k: np.ndarray, mean: np.ndarray, Sigma: np.ndarray,
                 compute_gradients: bool = False, Xref: np.ndarray = None,
                 dmean: np.ndarray = None, ds2: np.ndarray = None) -> None:
        """
        Generic prediction function
        
        Args:
            n: Number of training points
            phidf: phi/df value
            Z: Z values
            Ki: Inverse covariance matrix
            nn: Number of prediction points
            k: Covariance between training and prediction points
            mean: Output array for mean predictions
            Sigma: Output array for covariance predictions
            compute_gradients: If True, compute gradients
            Xref: Reference points for prediction (needed for gradients)
            dmean: Output array for mean gradients (if compute_gradients=True)
            ds2: Output array for variance gradients (if compute_gradients=True)
        """
        # Calculate ktKi = k.T @ Ki
        ktKi = k @ Ki
        
        # Calculate mean predictions: μ = k.T @ Ki @ Z
        mean[:] = ktKi @ Z
        
        # Calculate covariance predictions: Σ = K* - k.T @ Ki @ k
        for i in range(nn):
            Sigma[i, i] = 1.0 + phidf * (1.0 - ktKi[i, :] @ k[:, i])
        
        # Calculate derivatives if requested
        if compute_gradients and Xref is not None and dmean is not None and ds2 is not None:
            m = self.m  # number of input dimensions
            
            # Calculate derivatives of mean: ∂μ/∂x = ∂k/∂x @ Ki @ Z
            for i in range(nn):
                for j in range(m):
                    # Calculate ∂k/∂x_j for point i
                    dk_dx = self.compute_dk_dx(Xref[i:i+1], self.X, j)
                    dmean[i, j] = dk_dx @ Ki @ Z
            
            # Calculate derivatives of variance: ∂σ²/∂x = 2 * phidf * (∂k/∂x @ Ki @ k.T - k @ Ki @ ∂k/∂x.T)
            for i in range(nn):
                for j in range(m):
                    dk_dx = self.compute_dk_dx(Xref[i:i+1], self.X, j)
                    dk_dx_T = dk_dx.T
                    
                    # ∂σ²/∂x = 2 * phidf * (∂k/∂x @ Ki @ k.T - k @ Ki @ ∂k/∂x.T)
                    term1 = dk_dx @ Ki @ k[i:i+1].T
                    term2 = k[i:i+1] @ Ki @ dk_dx_T
                    ds2[i, j] = 2.0 * phidf * (term1 - term2)

    def new_predutilGP_lite(self, nn: int, XX: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Utility function for prediction calculations.
        
        Args:
            nn: Number of prediction points
            XX: Prediction locations
        
        Returns:
            k: Covariance between training and test points
            ktKi: k.T @ Ki
            ktKik: Diagonal of ktKi @ k
        """
        # Calculate covariance between training and test points
        k = covar(XX, self.X, self.d, self.kernel)
        
        # Calculate ktKi and ktKik
        ktKi = k @ self.Ki
        ktKik = np.sum(ktKi * k, axis=1)
        
        return k, ktKi, ktKik

    def fisher_info(self) -> float:
        """
        Calculate approximate Fisher information.
        
        Returns:
            Approximate Fisher information
        """
        # F = 0.5 * tr((Ki @ dK)^2)
        KidK = self.Ki @ self.dK
        F = 0.5 * np.trace(KidK @ KidK)
        return F
    
    def compute_dk_dx(self, x: np.ndarray, X: np.ndarray, dim: int) -> np.ndarray:
        """
        Compute derivative of covariance vector k with respect to input dimension dim.
        
        Args:
            x: Single prediction point (1, m)
            X: Training points (n, m)
            dim: Input dimension index
            
        Returns:
            Derivative of covariance vector (1, n)
        """
        from .utils.distance import distance
        
        # Calculate squared distances
        D_sq = distance(x, X)  # Returns squared distances
        D = np.sqrt(np.maximum(D_sq, 0))  # Euclidean distances
        
        x_diff = x[0, dim] - X[:, dim] 
        
        if self.kernel == 'squared_exponential':
            # k = exp(-||x-x'||²/d)
            # dk/dx = -2 * k * (x - x') / d
            k = np.exp(-D_sq / self.d)
            dk_dx = -2 * k * x_diff / self.d
            
        elif self.kernel == 'exponential':
            # k = exp(-||x-x'||/d)
            # dk/dx = -k * (x - x') / (d * ||x-x'||)
            k = np.exp(-D / self.d)
            D_safe = np.maximum(D, np.finfo(float).eps)
            dk_dx = -k * x_diff / (self.d * D_safe)
            
        elif self.kernel == 'matern32':
            # k = (1 + √3*r/d) * exp(-√3*r/d), where r = ||x-x'||
            # dk/dx = -3 * (x - x') * exp(-√3*r/d) / d²
            sqrt3 = np.sqrt(3)
            sqrt3_D_d = sqrt3 * D / self.d
            k = (1 + sqrt3_D_d) * np.exp(-sqrt3_D_d)
            D_safe = np.maximum(D, np.finfo(float).eps)
            dk_dx = -3 * x_diff * np.exp(-sqrt3_D_d) / (self.d**2 * D_safe)
            
        elif self.kernel == 'matern52':
            # k = (1 + √5*r/d + 5*r²/(3*d²)) * exp(-√5*r/d)
            # dk/dx = -5 * (x - x') * (1 + √5*r/d) * exp(-√5*r/d) / (3*d²*r)
            sqrt5 = np.sqrt(5)
            sqrt5_D_d = sqrt5 * D / self.d
            k = (1 + sqrt5_D_d + 5 * D**2 / (3 * self.d**2)) * np.exp(-sqrt5_D_d)
            D_safe = np.maximum(D, np.finfo(float).eps)
            dk_dx = -5 * x_diff * (1 + sqrt5_D_d) * np.exp(-sqrt5_D_d) / (3 * self.d**2 * D_safe)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")
       
        return dk_dx

def newGP(X: np.ndarray, Z: np.ndarray, d: float, g: float, kernel: KernelType = 'squared_exponential',
           compute_derivs: bool = False) -> GP:
    """
    Create a new Gaussian Process
    
    Args:
        X: Design matrix
        Z: Response vector
        d: Length scale parameter
        g: Nugget parameter
        compute_derivs: Whether to compute derivatives
        kernel: Kernel type
        
    Returns:
        New GP instance
    """    
    # Calculate covariance matrix
    K = covar_symm(X, d, g, kernel)
    
    # Calculate inverse and log determinant
    L = np.linalg.cholesky(K)
    Ki = np.linalg.inv(K)
    ldetK = 2 * np.sum(np.log(np.diag(L)))
    
    # Calculate KiZ
    KiZ = Ki @ Z
    
    # Calculate phi
    if Z.ndim == 1:
        phi = Z.dot(KiZ)
    elif Z.shape[1] == 1:
        phi = Z.T @ KiZ
    else:
        phi = Z @ KiZ
    
    gp = GP(X=X, K=K, Ki=Ki, Z=Z, KiZ=KiZ, 
            ldetK=ldetK, d=d, g=g, phi=phi, kernel=kernel)
    
    if compute_derivs:
        gp.new_dK()
    return gp

def updateGP(gp: GP, X_new: np.ndarray, Z_new: np.ndarray) -> GP:
    """
    Update GP with new observations
    
    Args:
        gp: GP instance to update
        X_new: New input points
        Z_new: New observations
        
    Returns:
        Updated GP instance
    """
    gp.X = np.vstack([gp.X, X_new])
    gp.Z = np.concatenate([gp.Z, Z_new])
    
    # Recalculate covariance matrix
    gp.K = covar_symm(gp.X, gp.d, gp.g, gp.kernel)
    
    # Update inverse and log determinant
    try:
        L = np.linalg.cholesky(gp.K)
        gp.Ki = np.linalg.inv(gp.K)
        gp.ldetK = 2 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix became singular during update")
    
    # Update KiZ
    gp.KiZ = gp.Ki @ gp.Z
    
    # Update phi
    gp.phi = gp.Z @ gp.KiZ
    
    return gp

#these are placeholder functions for now. Will develop these later if needed.
def mspe(gp: GP, Xcand: np.ndarray, Xref: np.ndarray, verb: int = 0) -> np.ndarray:
    """Calculate MSPE criterion"""
    # Implementation of MSPE calculations
    pass

def alcray_selection(gp: GP, Xcand: np.ndarray, Xref: np.ndarray, 
                    roundrobin: int, numstart: int, rect: np.ndarray,
                    verb: int = 0) -> int:
    """ALCRAY point selection"""
    # Implementation of ALCRAY selection
    pass

def buildGP(X: np.ndarray, 
         Z: np.ndarray, 
         d: Optional[Union[float, Tuple[float, float]]] = None,
         g: float = 1/10000,
         wdir: str = '.',
         fname: str = 'GPRmodel.gp',
         export: bool = True,
         kernel: KernelType = 'squared_exponential',
         verb: int = 0) -> GP:
    """
    Builds GP for Gaussian Process Regression. 
    Uses all the training data to build the model (i.e., not a local approximate GP!).
    
    Args:
        X: Training inputs (n × m)
        Z: Training outputs (n,)
        d: Lengthscale parameter or tuple of (start, mle)
        g: Nugget parameter
        wdir: Directory to save the GP model
        fname: Name of the GP model file
        export: Whether to export the GP model to a file
        kernel: Kernel type
        verb: Verbosity level
        
    Returns:
        Pickle (.gp) file containing GP model with the following attributes:
            m: Number of dimensions
            n: Number of data points
            Ki: Inverse of the covariance matrix
            d: Lengthscale parameter
            g: Nugget parameter
            phi: Precision parameter
            X: Design matrix
    """
    d_prior = darg(d, X)
    g_prior = garg(g, Z)
    
    gp = newGP(X, Z, get_value(d_prior, 'start'), get_value(g_prior, 'start'), kernel=kernel)
    optimize_parameters(gp, d_prior, g_prior, verb)

    #if required, save GP model to file that can be readily imported
    if export:
        full_path = os.path.join(wdir, fname)
        with open(full_path, 'wb') as file:
            pickle.dump(gp, file)

        if verb > 0:
            print(f"GP model saved to {fname}")

    return gp


def loadGP(wdir: str = '.', fname: Optional[str] = None) -> GP:
    """Load GP model from a specified .gp file or the first .gp file found in the directory.
    
    Args:
        wdir: Directory to search for .gp files
        fname: Specific .gp file to load (if None, the first .gp file found in the directory is loaded)
    Returns:
        GP model
    """

    if fname:
        # If a filename is provided, construct the full path
        full_path = os.path.join(wdir, fname)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"The specified file '{fname}' does not exist in the directory.")
    else:
        # List all files in the specified directory
        files = os.listdir(wdir)
        
        # Filter for files with a .gp extension
        gp_files = [f for f in files if f.endswith('.gp')]
        
        if not gp_files:
            raise FileNotFoundError("No .gp file found in the directory.")
        
        if len(gp_files) > 1:
            raise ValueError("Multiple .gp files found. Please specify a specific filename.")
        
        # Use the first .gp file found
        full_path = os.path.join(wdir, gp_files[0])
    
    with open(full_path, 'rb') as file:
        return pickle.load(file)