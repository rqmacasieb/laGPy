from typing import Callable, Any
import numpy as np

def brent_fmin(ax: float, bx: float, f: Callable[[float, Any], float], 
               info: Any, tol: float) -> float:
    """
    Brent's method for finding minimum of function f between ax and bx.
    
    Args:
        ax: Left endpoint of interval
        bx: Right endpoint of interval
        f: Function to minimize, takes (x, info) as arguments
        info: Additional info passed to f
        tol: Tolerance for convergence
        
    Returns:
        x: Value minimizing f in interval [ax, bx]
    """
    # Constants
    GOLDEN_RATIO = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.sqrt(np.finfo(float).eps)
    tol3 = tol / 3.0
    
    # Initialize using vectorized operations
    a, b = ax, bx
    points = np.array([a + GOLDEN_RATIO * (b - a)] * 3)  # [v, w, x]
    v, w, x = points
    
    # Initial function evaluations using vectorized call
    f_values = np.array([f(x, info)] * 3)  # [fv, fw, fx]
    fv, fw, fx = f_values
    
    d = e = 0.0
    
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        
        # Vectorized stopping criterion
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
            
        # Vectorized parabolic interpolation calculations
        if abs(e) > tol1:
            diffs = np.array([x - w, x - v])
            f_diffs = np.array([fx - fv, fx - fw])
            r, q = diffs * f_diffs
            p = diffs[1] * q - diffs[0] * r
            q = (q - r) * 2.0
            
            if q > 0.0:
                p = -p
            else:
                q = -q
            r, e = e, d
            
            # Vectorized step selection
            use_golden = (abs(p) >= abs(q * 0.5 * r)) or (p <= q * (a - x)) or (p >= q * (b - x))
            if use_golden:
                e = b - x if x < xm else a - x
                d = GOLDEN_RATIO * e
            else:
                d = p / q
                u = x + d
                if (u - a < t2) or (b - u < t2):
                    d = np.copysign(tol1, xm - x)
        else:
            e = b - x if x < xm else a - x
            d = GOLDEN_RATIO * e
        
        # Vectorized point selection
        u = x + (np.copysign(tol1, d) if abs(d) < tol1 else d)
        fu = f(u, info)
        
        # Vectorized updates using boolean indexing
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
                
    return x
