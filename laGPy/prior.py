from typing import Optional, Union, Dict, List
import numpy as np
from scipy.special import gamma
from .utils.distance import distance

def check_arg(d: Dict) -> None:
    """
    Check validity of parameter arguments.
    
    Args:
        d: Dictionary containing parameter settings
    
    Raises:
        ValueError: If any parameter settings are invalid
    """
    if not isinstance(d['max'], (int, float)) or d['max'] <= 0:
        raise ValueError("d['max'] should be a positive scalar")
    
    if not isinstance(d['min'], (int, float)) or d['min'] < 0 or d['min'] > d['max']:
        raise ValueError("d['min'] should be a positive scalar < d['max']")
    
    if any(s < d['min'] or s > d['max'] for s in np.atleast_1d(d['start'])):
        raise ValueError("(all) starting d-value(s) should be positive scalars in [d['min'], d['max']]")
    
    if not isinstance(d['mle'], bool):
        raise ValueError("d['mle'] should be a scalar logical")
    
    if len(d['ab']) != 2 or any(a < 0 for a in d['ab']):
        raise ValueError("ab should be a positive 2-vector")

def Igamma_inv(a: float, y: float, lower: bool = False, log: bool = False) -> float:
    """
    Calculate the beta parameter of an Inverse Gamma distribution.
    
    Args:
        a: Shape parameter
        y: Location parameter
        lower: Whether to use lower tail
        log: Whether y is in log scale
    
    Returns:
        Beta parameter value
    """
    from scipy.stats import invgamma
    if log:
        y = np.exp(y)
    if lower:
        p = y
    else:
        p = 1 - y
    return invgamma.ppf(p, a)

def get_Ds(X: np.ndarray, p: float = 0.1, samp_size: int = 1000) -> Dict:
    """
    Calculate initial starting value and range for lengthscale parameter.
    """
    if X.shape[0] > samp_size:
        idx = np.random.choice(X.shape[0], samp_size, replace=False)
        X = X[idx]
    
    D = distance(X, X)
    D = D[np.triu_indices_from(D, k=1)]
    D = D[D > 0]
    
    return {
        'start': np.quantile(D, p),
        'min': np.min(D),
        'max': np.max(D)
    }

def garg(g: Optional[Union[float, Dict]] = None, 
         y: np.ndarray = None) -> Dict:
    """
    Process the nugget parameter arguments for GP models.
    
    Args:
        g: Nugget parameter specification:
           - None: use defaults
           - float: use as starting value
           - dict: full specification
        y: Output values for calculating defaults
    
    Returns:
        Dictionary containing processed parameter settings
    """
    # Coerce inputs
    if g is None:
        g = {}
    elif isinstance(g, (int, float)):
        g = {'start': g}
    if not isinstance(g, dict):
        raise ValueError("g should be a dict, numeric, or None")
    
    # Check mle
    g.setdefault('mle', False)
    if not isinstance(g['mle'], bool):
        raise ValueError("g['mle'] should be a scalar logical")
    
    # Calculate squared residuals if needed
    need_r2s = (
        'start' not in g or 
        (g['mle'] and (
            'max' not in g or 
            'ab' not in g or 
            (isinstance(g.get('ab'), (list, np.ndarray)) and 
             len(g['ab']) > 1 and 
             np.isnan(g['ab'][1]))
        ))
    )
    
    if need_r2s:
        r2s = (y - np.mean(y))**2
    
    # Check for starting value
    if 'start' not in g:
        g['start'] = float(np.quantile(r2s, 0.025))
    
    # Check for max value
    if 'max' not in g:
        if g['mle']:
            g['max'] = np.max(r2s)
        else:
            g['max'] = np.max(g['start'])
    
    # Check for min value
    if 'min' not in g:
        g['min'] = np.sqrt(np.finfo(float).eps)
    
    # Check for priors
    if not g['mle']:
        g['ab'] = [0, 0]
    else:
        if 'ab' not in g:
            g['ab'] = [3/2, np.nan]
        if isinstance(g['ab'], (list, np.ndarray)) and len(g['ab']) > 1 and np.isnan(g['ab'][1]):
            s2max = np.mean(r2s)
            g['ab'][1] = Igamma_inv(
                g['ab'][0], 
                0.95 * gamma(g['ab'][0]), 
                lower=True, 
                log=False
            ) / s2max
    
    # Check validity of values
    check_arg(g)
    
    return g

def darg(d: Optional[Union[float, Dict]] = None, 
         X: np.ndarray = None, 
         samp_size: int = 1000) -> Dict:
    """
    Process the lengthscale parameter arguments for GP models.
    
    Args:
        d: Lengthscale parameter specification:
           - None: use defaults
           - float: use as starting value
           - dict: full specification
        X: Input matrix for distance calculations
        samp_size: Sample size for distance calculations
    
    Returns:
        Dictionary containing processed parameter settings
    """
    # Coerce inputs
    if d is None:
        d = {}
    elif isinstance(d, (int, float)):
        d = {'start': d}
    if not isinstance(d, dict):
        raise ValueError("d should be a dict, numeric, or None")
    
    # Check for MLE
    d.setdefault('mle', True)
    
    # Check if we need to build Ds
    need_Ds = (
        'start' not in d or 
        (d['mle'] and (
            'max' not in d or 
            'min' not in d or 
            'ab' not in d or 
            (isinstance(d.get('ab'), (list, np.ndarray)) and 
             len(d['ab']) > 1 and 
             np.isnan(d['ab'][1]))
        ))
    )
    
    if need_Ds:
        Ds = get_Ds(X, samp_size=samp_size)
    
    # Check for starting value
    if 'start' not in d:
        d['start'] = Ds['start']
    
    # Check for max value
    if 'max' not in d:
        if d['mle']:
            d['max'] = Ds['max']
        else:
            d['max'] = np.max(d['start'])
    
    # Check for min value
    if 'min' not in d:
        if d['mle']:
            d['min'] = Ds['min'] / 2
        else:
            d['min'] = np.min(d['start'])
        if d['min'] < np.sqrt(np.finfo(float).eps):
            d['min'] = np.sqrt(np.finfo(float).eps)
    
    # Check for priors
    if not d['mle']:
        d['ab'] = [0, 0]
    else:
        if 'ab' not in d:
            d['ab'] = [3/2, np.nan]
        if isinstance(d['ab'], (list, np.ndarray)) and len(d['ab']) > 1 and np.isnan(d['ab'][1]):
            d['ab'][1] = Igamma_inv(
                d['ab'][0], 
                0.95 * gamma(d['ab'][0]), 
                lower=True, 
                log=False
            ) / Ds['max']
    
    # Check validity of values
    check_arg(d)
    
    return d

def get_start_value(param):
    """Helper function to get the start value from a parameter."""
    if isinstance(param, dict):
        return param['start']
    elif isinstance(param, list) or isinstance(param, np.ndarray):
        return param[0]
    else:
        raise ValueError("Parameter must be a dictionary or a list/array.")
