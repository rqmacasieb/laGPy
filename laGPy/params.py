from typing import Optional, Union, Dict
import numpy as np
from scipy.special import gamma as sp_gamma, gammaln as sp_gammaln, gammaincinv as sp_gammaincinv
from laGPy.gamma import Rgamma_inv as lgp_Rgamma_inv, Cgamma as lgp_Cgamma, Igamma_inv as lgp_Igamma_inv
from .utils.distance import distance
from .gp import *

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
    
    start_values = np.atleast_1d(d['start'])
    if np.any((start_values < d['min']) | (start_values > d['max'])):
        raise ValueError("(all) starting d-value(s) should be positive scalars in [d['min'], d['max']]")
    
    if not isinstance(d['mle'], bool):
        raise ValueError("d['mle'] should be a scalar logical")
    
    if len(d['ab']) != 2 or np.any(np.array(d['ab']) < 0):
        raise ValueError("d['ab'] should be a length 2 vector of non-negative scalars")

def get_Ds(X: np.ndarray, p: float = 0.1, samp_size: int = 1000) -> Dict:
    """
    Calculate initial starting value and range for lengthscale parameter.
    
    Args:
        X: Input data matrix.
        samp_size: Sample size for distance calculations.
        
    Returns:
        Dictionary with 'start', 'min', and 'max' values.
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
        g = {'mle': True}
    elif isinstance(g, (int, float)):
        g = {'start': g}
    elif not isinstance(g, dict):
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
            g['ab'][1] = lgp_Igamma_inv(
                g['ab'][0], 
                0.95 * sp_gamma(g['ab'][0]), 
                lower=True, 
                ulog=False
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
           - None: perform prior calculations and mle for optimization
           - float: use as starting value
           - dict: full specification
        X: Input matrix for distance calculations
        samp_size: Sample size for distance calculations
    
    Returns:
        Dictionary containing processed parameter settings
    """
    # Coerce inputs first
    if d is None:
        d = {'mle': True}
    elif isinstance(d, (int, float)):
        d = {'start': d}
    elif not isinstance(d, dict):
        raise ValueError("d should be a dict, numeric, or None")
    
    # Now d is guaranteed to be a dict, so we can use setdefault
    d.setdefault('mle', False)
    
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
            d['ab'][1] = lgp_Igamma_inv(
                d['ab'][0], 
                0.95 * sp_gamma(d['ab'][0]), 
                lower=True, 
                ulog=False
            ) / Ds['max']
    
    # Check validity of values
    check_arg(d)
    
    return d

def get_value(param, key_or_index):
    """
    Retrieve a value from a parameter that can be either a dictionary or a list.
    
    Args:
        param: The parameter, which can be a dictionary or a list.
        key_or_index: The key (if param is a dictionary) or index (if param is a list).
        
    Returns:
        The value corresponding to the key or index.
        
    Raises:
        ValueError: If the parameter is not a dictionary or a list.
    """
    if isinstance(param, dict):
        return param.get(key_or_index)
    elif isinstance(param, (list, tuple)):
        return param[key_or_index]
    else:
        raise ValueError("Parameter must be a dictionary or a list/tuple.")
    
def optimize_parameters(gp, d, g, verb):
    """
    Optimize the GP parameters using JMLE or MLE based on the provided settings.

    Args:
        gp: The Gaussian Process instance.
        d: Lengthscale parameter specification (dict or list).
        g: Nugget parameter specification (dict or list).
        verb: Verbosity level.
    """
    if get_value(d, 'mle') and get_value(g, 'mle'):
        if gp.dK is None:
            gp.new_dK()
        gp.jmle(
            drange=(get_value(d, 'min'), get_value(d, 'max')),
            grange=(get_value(g, 'min'), get_value(g, 'max')),
            dab=get_value(d, 'ab'),
            gab=get_value(g, 'ab'),
            verb=verb
        )
    elif get_value(d, 'mle'):
        if gp.dK is None:
            gp.new_dK()
        gp.mle(
            'lengthscale',
            get_value(d, 'min'),
            get_value(d, 'max'),
            get_value(d, 'ab'),
            verb
        )
    elif get_value(g, 'mle'):
        gp.mle(
            'nugget',
            get_value(g, 'min'),
            get_value(g, 'max'),
            get_value(g, 'ab'),
            verb
        )