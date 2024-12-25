import numpy as np
from scipy.special import gammaincinv, gamma, gammaln
from scipy.stats import gamma as gamma_dist

def Rgamma_inv(a, y, lower, ulog):
    """
    Compute the inverse of the regularized gamma function.
    
    Args:
        a: Shape parameter.
        y: Location parameter.
        lower: Whether to use the lower tail.
        ulog: Whether y is in log scale.
        
    Returns:
        Inverse of the regularized gamma function.
    """
    if ulog:
        y = y * np.log(10)  # Convert from base-10 log scale to natural log scale
    if lower:
        r = gamma_dist.ppf(y, a, scale=1.0)  # Use ppf for the lower tail
    else:
        r = gamma_dist.isf(y, a, scale=1.0)  # Use isf for the upper tail (1 - CDF)
    
    assert not np.isnan(r), "Result is NaN"
    return r

def Cgamma(a, ulog):
    """
    Compute the complete gamma function or its logarithm.
    
    Args:
        a: Shape parameter.
        ulog: Whether to return the logarithm of the gamma function.
        
    Returns:
        The gamma function value or its logarithm divided by ln(10).
    """
    if ulog:
        r = gammaln(a) / np.log(10)  # Use gammaln for log(gamma)
    else:
        r = gamma(a)  # Use gamma for the gamma function
    
    assert not np.isnan(r), "Result is NaN"
    return r

def Igamma_inv(a, y, lower, ulog):
    """
    Incomplete gamma inverse function.
    
    Args:
        a: Shape parameter.
        y: Location parameter.
        lower: Whether to use the lower tail.
        ulog: Whether y is in log scale.
        
    Returns:
        Inverse of the incomplete gamma function.
    """
    if ulog:
        r = Rgamma_inv(a, y - Cgamma(a, ulog), lower, ulog)
    else:
        r = Rgamma_inv(a, y / Cgamma(a, ulog), lower, ulog)
    
    assert not np.isnan(r), "Result is NaN"
    return r