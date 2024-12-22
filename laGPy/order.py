import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Rank:
    """Structure for ranking"""
    value: float
    index: int

def order(s: np.ndarray) -> np.ndarray:
    """
    Obtain the integer order of the indices of s from least to greatest.
    The returned indices o applied to s (e.g. s[o]) would result in a sorted list.
    
    Args:
        s: Input array
        
    Returns:
        Array of indices that would sort the input
    """
    ranks = [Rank(value=val, index=idx) for idx, val in enumerate(s)]
    ranks.sort(key=lambda x: x.value)
    return np.array([r.index for r in ranks])

def rank(s: np.ndarray) -> np.ndarray:
    """
    Obtain the integer rank of the elements of s
    
    Args:
        s: Input array
        
    Returns:
        Array of ranks for each element
    """
    ranks = [Rank(value=val, index=idx) for idx, val in enumerate(s)]
    ranks.sort(key=lambda x: x.value)
    result = np.zeros(len(s), dtype=int)
    for i, r in enumerate(ranks):
        result[r.index] = i
    return result

def rand_indices(N: int) -> np.ndarray:
    """
    Return a random permutation of the indices 0...N-1
    
    Args:
        N: Size of permutation
        
    Returns:
        Random permutation of indices
    """
    return np.random.permutation(N)