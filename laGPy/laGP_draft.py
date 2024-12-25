import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict, NamedTuple
from .gp import *
from .matrix import get_data_rect
from .order import order
from .covar_sep import *
from .covar import *
from .prior import *
from .utils.distance import distance

class Method(Enum):
    ALC = 1
    ALCOPT = 2
    ALCRAY = 3
    MSPE = 4
    FISH = 5
    NN = 6

def laGP(m: int, start: int, end: int, Xref: np.ndarray, nref: int, 
         n: int, X: np.ndarray, Z: np.ndarray, d: np.ndarray, g: np.ndarray,
         method: Method, close: int, alc_gpu: int, numstart: int,
         rect: np.ndarray, verb: int, Xi_ret: bool,
         mean: np.ndarray, s2: np.ndarray, lite: bool) -> Tuple[float, float, float, float]:
    """
    Core implementation (equivalent to laGP in laGP.c)
    """
    # Initialize GP
    gp = GP(X[:start], Z[:start], d[0], g[0])
    
    # Sort candidates by distance to Xref
    dst = np.min(distance(Xref, X), axis=0)
    cands = np.argsort(dst)
    Xi = cands[:start]
    
    # Store indices if requested
    if Xi_ret is not None:
        Xi_ret[:start] = Xi
    
    # Determine remaining candidates
    if close >= n:
        close = 0
    if close > 0:
        if close >= n - start:
            raise ValueError("close not less than remaining cands")
        cands = cands[start:close]
    else:
        cands = cands[start:]
    
    # Set up rect from cands if not specified
    if method in [Method.ALCRAY, Method.ALCOPT]:
        if rect is None:
            rect = np.vstack([
                np.min(X[cands], axis=0),
                np.max(X[cands], axis=0)
            ])
    
    # Main selection loop
    for t in range(start, end):
        # Select next point based on method
        if method == Method.ALCRAY:
            offset = ((t - start) % int(np.sqrt(t - start))) + 1
            w = lalcray(gp, Xref, X[cands], rect, offset, numstart, verb-2)
        elif method == Method.ALCOPT:
            offset = t - start
            w = lalcopt(gp, Xref, X[cands], rect, offset, numstart, verb-2)
        else:
            if method == Method.ALC:
                als = alc(gp, X[cands], Xref, alc_gpu, verb-2)
            elif method == Method.MSPE:
                als = -mspe(gp, X[cands], Xref, verb-2)
            elif method == Method.FISH:
                als = fish(gp, X[cands])
            else:  # Method.NN
                als = np.zeros(len(cands))
                als[0] = 1
            
            als[~np.isfinite(als)] = np.nan
            w = np.nanargmax(als)
        
        # Update GP with chosen point
        gp.update(X[cands[w]:cands[w]+1], Z[cands[w]:cands[w]+1], verb-1)
        if Xi_ret is not None:
            Xi_ret[t] = cands[w]
        
        # Update candidate set
        cands = np.delete(cands, w)
    
    # Final prediction
    if lite:
        mean_pred, s2_pred, df, llik = gp.predict_lite(Xref)
    else:
        mean_pred, s2_pred, df, llik = gp.predict(Xref)
    
    mean[:] = mean_pred
    s2[:] = s2_pred
    
    return df, llik

def laGP_R(m: int, start: int, end: int, Xref: np.ndarray, nref: int,
           n: int, X: np.ndarray, Z: np.ndarray, d: np.ndarray, g: np.ndarray,
           method: int, close: int, alc_gpu: int, numstart: int,
           rect: np.ndarray, lite: bool, verb: int, Xi_ret: bool) -> Dict:
    """
    Python equivalent of laGP_R in laGP.c
    """
    # Convert method integer to enum
    method_enum = Method(method)
    
    # Initialize output arrays
    mean = np.zeros(nref)
    s2dim = nref if lite else nref * nref
    s2 = np.zeros(s2dim)
    Xi = np.zeros(end, dtype=int) if Xi_ret else None
    
    # Call core implementation
    df, llik = laGP(m, start, end, Xref, nref, n, X, Z, d, g,
                    method_enum, close, alc_gpu, numstart, rect,
                    verb, Xi, mean, s2, lite)
    
    return {
        'mean': mean,
        's2': s2,
        'df': df,
        'llik': llik,
        'Xi': Xi
    }

def laGP_interface(Xref: np.ndarray, 
                  start: int, 
                  end: int, 
                  X: np.ndarray, 
                  Z: np.ndarray, 
                  d: Optional[Union[float, Tuple[float, float]]] = None,
                  g: float = 1/10000,
                  method: str = "alc",
                  Xi_ret: bool = True,
                  close: Optional[int] = None,
                  alc_gpu: bool = False,
                  numstart: Optional[int] = None,
                  rect: Optional[np.ndarray] = None,
                  lite: bool = True,
                  verb: int = 0) -> Dict:
    """
    Python equivalent of laGP in laGP.R
    High-level interface for users
    """
    # Method mapping
    method_map = {
        "alc": 1, "alcopt": 2, "alcray": 3,
        "mspe": 4, "fish": 5, "nn": 6
    }
    
    # Input processing
    method = method.lower()
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")
    imethod = method_map[method]
    
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    Z = np.asarray(Z)
    Xref = np.atleast_2d(Xref)
    
    # Get dimensions
    m = X.shape[1]
    n = X.shape[0]
    nref = Xref.shape[0]
    
    # Input validation
    if start < 6 or end <= start:
        raise ValueError("must have 6 <= start < end")
    if Xref.shape[1] != m:
        raise ValueError("bad dims")
    if len(Z) != n:
        raise ValueError("bad dims")
    
    # Set defaults
    if close is None:
        mult = 10 if method in ["alcray", "alcopt"] else 1
        close = min((1000 + end) * mult, n)
    if numstart is None:
        numstart = m if method == "alcray" else 1
    
    # Process rect
    if method in ["alcray", "alcopt"]:
        if rect is None:
            rect = np.zeros((2, m))
        if method == "alcray" and nref != 1:
            raise ValueError("alcray only implemented for nrow(Xref) = 1")
    else:
        rect = np.zeros(1)
    
    # Process parameters
    d_proc = process_params(d, X)
    g_proc = process_params(g, Z)
    
    # Start timing
    tic = time.time()
    
    # Call R interface equivalent
    result = laGP_R(
        m=m, start=start, end=end,
        Xref=Xref, nref=nref, n=n,
        X=X, Z=Z,
        d=d_proc, g=g_proc,
        method=imethod,
        close=close,
        alc_gpu=int(alc_gpu),
        numstart=numstart,
        rect=rect,
        lite=lite,
        verb=verb,
        Xi_ret=Xi_ret
    )
    
    # Add timing and method info
    result['time'] = time.time() - tic
    result['method'] = method
    result['d'] = d_proc
    result['g'] = g_proc
    result['close'] = close
    
    # Convert s2 to Sigma if needed
    if not lite:
        result['Sigma'] = result['s2'].reshape(nref, nref)
        del result['s2']
    
    # Add ray info if needed
    if method in ["alcray", "alcopt"]:
        result['numstart'] = numstart
    
    return result