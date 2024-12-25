"""laGPy: Python implementation of local approximate GP"""

# lagpy/__init__.py
from .laGPy import *
from .gp import *
from .covar import *
from .matrix import get_data_rect
from .order import order, rank, rand_indices
from .covar_sep import covar_sep_symm, covar_sep, diff_covar_sep, diff_covar_sep_symm
from .utils.distance import *
from .prior import *

from . import _version
__version__ = _version.get_versions()['version']

__all__ = [
    'laGP',
    'Method',
    'MLEResult',
    'GP',
    'new_gp',
    'pred_gp',
    'update_gp',
    'covar',
    'covar_symm',
    'distance',
    'get_data_rect',
    'order',
    'rank',
    'rand_indices',
    '__version__'
]