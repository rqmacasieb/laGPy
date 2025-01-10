"""laGPy: Python implementation of local approximate GP"""


from .laGPy import (
    Method,
    closest_indices,
    laGP,
    calc_ktKikx, 
    fullGP
)

# Import from covar.py
from .covar import (
    covar,
    diff_covar_symm
)

# Import from params.py
from .params import (
    darg,
    garg
)

# Import from utils.distance
from .utils.distance import (
    distance,
    distance_asymm,
    closest_indices
)

from .gp import (
    buildGP,
    loadGP
)

from . import _version
__version__ = _version.get_versions()['version']


# from .laGPy import *
# from .gp import *
# from .covar import *
# from .matrix import get_data_rect
# from .order import order, rank, rand_indices
# from .covar_sep import covar_sep_symm, covar_sep, diff_covar_sep, diff_covar_sep_symm
# from .utils.distance import *
# from .params import *
# from .gamma import *



__all__ = [
    'laGP',
    'buildGP',
    'loadGP',
    'fullGP',
    '__version__'
]