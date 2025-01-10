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


__all__ = [
    'laGP',
    'buildGP',
    'loadGP',
    'fullGP',
    '__version__'
]