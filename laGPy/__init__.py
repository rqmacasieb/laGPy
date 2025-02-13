"""laGPy: Python implementation of local approximate GP"""

from .laGPy import (
    Method,
    laGP,
    fullGP
)

from .covar import (
    covar,
    diff_covar_symm
)

from .params import (
    darg,
    garg
)

from .utils.distance import (
    distance,
    distance_asymm
)

from .utils.brent_fmin import (
    brent_fmin
)

from .gp import (
    buildGP,

    loadGP, 
    newGP,
    updateGP
)

from . import _version
__version__ = _version.get_versions()['version']


__all__ = [
    'laGP',
    'buildGP',
    'loadGP',
    'fullGP',
    'newGP',
    'updateGP',
    '__version__'
]