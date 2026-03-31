from .adaptive import run as run_adaptive
from .anisotropic import run as run_anisotropic
from .baseline import run as run_baseline
from .beta import run as run_beta
from .multires import run as run_multires

__all__ = [
    "run_baseline",
    "run_adaptive",
    "run_anisotropic",
    "run_multires",
    "run_beta",
]
