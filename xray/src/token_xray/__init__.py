"""Token X-ray: measured per-token gradient weights and cross-sequence leaks through real training stacks."""
from .api import Report, xray
from .probe import build_probe, make_batch

__all__ = ["xray", "Report", "build_probe", "make_batch"]
__version__ = "0.1.1"
