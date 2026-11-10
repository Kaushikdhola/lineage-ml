"""Token X-ray: measured per-token gradient weights and cross-sequence leaks through real training stacks."""
from .api import Report, xray
from .probe import build_probe, make_batch
from .trainer_xray import TrainerReport, xray_trainer

__all__ = ["xray", "xray_trainer", "Report", "TrainerReport", "build_probe", "make_batch"]
__version__ = "0.2.0"
