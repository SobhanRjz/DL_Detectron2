"""Few-shot learning library."""

from .core import (
    ModelConfig,
    DataConfig,
    PrototypicalNetwork,
    ModelFactory,
    DatasetManager,
    Trainer,
    Evaluator,
)
from .pipeline import Pipeline
from .utils import set_seed, get_device

__version__ = "2.0.0"

__all__ = [
    "ModelConfig",
    "DataConfig",
    "PrototypicalNetwork",
    "ModelFactory",
    "DatasetManager",
    "Trainer",
    "Evaluator",
    "Pipeline",
    "set_seed",
    "get_device",
]

