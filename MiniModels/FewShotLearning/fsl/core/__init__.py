"""Core components for few-shot learning."""

from .config import ModelConfig, DataConfig
from .model import PrototypicalNetwork, ModelFactory
from .dataset import DatasetManager
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    "ModelConfig",
    "DataConfig",
    "PrototypicalNetwork",
    "ModelFactory",
    "DatasetManager",
    "Trainer",
    "Evaluator",
]

