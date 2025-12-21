"""Core components for few-shot learning."""

from .config import ModelConfig, DataConfig
from .model import PrototypicalNetwork, ModelFactory
from .dataset import DatasetManager
from .trainer import Trainer
from .evaluator import Evaluator
from .optimizer import (
    HyperparameterOptimizer,
    calculate_model_efficiency,
    create_model_config_from_trial,
    objective_function,
    apply_best_params_to_config
)

__all__ = [
    "ModelConfig",
    "DataConfig",
    "PrototypicalNetwork",
    "ModelFactory",
    "DatasetManager",
    "Trainer",
    "Evaluator",
    "HyperparameterOptimizer",
    "calculate_model_efficiency",
    "create_model_config_from_trial",
    "objective_function",
    "apply_best_params_to_config",
]

