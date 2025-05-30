"""Few-shot learning package for root defect classification."""

from .main_few_shot_learning import FewShotLearningPipeline
from .config import ModelConfig, DataConfig
from .models import PrototypicalNetworks, ModelFactory
from .trainer import FewShotTrainer
from .evaluator import FewShotEvaluator
from .data_manager import DataManager, RootDefectDataset
from .utils import set_random_seeds, get_device, Visualizer

__all__ = [
    "FewShotLearningPipeline",
    "ModelConfig",
    "DataConfig", 
    "PrototypicalNetworks",
    "ModelFactory",
    "FewShotTrainer",
    "FewShotEvaluator",
    "DataManager",
    "RootDefectDataset",
    "set_random_seeds",
    "get_device",
    "Visualizer",
]

__version__ = "1.0.0" 