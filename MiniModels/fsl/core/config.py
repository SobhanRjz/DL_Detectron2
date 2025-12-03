"""Configuration classes for few-shot learning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""
    
    # Model
    backbone: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True
    
    # Few-shot parameters
    n_way: int = 3
    n_shot: int = 5
    n_query: int = 1
    
    # Training
    n_training_episodes: int = 500
    n_validation_tasks: int = 50
    n_evaluation_tasks: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 1e-4
    validation_frequency: int = 100
    log_frequency: int = 10
    
    # Data
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 0
    
    # System
    device: str = "cuda"
    random_seed: int = 42


@dataclass
class DataConfig:
    """Dataset paths and configuration."""

    train_root: str
    test_root: str
    output_root: str = "output"
    
    def __post_init__(self) -> None:
        """Validate paths."""
        if not Path(self.train_root).exists():
            raise FileNotFoundError(f"Train path not found: {self.train_root}")
        if not Path(self.test_root).exists():
            raise FileNotFoundError(f"Test path not found: {self.test_root}")

