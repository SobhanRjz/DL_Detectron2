"""Configuration settings for few-shot learning."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    RANDOM_SEED: int = 42
    DEVICE: str = "cuda"
    BACKBONE_NAME: str = "resnet18"
    PRETRAINED: bool = True
    
    # Few-shot learning parameters
    N_WAY: int = 3 # Number of classes in the support set (Root fine, Root mass, Root tap)
    N_SHOT: int = 5 # Number of images per class in the support set
    N_QUERY: int = 1 # Number of images per class in the query set
    N_EVALUATION_TASKS: int = 100 # Number of tasks for evaluation
    N_TRAINING_EPISODES: int = 500 # Reduced training episodes to prevent overfitting
    N_VALIDATION_TASKS: int = 50 # Number of tasks for validation
    
    # Training parameters
    LEARNING_RATE: float = 0.0001  # Reduced learning rate for few-shot learning
    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 0  # Set to 0 to avoid Windows multiprocessing issues
    LOG_UPDATE_FREQUENCY: int = 10
    VALIDATION_FREQUENCY: int = 50  # Validate every 50 episodes
    
    # Regularization
    WEIGHT_DECAY: float = 1e-4
    FREEZE_BACKBONE: bool = True  # Freeze backbone layers
    
    # Image parameters
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    
    # Data Augmentation parameters
    USE_AUGMENTATION: bool = True
    AUGMENTATION_STRATEGY: str = "basic"  # "none", "basic", "medium", "strong", "custom"
    
    # Individual augmentation controls
    HORIZONTAL_FLIP_PROB: float = 0.5
    VERTICAL_FLIP_PROB: float = 0.0
    ROTATION_DEGREES: float = 15.0
    COLOR_JITTER_BRIGHTNESS: float = 0.2
    COLOR_JITTER_CONTRAST: float = 0.2
    COLOR_JITTER_SATURATION: float = 0.2
    COLOR_JITTER_HUE: float = 0.1
    GAUSSIAN_BLUR_PROB: float = 0.0
    GAUSSIAN_BLUR_KERNEL_SIZE: int = 3
    RANDOM_ERASING_PROB: float = 0.0
    RANDOM_ERASING_SCALE: Tuple[float, float] = (0.02, 0.33)
    CUTOUT_PROB: float = 0.0
    CUTOUT_SIZE: int = 16
    MIXUP_ALPHA: float = 0.0  # 0.0 means no mixup
    CUTMIX_ALPHA: float = 0.0  # 0.0 means no cutmix


@dataclass
class DataConfig:
    """Data configuration parameters."""
    
    BASE_DIR: str = r"C:\Users\sobha\Desktop\detectron2"
    TRAIN_IMAGE_ROOT: str = os.getenv(
        "TRAIN_IMAGE_ROOT", 
        os.path.join(BASE_DIR, "Data", "Third Names", "Roots", "Data", "Train")
    )
    TEST_IMAGE_ROOT: str = os.getenv(
        "TEST_IMAGE_ROOT", 
        os.path.join(BASE_DIR, "Data", "Third Names", "Roots", "Data", "Test")
    )
    
    def __post_init__(self) -> None:
        """Validate paths exist."""
        if not Path(self.TRAIN_IMAGE_ROOT).exists():
            raise FileNotFoundError(f"Train path not found: {self.TRAIN_IMAGE_ROOT}")
        if not Path(self.TEST_IMAGE_ROOT).exists():
            raise FileNotFoundError(f"Test path not found: {self.TEST_IMAGE_ROOT}")