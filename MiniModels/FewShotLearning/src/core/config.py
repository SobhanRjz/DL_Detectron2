"""Configuration classes for few-shot learning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from torchvision import transforms


@dataclass
class AugmentationConfig:
    """Data augmentation configuration for few-shot learning."""

    # Flip augmentations
    horizontal_flip: bool = False
    vertical_flip: bool = False

    # Rotation augmentation
    rotation_degrees: int = 0  # 0 means no rotation, positive values enable random rotation

    # Scale augmentation
    random_resized_crop: bool = False
    random_resized_crop_scale: Tuple[float, float] = (0.8, 1.0)
    random_resized_crop_ratio: Tuple[float, float] = (0.75, 1.333)

    # Color augmentations
    color_jitter: bool = False
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.05

    # Combined augmentation strategies (mutually exclusive with individual settings)
    augmentation_strategy: str = "none"  # "none", "basic", "moderate", "strong", "robust"

    def __post_init__(self) -> None:
        """Validate augmentation configuration."""
        if self.augmentation_strategy != "none":
            # If using strategy, override individual settings
            self._apply_strategy()
        else:
            # Validate individual settings
            if self.rotation_degrees < 0:
                raise ValueError("rotation_degrees must be non-negative")

    def _apply_strategy(self) -> None:
        """Apply predefined augmentation strategy."""
        if self.augmentation_strategy == "basic":
            # Basic: Only horizontal flip
            self.horizontal_flip = True
            self.vertical_flip = False
            self.rotation_degrees = 0
            self.random_resized_crop = False
            self.color_jitter = False
        elif self.augmentation_strategy == "moderate":
            # Moderate: Horizontal flip + small rotation + random crop
            self.horizontal_flip = True
            self.vertical_flip = False
            self.rotation_degrees = 15
            self.random_resized_crop = True
            self.color_jitter = False
        elif self.augmentation_strategy == "strong":
            # Strong: Horizontal flip + rotation + random crop + color jitter
            self.horizontal_flip = True
            self.vertical_flip = False
            self.rotation_degrees = 30
            self.random_resized_crop = True
            self.color_jitter = True
        elif self.augmentation_strategy == "robust":
            # Robust: All augmentations with aggressive settings
            self.horizontal_flip = True
            self.vertical_flip = True
            self.rotation_degrees = 45
            self.random_resized_crop = True
            self.random_resized_crop_scale = (0.5, 1.0)
            self.color_jitter = True
            self.color_jitter_brightness = 0.2
            self.color_jitter_contrast = 0.2
            self.color_jitter_saturation = 0.2
            self.color_jitter_hue = 0.1
        else:
            raise ValueError(f"Unknown augmentation strategy: {self.augmentation_strategy}")

    def get_transforms(self, image_size: Tuple[int, int], is_training: bool = True):
        """Get torchvision transforms based on configuration.

        Args:
            image_size: Target image size (height, width)
            is_training: Whether transforms are for training (affects random augmentations)

        Returns:
            Composed transforms
        """
        transforms_list = []

        if is_training and (self.horizontal_flip or self.vertical_flip):
            # Apply flips consistently within episodes
            if self.horizontal_flip and self.vertical_flip:
                transforms_list.append(transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomVerticalFlip(p=1.0),
                    transforms.Compose([
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.RandomVerticalFlip(p=1.0)
                    ])
                ]))
            elif self.horizontal_flip:
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
            elif self.vertical_flip:
                transforms_list.append(transforms.RandomVerticalFlip(p=0.5))

        if is_training and self.rotation_degrees > 0:
            transforms_list.append(
                transforms.RandomRotation(degrees=self.rotation_degrees)
            )

        if is_training and self.color_jitter:
            transforms_list.append(transforms.ColorJitter(
                brightness=self.color_jitter_brightness,
                contrast=self.color_jitter_contrast,
                saturation=self.color_jitter_saturation,
                hue=self.color_jitter_hue
            ))

        if is_training and self.random_resized_crop:
            transforms_list.append(transforms.RandomResizedCrop(
                size=image_size,
                scale=self.random_resized_crop_scale,
                ratio=self.random_resized_crop_ratio
            ))
        else:
            # Standard resize if not using random crop
            transforms_list.append(transforms.Resize(image_size))

        # Always applied transforms
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transforms.Compose(transforms_list)


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
    n_training_episodes: int = 150
    n_validation_tasks: int = 50
    n_evaluation_tasks: int = 50
    learning_rate: float = 0.0001
    weight_decay: float = 1e-4
    validation_frequency: int = 100
    log_frequency: int = 10
    early_stopping_patience: int = 5
    scheduler_step_divisor: int = 4  # step_size = n_training_episodes // scheduler_step_divisor

    # Data augmentation
    augmentation: AugmentationConfig = None

    # Data
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 0

    # System
    device: str = "cuda"
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Initialize augmentation config if not provided."""
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()



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

