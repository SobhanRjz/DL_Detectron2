"""Dataset management for few-shot learning."""

from typing import Tuple, Optional
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from easyfsl.datasets import FewShotDataset
from easyfsl.samplers import TaskSampler

from .config import DataConfig, ModelConfig, AugmentationConfig


class EpisodeAwareDataLoader:
    """DataLoader that ensures consistent augmentations within episodes."""

    def __init__(self, dataset, batch_sampler, num_workers=0, pin_memory=True, collate_fn=None):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

    def __iter__(self):
        # Reset episode counter at the start of each iteration
        episode_counter = 0
        
        # Create the underlying DataLoader
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

        for batch in dataloader:
            # Set episode seed for consistent augmentations within this episode
            if hasattr(self.dataset, 'transform') and hasattr(self.dataset.transform, 'set_episode_seed'):
                self.dataset.transform.set_episode_seed(episode_counter)

            episode_counter += 1
            yield batch

    def __len__(self):
        return len(self.batch_sampler)


class EpisodeAwareTransform:
    """Transform that applies consistent augmentations within episodes for few-shot learning."""

    def __init__(self, augmentation_config: AugmentationConfig, image_size: Tuple[int, int], is_training: bool = True):
        """Initialize episode-aware transform.

        Args:
            augmentation_config: Augmentation configuration
            image_size: Target image size
            is_training: Whether this is for training (enables random augmentations)
        """
        self.augmentation_config = augmentation_config
        self.image_size = image_size
        self.is_training = is_training
        self.episode_seed: Optional[int] = None

        # Create base transforms list
        self.transforms_list = []

        if is_training:
            # Add random augmentations that need episode-level consistency
            if augmentation_config.horizontal_flip or augmentation_config.vertical_flip:
                self.transforms_list.append(self._get_flip_transform())

            if augmentation_config.rotation_degrees > 0:
                self.transforms_list.append(
                    transforms.RandomRotation(degrees=augmentation_config.rotation_degrees)
                )

            if augmentation_config.color_jitter:
                self.transforms_list.append(transforms.ColorJitter(
                    brightness=augmentation_config.color_jitter_brightness,
                    contrast=augmentation_config.color_jitter_contrast,
                    saturation=augmentation_config.color_jitter_saturation,
                    hue=augmentation_config.color_jitter_hue
                ))

        # Resize transforms (either RandomResizedCrop or standard Resize)
        if is_training and augmentation_config.random_resized_crop:
            self.transforms_list.append(transforms.RandomResizedCrop(
                size=image_size,
                scale=augmentation_config.random_resized_crop_scale,
                ratio=augmentation_config.random_resized_crop_ratio
            ))
        else:
            self.transforms_list.append(transforms.Resize(image_size))

        # Always applied transforms
        self.transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform = transforms.Compose(self.transforms_list)

    def _get_flip_transform(self):
        """Get flip transform based on configuration."""
        augmentation_config = self.augmentation_config

        if augmentation_config.horizontal_flip and augmentation_config.vertical_flip:
            # Allow choice between horizontal, vertical, or both
            return transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomVerticalFlip(p=1.0)
                ])
            ])
        elif augmentation_config.horizontal_flip:
            return transforms.RandomHorizontalFlip(p=0.5)
        elif augmentation_config.vertical_flip:
            return transforms.RandomVerticalFlip(p=0.5)
        else:
            return None

    def set_episode_seed(self, seed: int):
        """Set random seed for this episode to ensure consistent augmentations.
        
        Args:
            seed: Seed value for this episode
        """
        self.episode_seed = seed

    def __call__(self, img):
        """Apply transforms with episode-level random seed if set.
        
        Args:
            img: PIL Image to transform
            
        Returns:
            Transformed tensor
        """
        if self.episode_seed is not None and self.is_training:
            # Save current random states
            torch_state = torch.get_rng_state()
            np_state = np.random.get_state()
            random_state = random.getstate()

            try:
                # Set episode-specific seed for reproducible augmentations
                torch.manual_seed(self.episode_seed)
                np.random.seed(self.episode_seed)
                random.seed(self.episode_seed)
                
                result = self.transform(img)
            finally:
                # Always restore original random states
                torch.set_rng_state(torch_state)
                np.random.set_state(np_state)
                random.setstate(random_state)

            return result
        else:
            # No episode seed set or not training - apply transforms normally
            return self.transform(img)


class PicklableImageFolder(datasets.ImageFolder):
    """ImageFolder that can be pickled and provides get_labels method."""

    def get_labels(self) -> list[int]:
        """Get labels for all samples."""
        return self.targets


class FewShotWrapper(FewShotDataset):
    """Wrapper for standard datasets to work with EasyFSL."""

    def __init__(self, subset: Subset) -> None:
        """Initialize wrapper.

        Args:
            subset: Dataset subset
        """
        super().__init__()
        self.subset = subset
        self.dataset = subset.dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.subset[index]

    def __len__(self) -> int:
        return len(self.subset)

    def get_labels(self) -> list[int]:
        return [self.dataset.samples[i][1] for i in self.subset.indices]


class DatasetManager:
    """Manages data loading and preprocessing."""

    def __init__(self, data_config: DataConfig, model_config: ModelConfig) -> None:
        """Initialize manager.

        Args:
            data_config: Data paths configuration
            model_config: Model configuration for transforms
        """
        self.data_config = data_config
        self.model_config = model_config

        # Create episode-aware transforms for training and testing
        self.train_transform = EpisodeAwareTransform(
            model_config.augmentation,
            model_config.image_size,
            is_training=True
        )
        self.test_transform = EpisodeAwareTransform(
            model_config.augmentation,
            model_config.image_size,
            is_training=False
        )

        # For backward compatibility
        self.transform = self.train_transform

        self.train_set = PicklableImageFolder(
            root=data_config.train_root,
            transform=self.train_transform
        )
        self.test_set = PicklableImageFolder(
            root=data_config.test_root,
            transform=self.test_transform
        )
    
    def _create_task_sampler(
        self,
        dataset: PicklableImageFolder,
        n_tasks: int
    ) -> TaskSampler:
        """Create episodic task sampler.
        
        Args:
            dataset: Source dataset
            n_tasks: Number of episodes
            
        Returns:
            TaskSampler instance
        """
        return TaskSampler(
            dataset,
            n_way=self.model_config.n_way,
            n_shot=self.model_config.n_shot,
            n_query=self.model_config.n_query,
            n_tasks=n_tasks
        )

    def _create_loader(
        self,
        dataset: PicklableImageFolder,
        sampler: TaskSampler,
        is_training: bool = True
    ) -> DataLoader:
        """Create DataLoader with episodic sampling.

        Args:
            dataset: Source dataset
            sampler: Task sampler
            is_training: Whether this is for training (affects episode seeding)

        Returns:
            DataLoader instance
        """
        return EpisodeAwareDataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.model_config.num_workers,
            pin_memory=True,
            collate_fn=sampler.episodic_collate_fn
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        sampler = self._create_task_sampler(
            self.train_set,
            self.model_config.n_training_episodes
        )
        return self._create_loader(self.train_set, sampler)
    
    def get_validation_loader(self) -> DataLoader:
        """Get validation data loader."""
        sampler = self._create_task_sampler(
            self.test_set,
            self.model_config.n_validation_tasks
        )
        return self._create_loader(self.test_set, sampler)
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        sampler = self._create_task_sampler(
            self.test_set,
            self.model_config.n_evaluation_tasks
        )
        return self._create_loader(self.test_set, sampler)

