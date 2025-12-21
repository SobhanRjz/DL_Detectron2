"""Dataset management for few-shot learning."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from easyfsl.datasets import FewShotDataset
from easyfsl.samplers import TaskSampler

from .config import DataConfig, ModelConfig


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
        
        self.transform = transforms.Compose([
            transforms.Resize(model_config.image_size),
            transforms.ToTensor(),
        ])
        
        self.train_set = PicklableImageFolder(
            root=data_config.train_root,
            transform=self.transform
        )
        self.test_set = PicklableImageFolder(
            root=data_config.test_root,
            transform=self.transform
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
        sampler: TaskSampler
    ) -> DataLoader:
        """Create DataLoader with episodic sampling.
        
        Args:
            dataset: Source dataset
            sampler: Task sampler
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
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

