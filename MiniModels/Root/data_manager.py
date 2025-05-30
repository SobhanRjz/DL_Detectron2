"""Data management for few-shot learning."""

from typing import Tuple, Any
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from easyfsl.datasets import FewShotDataset
from easyfsl.samplers import TaskSampler

from config import DataConfig, ModelConfig


class RootDefectDataset(FewShotDataset):
    """Custom wrapper for root defect dataset."""
    
    def __init__(self, subset: Subset) -> None:
        """Initialize dataset wrapper.
        
        Args:
            subset: PyTorch subset of the original dataset
        """
        super().__init__()
        self.subset = subset
        self.dataset = subset.dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        return self.subset[index]

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.subset)

    def get_labels(self) -> list[int]:
        """Get labels for all samples."""
        return [self.dataset.samples[i][1] for i in self.subset.indices]


class PicklableImageFolder(datasets.ImageFolder):
    """ImageFolder that can be pickled for multiprocessing."""
    
    def get_labels(self) -> list[int]:
        """Get labels for all samples."""
        return self.targets


class DataManager:
    """Manages data loading and preprocessing for few-shot learning."""
    
    def __init__(self, data_config: DataConfig, model_config: ModelConfig) -> None:
        """Initialize data manager.
        
        Args:
            data_config: Data configuration
            model_config: Model configuration
        """
        self.data_config = data_config
        self.model_config = model_config
        self._setup_transforms()
        self._load_datasets()
    
    def _setup_transforms(self) -> None:
        """Setup image transformations."""
        self.transform = transforms.Compose([
            transforms.Resize(self.model_config.IMAGE_SIZE),
            transforms.ToTensor(),
        ])
    
    def _load_datasets(self) -> None:
        """Load train and test datasets."""
        self.train_set = PicklableImageFolder(
            root=self.data_config.TRAIN_IMAGE_ROOT, 
            transform=self.transform
        )
        self.test_set = PicklableImageFolder(
            root=self.data_config.TEST_IMAGE_ROOT, 
            transform=self.transform
        )
    
    def create_task_sampler(
        self, 
        dataset: datasets.ImageFolder, 
        n_tasks: int
    ) -> TaskSampler:
        """Create task sampler for episodic training.
        
        Args:
            dataset: Dataset to sample from
            n_tasks: Number of tasks to sample
            
        Returns:
            TaskSampler instance
        """
        return TaskSampler(
            dataset,
            n_way=self.model_config.N_WAY,
            n_shot=self.model_config.N_SHOT,
            n_query=self.model_config.N_QUERY,
            n_tasks=n_tasks
        )
    
    def create_dataloader(
        self, 
        dataset: datasets.ImageFolder, 
        sampler: TaskSampler
    ) -> DataLoader:
        """Create DataLoader with episodic sampling.
        
        Args:
            dataset: Dataset to load from
            sampler: Task sampler for episodic loading
            
        Returns:
            DataLoader instance
        """
        # Use num_workers=0 on Windows to avoid multiprocessing issues
        num_workers = 0 if self.model_config.NUM_WORKERS > 0 else self.model_config.NUM_WORKERS
        
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=sampler.episodic_collate_fn
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        train_sampler = self.create_task_sampler(
            self.train_set, 
            self.model_config.N_TRAINING_EPISODES
        )
        return self.create_dataloader(self.train_set, train_sampler)
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        test_sampler = self.create_task_sampler(
            self.test_set, 
            self.model_config.N_EVALUATION_TASKS
        )
        return self.create_dataloader(self.test_set, test_sampler)
    
    def get_validation_loader(self) -> DataLoader:
        """Get validation data loader."""
        val_sampler = self.create_task_sampler(
            self.test_set, 
            self.model_config.N_VALIDATION_TASKS
        )
        return self.create_dataloader(self.test_set, val_sampler) 