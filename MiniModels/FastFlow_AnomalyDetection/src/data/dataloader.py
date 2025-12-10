"""DataLoader management for training and testing"""

import torch.utils.data

import config.constants as const
from .dataset import MVTecDataset


class DataLoaderManager:
    """Manages data loaders for training and testing"""
    
    def __init__(self, data_root, category, input_size, batch_size=None, num_workers=0):
        """Initialize DataLoader manager
        
        Args:
            data_root: Root directory of dataset
            category: Dataset category
            input_size: Input image size
            batch_size: Batch size (default: from constants)
            num_workers: Number of worker processes (default: 0 for Windows)
        """
        self.data_root = data_root
        self.category = category
        self.input_size = input_size
        self.batch_size = batch_size or const.BATCH_SIZE
        self.num_workers = num_workers
        
        self._train_loader = None
        self._test_loader = None
    
    def get_train_loader(self):
        """Get training data loader
        
        Returns:
            DataLoader for training
        """
        if self._train_loader is None:
            train_dataset = MVTecDataset(
                root=self.data_root,
                category=self.category,
                input_size=self.input_size,
                is_train=True,
            )
            self._train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
        return self._train_loader
    
    def get_test_loader(self):
        """Get testing data loader
        
        Returns:
            DataLoader for testing
        """
        if self._test_loader is None:
            test_dataset = MVTecDataset(
                root=self.data_root,
                category=self.category,
                input_size=self.input_size,
                is_train=False,
            )
            self._test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
            )
        return self._test_loader
    
    def get_loaders(self):
        """Get both train and test loaders
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        return self.get_train_loader(), self.get_test_loader()

