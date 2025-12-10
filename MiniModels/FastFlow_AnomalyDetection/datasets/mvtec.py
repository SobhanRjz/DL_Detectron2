"""MVTec AD dataset implementation."""
from typing import Optional, Tuple, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.interfaces import IDataset


class MVTecDataset(Dataset, IDataset):
    """MVTec Anomaly Detection dataset."""

    def __init__(self, root: str, category: str, split: str = 'train',
                 image_size: Tuple[int, int] = (256, 256),
                 transform: Optional[Callable] = None):
        """
        Args:
            root: Root directory of MVTec dataset
            category: Product category (e.g., 'bottle', 'cable')
            split: Dataset split ('train', 'test', 'val')
            image_size: Target image size
            transform: Optional transform to apply
        """
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size

        # Handle custom data structure
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.transform = transform or self._default_transform()
        self.samples = self._load_samples()
    
    def _default_transform(self) -> transforms.Compose:
        """Create default image transforms."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_samples(self) -> list:
        """Load dataset samples."""
        samples = []

        
        # Handle custom data structure: 0_Normal/, defect_*/
        if self.split == 'train':
            # Training data: only normal samples from 0_Normal
            normal_dir = self.root/ 'train' / 'good'
            if normal_dir.exists():
                for img_path in sorted(normal_dir.glob('*.jpg')):  # Changed to .jpg
                    samples.append({
                        'image_path': img_path,
                        'label': 0,  # Normal
                        'mask_path': None
                    })
        else:
            # Test data: normal and anomalous samples
            # Include some normal samples for testing
            normal_dir = self.root / 'test' / 'good'
            if normal_dir.exists():
                normal_files = sorted(normal_dir.glob('*.jpg'))
                # Use 20% of normal images for testing
                test_normal_count = max(1, len(normal_files) // 5)
                for img_path in normal_files[:test_normal_count]:
                    samples.append({
                        'image_path': img_path,
                        'label': 0,  # Normal
                        'mask_path': None
                    })

            # Include anomalous samples from defect directories
            defect_base = self.root / 'test'
            if defect_base.exists():
                for defect_dir in sorted(defect_base.iterdir()):
                    if not defect_dir.is_dir() or defect_dir.name == 'good':
                        continue
                    for img_path in sorted(defect_dir.glob('*.jpg')):
                        samples.append({
                            'image_path': img_path,
                            'label': 1,  # Anomaly
                            'mask_path': None  # No masks available
                        })

        return samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image, label, mask)
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        label = sample['label']
        
        # Load mask if available
        mask = None
        if sample['mask_path'] is not None:
            mask = Image.open(sample['mask_path']).convert('L')
            mask = transforms.Resize(self.image_size)(mask)
            mask = transforms.ToTensor()(mask)
        
        return image, label, mask


class AnomalyDataset(Dataset, IDataset):
    """Generic anomaly detection dataset."""
    
    def __init__(self, image_paths: list, labels: list,
                 image_size: Tuple[int, int] = (256, 256),
                 transform: Optional[Callable] = None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0=normal, 1=anomaly)
            image_size: Target image size
            transform: Optional transform to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transform or self._default_transform()
        
        if len(image_paths) != len(labels):
            raise ValueError("Number of images and labels must match")
    
    def _default_transform(self) -> transforms.Compose:
        """Create default image transforms."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return image, label

