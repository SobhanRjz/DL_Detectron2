"""Dataset classes for anomaly detection"""

import os
from glob import glob
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MVTecDataset(torch.utils.data.Dataset):
    """MVTec-AD style dataset for anomaly detection
    
    Supports both training (normal images only) and testing (normal + anomaly images)
    """
    
    def __init__(self, root, category, input_size, is_train=True):
        """Initialize dataset
        
        Args:
            root: Root directory of dataset
            category: Category name (e.g., 'bottle', 'pipe_anomaly')
            input_size: Image size for model input
            is_train: Whether this is training or testing dataset
        """
        self.root = root
        self.category = category
        self.input_size = input_size
        self.is_train = is_train
        
        self.image_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Load image file paths
        self.image_files = self._load_image_paths()
        
        if not is_train:
            self.target_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])
        
        print(f"Dataset mode: {'train' if is_train else 'test'}, "
              f"found {len(self.image_files)} images")

    def _load_image_paths(self):
        """Load image file paths based on dataset mode
        
        Returns:
            List of image file paths
        """
        if self.is_train:
            # Training: only normal images
            image_files = glob(
                os.path.join("MiniModels", "FastFlow_AnomalyDetection", "data", 
                           "anomaly_data", "split_data", self.category, "train", 
                           "good", "*.jpg")
            )
        else:
            # Testing: normal and anomaly images
            normal_files = glob(
                os.path.join("MiniModels", "FastFlow_AnomalyDetection", "data", 
                           "anomaly_data", "split_data", self.category, "test", 
                           "good", "*.jpg")
            )
            # Balance classes by limiting normal images
            num_normal_test = min(len(normal_files) // 5, 20)
            normal_files = normal_files[:num_normal_test]

            # Get all anomaly images
            anomaly_files = []
            defect_dirs = glob(
                os.path.join("MiniModels", "FastFlow_AnomalyDetection", "data", 
                           "anomaly_data", "split_data", self.category, "test", 
                           "defect_*")
            )
            for defect_dir in defect_dirs:
                if os.path.isdir(defect_dir):
                    anomaly_files.extend(glob(os.path.join(defect_dir, "*.jpg")))

            image_files = normal_files + anomaly_files
        
        return image_files

    def __getitem__(self, index):
        """Get item by index
        
        Args:
            index: Dataset index
            
        Returns:
            Tuple of (image, label) tensors
        """
        image_file = self.image_files[index]
        image = Image.open(image_file).convert('RGB')
        image = self.image_transform(image)

        if self.is_train:
            # Training: always return normal label
            return image, torch.tensor(0, dtype=torch.long)
        else:
            # Testing: determine if normal or anomaly
            is_normal = "good" in image_file
            label = 0 if is_normal else 1
            return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """Get dataset size"""
        return len(self.image_files)

