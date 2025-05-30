"""Utility functions for few-shot learning."""

import random
from typing import List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from easyfsl.utils import plot_images


def set_random_seeds(seed: int = 0) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sliding_average(values: List[float], window_size: int) -> float:
    """Calculate sliding average of values.
    
    Args:
        values: List of values
        window_size: Size of sliding window
        
    Returns:
        Sliding average
    """
    if len(values) < window_size:
        return sum(values) / len(values)
    return sum(values[-window_size:]) / window_size


def get_device(preferred_device: str = "cuda") -> torch.device:
    """Get available device.
    
    Args:
        preferred_device: Preferred device name
        
    Returns:
        Available torch device
    """
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Visualizer:
    """Visualization utilities for few-shot learning."""
    
    @staticmethod
    def plot_predictions(
        query_images: torch.Tensor,
        predicted_labels: torch.Tensor,
        class_ids: torch.Tensor,
        class_names: List[str],
        images_per_row: int = 3,
        title: str = "Predicted Labels"
    ) -> None:
        """Plot query images with predicted labels.
        
        Args:
            query_images: Query images tensor
            predicted_labels: Predicted label indices
            class_ids: Class IDs for the episode
            class_names: List of class names
            images_per_row: Number of images per row
            title: Plot title
        """
        # Convert predicted labels to class names
        predicted_class_ids = [class_ids[i] for i in predicted_labels.tolist()]
        predicted_class_names = [class_names[i] for i in predicted_class_ids]
        
        # Set matplotlib to non-interactive mode
        plt.ioff()
        
        plot_images(query_images.cpu(), predicted_class_names, images_per_row)
        plt.suptitle(title, fontsize=16)
        
        # Show plot non-blocking and close after a short time
        plt.show(block=False)
        plt.pause(3)  # Display for 2 seconds
        plt.close('all')  # Close all figures
    
    @staticmethod
    def plot_support_and_query(
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        n_shot: int,
        n_query: int
    ) -> None:
        """Plot support and query images.
        
        Args:
            support_images: Support set images
            query_images: Query set images
            n_shot: Number of shots per class
            n_query: Number of queries per class
        """
        # Set matplotlib to non-interactive mode
        plt.ioff()
        
        plot_images(support_images, "Support Images", images_per_row=n_shot)
        plt.show(block=False)
        plt.pause(2)  # Display for 2 seconds
        plt.close('all')
        
        plot_images(query_images, "Query Images", images_per_row=n_query)
        plt.show(block=False)
        plt.pause(2)  # Display for 2 seconds
        plt.close('all') 