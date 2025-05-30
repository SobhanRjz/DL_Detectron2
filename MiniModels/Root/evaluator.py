"""Evaluation utilities for few-shot learning."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import PrototypicalNetworks


class FewShotEvaluator:
    """Evaluator for few-shot learning models."""
    
    def __init__(self, model: PrototypicalNetworks, device: torch.device) -> None:
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
    
    def evaluate_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[int, int]:
        """Evaluate a single episode.
        
        Args:
            support_images: Support set images
            support_labels: Support set labels
            query_images: Query set images
            query_labels: Query set labels
            
        Returns:
            Tuple of (correct_predictions, total_predictions)
        """
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        with torch.no_grad():
            scores = self.model(support_images, support_labels, query_images)
            _, predicted_labels = torch.max(scores.data, 1)
            
        correct = (predicted_labels == query_labels).sum().item()
        total = len(query_labels)
        
        return correct, total
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on a data loader.
        
        Args:
            data_loader: Data loader to evaluate on
            
        Returns:
            Accuracy as a percentage
        """
        self.model.eval()
        total_predictions = 0
        correct_predictions = 0

        with torch.no_grad():
            for (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm(data_loader, desc="Evaluating"):
                
                correct, total = self.evaluate_episode(
                    support_images, support_labels, query_images, query_labels
                )
                
                total_predictions += total
                correct_predictions += correct

        accuracy = 100 * correct_predictions / total_predictions
        print(f"Model tested on {len(data_loader)} tasks. Accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Make predictions on query images.
        
        Args:
            support_images: Support set images
            support_labels: Support set labels
            query_images: Query set images
            
        Returns:
            Predicted labels for query images
        """
        self.model.eval()
        
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        
        with torch.no_grad():
            scores = self.model(support_images, support_labels, query_images)
            _, predicted_labels = torch.max(scores.data, 1)
            
        return predicted_labels.cpu() 