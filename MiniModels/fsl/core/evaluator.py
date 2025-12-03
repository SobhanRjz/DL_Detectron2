"""Evaluation logic for few-shot learning."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import PrototypicalNetwork


class Evaluator:
    """Handles model evaluation."""
    
    def __init__(self, model: PrototypicalNetwork, device: torch.device) -> None:
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Evaluation device
        """
        self.model = model
        self.device = device
    
    def _evaluate_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[int, int]:
        """Evaluate single episode.
        
        Args:
            support_images: Support set
            support_labels: Support labels
            query_images: Query set
            query_labels: Query labels
            
        Returns:
            (correct_count, total_count)
        """
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        with torch.no_grad():
            scores = self.model(support_images, support_labels, query_images)
            _, preds = torch.max(scores, 1)
            
        correct = (preds == query_labels).sum().item()
        return correct, len(query_labels)
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on data loader.
        
        Args:
            data_loader: Evaluation data
            
        Returns:
            Accuracy percentage
        """
        self.model.eval()
        total = correct = 0

        with torch.no_grad():
            for support_imgs, support_lbls, query_imgs, query_lbls, _ in tqdm(data_loader, desc="Evaluating"):
                c, t = self._evaluate_episode(support_imgs, support_lbls, query_imgs, query_lbls)
                correct += c
                total += t

        accuracy = 100 * correct / total
        print(f"Tested on {len(data_loader)} tasks. Accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Make predictions.
        
        Args:
            support_images: Support set
            support_labels: Support labels
            query_images: Query set
            
        Returns:
            Predicted labels
        """
        self.model.eval()
        
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        
        with torch.no_grad():
            scores = self.model(support_images, support_labels, query_images)
            _, preds = torch.max(scores, 1)
            
        return preds.cpu()

