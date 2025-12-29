"""Training logic for few-shot learning."""

from typing import List
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ModelConfig
from .model import PrototypicalNetwork


def _sliding_average(values: List[float], window: int) -> float:
    """Calculate sliding window average."""
    if len(values) < window:
        return sum(values) / len(values) if values else 0.0
    return sum(values[-window:]) / window


class Trainer:
    """Handles model training."""
    
    def __init__(
        self,
        model: PrototypicalNetwork,
        config: ModelConfig,
        device: torch.device
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.n_training_episodes // config.scheduler_step_divisor,
            gamma=0.7
        )
        
        self.losses: List[float] = []
        self.val_accuracies: List[float] = []
    
    def _train_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> float:
        """Train on single episode.
        
        Args:
            support_images: Support set images
            support_labels: Support set labels
            query_images: Query set images
            query_labels: Query set labels
            
        Returns:
            Loss value
        """
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        self.optimizer.zero_grad()
        scores = self.model(support_images, support_labels, query_images)
        loss = self.criterion(scores, query_labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate model.
        
        Args:
            val_loader: Validation loader
            
        Returns:
            Validation accuracy percentage
        """
        self.model.eval()
        total = correct = 0
        
        with torch.no_grad():
            for support_images, support_labels, query_images, query_labels, _ in val_loader:
                support_images = support_images.to(self.device)
                support_labels = support_labels.to(self.device)
                query_images = query_images.to(self.device)
                query_labels = query_labels.to(self.device)
                
                scores = self.model(support_images, support_labels, query_images)
                _, preds = torch.max(scores, 1)
                
                correct += (preds == query_labels).sum().item()
                total += len(query_labels)
        
        self.model.train()
        return 100 * correct / total
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None
    ) -> List[float]:
        """Train model.
        
        Args:
            train_loader: Training data
            val_loader: Optional validation data
            
        Returns:
            Training losses
        """
        self.model.train()
        self.losses.clear()
        self.val_accuracies.clear()
        
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        patience = self.config.early_stopping_patience
        
        with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
            for ep_idx, (support_imgs, support_lbls, query_imgs, query_lbls, _) in pbar:
                loss = self._train_episode(support_imgs, support_lbls, query_imgs, query_lbls)
                self.losses.append(loss)

                if (val_loader and ep_idx % self.config.validation_frequency == 0 and ep_idx > 0):
                    val_acc = self._validate(val_loader)
                    self.val_accuracies.append(val_acc)
                    
                    # Step scheduler after validation check
                    self.scheduler.step()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_state = self.model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    pbar.set_postfix({
                        'loss': _sliding_average(self.losses, self.config.log_frequency),
                        'val': f'{val_acc:.2f}%',
                        'best': f'{best_val_acc:.2f}%'
                    })
                    
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at episode {ep_idx}")
                        break
                else:
                    if ep_idx % self.config.log_frequency == 0:
                        avg_loss = _sliding_average(self.losses, self.config.log_frequency)
                        pbar.set_postfix(loss=avg_loss)
        
        if best_state:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model: {best_val_acc:.2f}%")
        
        return self.losses

