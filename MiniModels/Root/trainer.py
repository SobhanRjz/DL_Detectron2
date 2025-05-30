"""Training utilities for few-shot learning."""

from typing import Tuple, List
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ModelConfig
from models import PrototypicalNetworks
from utils import sliding_average


class FewShotTrainer:
    """Trainer for few-shot learning models."""
    
    def __init__(
        self, 
        model: PrototypicalNetworks, 
        config: ModelConfig,
        device: torch.device
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Model configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.N_TRAINING_EPISODES // 4, 
            gamma=0.7
        )
        
        self.training_losses: List[float] = []
        self.validation_accuracies: List[float] = []
    
    def fit_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> float:
        """Train on a single episode.
        
        Args:
            support_images: Support set images
            support_labels: Support set labels
            query_images: Query set images
            query_labels: Query set labels
            
        Returns:
            Loss value for this episode
        """
        # Move to device
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        classification_scores = self.model(
            support_images, support_labels, query_images
        )
        
        # Backward pass
        loss = self.criterion(classification_scores, query_labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation accuracy
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
            ) in val_loader:
                
                support_images = support_images.to(self.device)
                support_labels = support_labels.to(self.device)
                query_images = query_images.to(self.device)
                query_labels = query_labels.to(self.device)
                
                scores = self.model(support_images, support_labels, query_images)
                _, predicted_labels = torch.max(scores.data, 1)
                
                correct = (predicted_labels == query_labels).sum().item()
                total = len(query_labels)
                
                total_predictions += total
                correct_predictions += correct
        
        accuracy = 100 * correct_predictions / total_predictions
        self.model.train()
        return accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> List[float]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            List of training losses
        """
        self.model.train()
        self.training_losses.clear()
        self.validation_accuracies.clear()
        
        best_val_acc = 0.0
        best_model_state = None
        patience = 5  # Early stopping patience
        patience_counter = 0
        
        with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in pbar:
                
                loss_value = self.fit_episode(
                    support_images, support_labels, query_images, query_labels
                )
                self.training_losses.append(loss_value)
                
                # Update learning rate
                self.scheduler.step()

                # Validation
                if (val_loader is not None and 
                    episode_index % self.config.VALIDATION_FREQUENCY == 0 and 
                    episode_index > 0):
                    
                    val_acc = self.validate(val_loader)
                    self.validation_accuracies.append(val_acc)
                    
                    # Save best model and early stopping
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = self.model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    pbar.set_postfix({
                        'loss': sliding_average(self.training_losses, self.config.LOG_UPDATE_FREQUENCY),
                        'val_acc': f'{val_acc:.2f}%',
                        'best_val': f'{best_val_acc:.2f}%',
                        'patience': f'{patience_counter}/{patience}'
                    })
                    
                    # Early stopping
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at episode {episode_index}")
                        break
                        
                else:
                    if episode_index % self.config.LOG_UPDATE_FREQUENCY == 0:
                        avg_loss = sliding_average(
                            self.training_losses, 
                            self.config.LOG_UPDATE_FREQUENCY
                        )
                        pbar.set_postfix(loss=avg_loss)
        
        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
        
        return self.training_losses 