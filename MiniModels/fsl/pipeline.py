"""Complete training and evaluation pipeline."""

from typing import Optional
import torch

from .core import ModelConfig, DataConfig, ModelFactory, DatasetManager, Trainer, Evaluator
from .utils import set_seed, get_device


class Pipeline:
    """Orchestrates training and evaluation workflow."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig
    ) -> None:
        """Initialize pipeline.
        
        Args:
            model_config: Model configuration
            data_config: Data configuration
        """
        self.model_config = model_config
        self.data_config = data_config
        
        set_seed(model_config.random_seed)
        self.device = get_device(model_config.device)
        
        print(f"Device: {self.device}")
        
        self.data_manager = DatasetManager(data_config, model_config)
        self.model = ModelFactory.create_model(model_config)
        self.trainer = Trainer(self.model, model_config, self.device)
        self.evaluator = Evaluator(self.model, self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
    
    def train(self) -> None:
        """Train model."""
        print("Training...")
        train_loader = self.data_manager.get_train_loader()
        val_loader = self.data_manager.get_validation_loader()
        self.trainer.train(train_loader, val_loader)
        print("Training complete")
    
    def evaluate(self) -> float:
        """Evaluate model.
        
        Returns:
            Test accuracy
        """
        print("Evaluating...")
        test_loader = self.data_manager.get_test_loader()
        return self.evaluator.evaluate(test_loader)
    
    def run(self) -> float:
        """Run complete pipeline.
        
        Returns:
            Final accuracy
        """
        print("=== Initial Evaluation ===")
        initial_acc = self.evaluate()
        
        print("\n=== Training ===")
        self.train()
        
        print("\n=== Final Evaluation ===")
        final_acc = self.evaluate()
        
        print(f"\nImprovement: {final_acc - initial_acc:.2f}%")
        return final_acc
    
    def save_model(self, path: str) -> None:
        """Save model weights.

        Args:
            path: Save path
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Only save state dict for security and compatibility
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model weights.

        Args:
            path: Model path
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

