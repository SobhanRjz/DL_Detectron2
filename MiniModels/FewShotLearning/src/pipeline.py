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

        # Validate configuration compatibility with dataset
        self._validate_config_compatibility()

    def _validate_config_compatibility(self) -> None:
        """Validate that model configuration is compatible with dataset.

        Raises:
            ValueError: If configuration is incompatible with dataset
        """
        print("Validating configuration compatibility...")

        # Check if we have enough samples per class for n_shot + n_query
        min_samples_per_class = self.model_config.n_shot + self.model_config.n_query

        # Get class counts from training set
        train_class_counts = {}
        for _, label in self.data_manager.train_set.samples:
            train_class_counts[label] = train_class_counts.get(label, 0) + 1

        # Get class counts from test set
        test_class_counts = {}
        for _, label in self.data_manager.test_set.samples:
            test_class_counts[label] = test_class_counts.get(label, 0) + 1

        # Check training set
        train_classes_below_threshold = [
            class_id for class_id, count in train_class_counts.items()
            if count < min_samples_per_class
        ]

        # Check test set
        test_classes_below_threshold = [
            class_id for class_id, count in test_class_counts.items()
            if count < min_samples_per_class
        ]

        if train_classes_below_threshold or test_classes_below_threshold:
            error_msg = f"Configuration incompatible with dataset. Need at least {min_samples_per_class} samples per class (n_shot={self.model_config.n_shot} + n_query={self.model_config.n_query}).\n"

            if train_classes_below_threshold:
                error_msg += f"Training set classes with insufficient samples: {train_classes_below_threshold}\n"
                for class_id in train_classes_below_threshold[:5]:  # Show first 5
                    error_msg += f"  Class {class_id}: {train_class_counts[class_id]} samples\n"

            if test_classes_below_threshold:
                error_msg += f"Test set classes with insufficient samples: {test_classes_below_threshold}\n"
                for class_id in test_classes_below_threshold[:5]:  # Show first 5
                    error_msg += f"  Class {class_id}: {test_class_counts[class_id]} samples\n"

            raise ValueError(error_msg)

        # Check number of classes
        n_train_classes = len(train_class_counts)
        n_test_classes = len(test_class_counts)

        if n_train_classes < self.model_config.n_way:
            raise ValueError(f"Not enough classes in training set. Need {self.model_config.n_way}, got {n_train_classes}")

        if n_test_classes < self.model_config.n_way:
            raise ValueError(f"Not enough classes in test set. Need {self.model_config.n_way}, got {n_test_classes}")

        print(f"✓ Configuration validated: {len(train_class_counts)} train classes, {len(test_class_counts)} test classes")
        print(f"✓ Minimum samples per class: {min_samples_per_class} (meets requirement)")

    def print_dataset_info(self) -> None:
        """Print detailed information about the dataset."""
        print("\n=== Dataset Information ===")

        # Get class counts from training set
        train_class_counts = {}
        for _, label in self.data_manager.train_set.samples:
            train_class_counts[label] = train_class_counts.get(label, 0) + 1

        # Get class counts from test set
        test_class_counts = {}
        for _, label in self.data_manager.test_set.samples:
            test_class_counts[label] = test_class_counts.get(label, 0) + 1

        print(f"Training set: {len(train_class_counts)} classes, {len(self.data_manager.train_set)} total samples")
        print(f"Test set: {len(test_class_counts)} classes, {len(self.data_manager.test_set)} total samples")

        # Show sample distribution
        print("\nTraining set class distribution:")
        for class_id in sorted(train_class_counts.keys()):
            print(f"  Class {class_id}: {train_class_counts[class_id]} samples")

        print("\nTest set class distribution:")
        for class_id in sorted(test_class_counts.keys()):
            print(f"  Class {class_id}: {test_class_counts[class_id]} samples")

        # Show current config requirements
        min_samples = self.model_config.n_shot + self.model_config.n_query
        print(f"\nCurrent config requirements:")
        print(f"  n_way: {self.model_config.n_way}")
        print(f"  n_shot: {self.model_config.n_shot}")
        print(f"  n_query: {self.model_config.n_query}")
        print(f"  Minimum samples per class needed: {min_samples}")

        # Check compatibility
        train_min = min(train_class_counts.values()) if train_class_counts else 0
        test_min = min(test_class_counts.values()) if test_class_counts else 0

        print(f"\nCompatibility check:")
        print(f"  Training set - min samples per class: {train_min} (need ≥{min_samples}) {'✓' if train_min >= min_samples else '✗'}")
        print(f"  Test set - min samples per class: {test_min} (need ≥{min_samples}) {'✓' if test_min >= min_samples else '✗'}")
        print(f"  Training classes: {len(train_class_counts)} (need ≥{self.model_config.n_way}) {'✓' if len(train_class_counts) >= self.model_config.n_way else '✗'}")
        print(f"  Test classes: {len(test_class_counts)} (need ≥{self.model_config.n_way}) {'✓' if len(test_class_counts) >= self.model_config.n_way else '✗'}")

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

