"""Main few-shot learning pipeline."""

from typing import Optional
import torch
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better Windows compatibility
import matplotlib.pyplot as plt

from config import ModelConfig, DataConfig
from data_manager import DataManager
from models import ModelFactory
from trainer import FewShotTrainer
from evaluator import FewShotEvaluator
from utils import set_random_seeds, get_device, Visualizer


class FewShotLearningPipeline:
    """Complete pipeline for few-shot learning."""
    
    def __init__(
        self, 
        model_config: Optional[ModelConfig] = None,
        data_config: Optional[DataConfig] = None
    ) -> None:
        """Initialize the pipeline.
        
        Args:
            model_config: Model configuration
            data_config: Data configuration
        """
        self.model_config = model_config or ModelConfig()
        self.data_config = data_config or DataConfig()
        
        # Set random seeds for reproducibility
        set_random_seeds(self.model_config.RANDOM_SEED)
        
        # Setup device
        self.device = get_device(self.model_config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.data_manager = DataManager(self.data_config, self.model_config)
        self.model = ModelFactory.create_prototypical_network(self.model_config)
        self.trainer = FewShotTrainer(self.model, self.model_config, self.device)
        self.evaluator = FewShotEvaluator(self.model, self.device)
        self.visualizer = Visualizer()
        
        print("Pipeline initialized successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train(self) -> None:
        """Train the model."""
        print("Starting training...")
        train_loader = self.data_manager.get_train_loader()
        val_loader = self.data_manager.get_validation_loader()
        self.trainer.train(train_loader, val_loader)
        print("Training completed!")
    
    def evaluate(self) -> float:
        """Evaluate the model.
        
        Returns:
            Test accuracy
        """
        print("Evaluating model...")
        test_loader = self.data_manager.get_test_loader()
        accuracy = self.evaluator.evaluate(test_loader)
        return accuracy
    
    def demonstrate_predictions(self) -> None:
        """Demonstrate model predictions with visualization."""
        print("Generating prediction demonstration...")
        test_loader = self.data_manager.get_test_loader()
        
        # Get one batch for demonstration
        (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) = next(iter(test_loader))
        
        # Show support and query images
        print("Displaying support and query images...")
        self.visualizer.plot_support_and_query(
            support_images, 
            query_images, 
            self.model_config.N_SHOT, 
            self.model_config.N_QUERY
        )
        
        # Make predictions
        predicted_labels = self.evaluator.predict(
            support_images, support_labels, query_images
        )
        
        # Show predictions
        print("Displaying predictions...")
        self.visualizer.plot_predictions(
            query_images,
            predicted_labels,
            class_ids,
            self.data_manager.test_set.classes
        )
        
        # Print ground truth vs predictions
        print("\nGround Truth / Predicted:")
        for i, (gt_label, pred_label) in enumerate(zip(query_labels, predicted_labels)):
            gt_class = self.data_manager.test_set.classes[class_ids[gt_label]]
            pred_class = self.data_manager.test_set.classes[class_ids[pred_label]]
            print(f"{gt_class} / {pred_class}")
    
    def run_complete_pipeline(self) -> float:
        """Run the complete training and evaluation pipeline.
        
        Returns:
            Final test accuracy
        """
        try:
            # Initial evaluation (before training)
            print("=== Initial Evaluation (Before Training) ===")
            initial_accuracy = self.evaluate()
            
            # Training
            print("\n=== Training ===")
            self.train()
            
            # Final evaluation
            print("\n=== Final Evaluation (After Training) ===")
            final_accuracy = self.evaluate()
            
            # Demonstration
            print("\n=== Prediction Demonstration ===")
            self.demonstrate_predictions()
            
            print(f"\nImprovement: {final_accuracy - initial_accuracy:.2f}%")
            return final_accuracy
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            return 0.0
        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            raise
        finally:
            # Ensure all matplotlib figures are closed
            plt.close('all')


def main() -> None:
    """Main function to run the few-shot learning pipeline."""
    try:
        # Create custom configurations if needed
        model_config = ModelConfig()
        data_config = DataConfig()
        
        # Initialize and run pipeline
        pipeline = FewShotLearningPipeline(model_config, data_config)
        final_accuracy = pipeline.run_complete_pipeline()
        
        print(f"Final accuracy: {final_accuracy:.2f}%")
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise
    finally:
        # Ensure all plots are closed
        plt.close('all')


if __name__ == "__main__":
    main() 