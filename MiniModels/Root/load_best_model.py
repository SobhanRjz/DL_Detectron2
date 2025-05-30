"""Utility script to load and use the best model from hyperparameter optimization."""

import torch
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt

from config import ModelConfig, DataConfig
from main_few_shot_learning import FewShotLearningPipeline
from models import ModelFactory


class BestModelLoader:
    """Utility class to load and use the best model from optimization."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        """Initialize the model loader.
        
        Args:
            outputs_dir: Directory containing the saved models
        """
        self.outputs_dir = Path(outputs_dir)
        self.best_model_dir = self.outputs_dir / "best_model"
        
        if not self.best_model_dir.exists():
            raise FileNotFoundError(
                f"Best model directory not found: {self.best_model_dir}\n"
                "Please run hyperparameter optimization first."
            )
    
    def load_model_info(self) -> Dict[str, Any]:
        """Load information about the best model.
        
        Returns:
            Dictionary containing model information
        """
        # Load trial info
        trial_info_path = self.best_model_dir / "trial_info.json"
        if trial_info_path.exists():
            with open(trial_info_path, 'r') as f:
                trial_info = json.load(f)
        else:
            trial_info = {}
        
        # Load model config
        model_config_path = self.best_model_dir / "model_config.json"
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = {}
        
        # Load best model summary if available
        summary_path = self.outputs_dir / "best_model_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        return {
            "trial_info": trial_info,
            "model_config": model_config,
            "summary": summary
        }
    
    def load_model(self, device: Optional[str] = None) -> torch.nn.Module:
        """Load the best model.
        
        Args:
            device: Device to load the model on ('cuda', 'cpu', or None for auto)
            
        Returns:
            Loaded PyTorch model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Try to load complete model first
        complete_model_path = self.best_model_dir / "complete_model.pth"
        if complete_model_path.exists():
            try:
                model = torch.load(complete_model_path, map_location=device)
                print(f"✅ Loaded complete model from {complete_model_path}")
                return model
            except Exception as e:
                print(f"⚠️ Failed to load complete model: {e}")
                print("Falling back to state dict loading...")
        
        # Fallback: load state dict
        model_path = self.best_model_dir / "model.pth"
        config_path = self.best_model_dir / "model_config.json"
        
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Required model files not found in {self.best_model_dir}\n"
                f"Expected: model.pth and model_config.json"
            )
        
        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Recreate model config
        model_config = ModelConfig()
        for key, value in config_dict.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        # Create model
        model = ModelFactory.create_prototypical_network(model_config)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        print(f"✅ Loaded model state dict from {model_path}")
        return model
    
    def load_configs(self) -> Tuple[ModelConfig, DataConfig]:
        """Load the model and data configurations.
        
        Returns:
            Tuple of (ModelConfig, DataConfig)
        """
        # Load model config
        model_config_path = self.best_model_dir / "model_config.json"
        with open(model_config_path, 'r') as f:
            model_config_dict = json.load(f)
        
        model_config = ModelConfig()
        for key, value in model_config_dict.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        # Load data config
        data_config_path = self.best_model_dir / "data_config.json"
        with open(data_config_path, 'r') as f:
            data_config_dict = json.load(f)
        
        data_config = DataConfig()
        for key, value in data_config_dict.items():
            if hasattr(data_config, key):
                setattr(data_config, key, value)
        
        return model_config, data_config
    
    def create_pipeline(self, device: Optional[str] = None) -> FewShotLearningPipeline:
        """Create a complete pipeline with the best model.
        
        Args:
            device: Device to use
            
        Returns:
            FewShotLearningPipeline with the best model loaded
        """
        # Load configurations
        model_config, data_config = self.load_configs()
        
        # Create pipeline
        pipeline = FewShotLearningPipeline(model_config, data_config)
        
        # Load the best model
        best_model = self.load_model(device)
        pipeline.model = best_model
        
        print(f"✅ Created pipeline with best model")
        return pipeline
    
    def evaluate_best_model(self, device: Optional[str] = None) -> float:
        """Evaluate the best model on the test set.
        
        Args:
            device: Device to use
            
        Returns:
            Test accuracy
        """
        pipeline = self.create_pipeline(device)
        accuracy = pipeline.evaluate()
        
        print(f"🎯 Best model test accuracy: {accuracy:.4f}")
        return accuracy
    
    def demonstrate_best_model(self, device: Optional[str] = None) -> None:
        """Run a demonstration of the best model with visualizations.
        
        Args:
            device: Device to use
        """
        pipeline = self.create_pipeline(device)
        
        print("🎭 Running best model demonstration...")
        pipeline.demonstrate_predictions()
        
        # Show model info
        info = self.load_model_info()
        if "trial_info" in info and info["trial_info"]:
            trial_info = info["trial_info"]
            print(f"\n📊 Best Model Information:")
            print(f"   Trial Number: {trial_info.get('trial_number', 'Unknown')}")
            print(f"   Accuracy: {trial_info.get('accuracy', 'Unknown'):.4f}")
            print(f"   Timestamp: {trial_info.get('timestamp', 'Unknown')}")
    
    def print_model_summary(self) -> None:
        """Print a summary of the best model."""
        info = self.load_model_info()
        
        print("🏆 BEST MODEL SUMMARY")
        print("=" * 50)
        
        # Trial information
        if "trial_info" in info and info["trial_info"]:
            trial_info = info["trial_info"]
            print(f"Trial Number: {trial_info.get('trial_number', 'Unknown')}")
            print(f"Accuracy: {trial_info.get('accuracy', 'Unknown'):.4f}")
            print(f"Timestamp: {trial_info.get('timestamp', 'Unknown')}")
        
        # Model configuration
        if "model_config" in info and info["model_config"]:
            config = info["model_config"]
            print(f"\n📋 Model Configuration:")
            print(f"   Learning Rate: {config.get('LEARNING_RATE', 'Unknown')}")
            print(f"   Weight Decay: {config.get('WEIGHT_DECAY', 'Unknown')}")
            print(f"   N-Shot: {config.get('N_SHOT', 'Unknown')}")
            print(f"   N-Query: {config.get('N_QUERY', 'Unknown')}")
            print(f"   Backbone: {config.get('BACKBONE_NAME', 'Unknown')}")
            print(f"   Freeze Backbone: {config.get('FREEZE_BACKBONE', 'Unknown')}")
            print(f"   Batch Size: {config.get('BATCH_SIZE', 'Unknown')}")
            print(f"   Augmentation: {config.get('AUGMENTATION_STRATEGY', 'Unknown')}")
        
        # File locations
        print(f"\n📁 Model Files:")
        print(f"   Directory: {self.best_model_dir}")
        print(f"   Complete Model: complete_model.pth")
        print(f"   State Dict: model.pth")
        print(f"   Configuration: model_config.json")
        print(f"   Trial Info: trial_info.json")


def load_and_evaluate_best_model(outputs_dir: str = "outputs", device: Optional[str] = None) -> float:
    """Quick function to load and evaluate the best model.
    
    Args:
        outputs_dir: Directory containing saved models
        device: Device to use
        
    Returns:
        Test accuracy
    """
    BaseDir = Path(__file__).parent.parent.parent
    outputs_dir = BaseDir / "MiniModels" / "Root" / "outputs"
    loader = BestModelLoader(outputs_dir)
    return loader.evaluate_best_model(device)


def demonstrate_best_model(outputs_dir: str = "outputs", device: Optional[str] = None) -> None:
    """Quick function to demonstrate the best model.
    
    Args:
        outputs_dir: Directory containing saved models
        device: Device to use
    """
    BaseDir = Path(__file__).parent.parent.parent
    outputs_dir = BaseDir / "MiniModels" / "Root" / "outputs"
    loader = BestModelLoader(outputs_dir)
    loader.demonstrate_best_model(device)


def print_best_model_info(outputs_dir: str = "outputs") -> None:
    """Quick function to print best model information.
    
    Args:
        outputs_dir: Directory containing saved models
    """
    BaseDir = Path(__file__).parent.parent.parent
    outputs_dir = BaseDir / "MiniModels" / "Root" / "outputs"
    loader = BestModelLoader(outputs_dir)
    loader.print_model_summary()


if __name__ == "__main__":
    """Example usage of the best model loader."""
    
    try:
        print("🔍 Loading best model information...")
        
        # Print model summary
        print_best_model_info()
        
        # Evaluate the model
        print("\n🧪 Evaluating best model...")
        accuracy = load_and_evaluate_best_model()
        
        # Demonstrate the model
        print("\n🎭 Demonstrating best model...")
        demonstrate_best_model()
        
        print(f"\n✅ Best model evaluation completed!")
        print(f"Final accuracy: {accuracy:.4f}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run hyperparameter optimization first to generate a best model.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc() 