"""Hyperparameter optimization for few-shot learning using Optuna."""

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for optimization
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import json
import logging
from pathlib import Path
from datetime import datetime
import shutil

from config import ModelConfig, DataConfig
from main_few_shot_learning import FewShotLearningPipeline


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage_url: Optional[str] = None,
        save_best_model: bool = True
    ) -> None:
        """Initialize the hyperparameter optimizer.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds for optimization
            study_name: Name of the study for persistence
            storage_url: Database URL for study persistence
            save_best_model: Whether to save the best model found
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"few_shot_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_url = storage_url
        self.save_best_model = save_best_model
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create results and outputs directories
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Track best model information
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.best_trial_number = None
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Tuple[ModelConfig, DataConfig]:
        """Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Tuple of model and data configurations
        """
        # Create base configs
        model_config = ModelConfig()
        data_config = DataConfig()
        
        # Suggest model hyperparameters
        model_config.LEARNING_RATE = trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        )
        model_config.WEIGHT_DECAY = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        )
        model_config.N_SHOT = trial.suggest_int("n_shot", 1, 5)
        model_config.N_QUERY = trial.suggest_int("n_query", 1, 3)
        model_config.N_TRAINING_EPISODES = trial.suggest_int("n_training_episodes", 100, 1000)
        model_config.VALIDATION_FREQUENCY = trial.suggest_int("validation_frequency", 20, 100)
        
        # Suggest backbone architecture
        model_config.BACKBONE_NAME = trial.suggest_categorical(
            "backbone_name", ["resnet18", "resnet34", "resnet50"]
        )
        
        # Suggest whether to freeze backbone
        model_config.FREEZE_BACKBONE = trial.suggest_categorical("freeze_backbone", [True, False])
        
        # Suggest batch size
        model_config.BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128, 256])
        
        # Suggest augmentation strategy
        model_config.AUGMENTATION_STRATEGY = trial.suggest_categorical(
            "augmentation_strategy", ["none", "basic", "medium", "strong"]
        )
        
        return model_config, data_config
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Accuracy to maximize
        """
        try:
            # Get suggested hyperparameters
            model_config, data_config = self.suggest_hyperparameters(trial)
            
            # Log trial parameters
            self.logger.info(f"Trial {trial.number}: {trial.params}")
            
            # Create and run pipeline
            pipeline = FewShotLearningPipeline(model_config, data_config)
            
            # Train and evaluate
            pipeline.train()
            accuracy = pipeline.evaluate()
            
            # Log results
            self.logger.info(f"Trial {trial.number} accuracy: {accuracy:.4f}")
            
            # Save model if it's the best so far
            if self.save_best_model and accuracy > self.best_accuracy:
                self._save_best_model(trial, pipeline, accuracy, model_config, data_config)
            
            # Save trial results
            self._save_trial_results(trial, accuracy, model_config, data_config)
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        finally:
            # Clean up
            plt.close('all')
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _save_best_model(
        self, 
        trial: optuna.Trial, 
        pipeline: FewShotLearningPipeline, 
        accuracy: float,
        model_config: ModelConfig,
        data_config: DataConfig
    ) -> None:
        """Save the best model found so far.
        
        Args:
            trial: Current trial
            pipeline: Pipeline containing the trained model
            accuracy: Model accuracy
            model_config: Model configuration
            data_config: Data configuration
        """
        self.best_accuracy = accuracy
        self.best_trial_number = trial.number
        
        # Create model directory
        model_dir = self.outputs_dir / f"best_model_trial_{trial.number}"
        model_dir.mkdir(exist_ok=True)
        
        # Save model state dict
        model_path = model_dir / "model.pth"
        torch.save(pipeline.model.state_dict(), model_path)
        
        # Save complete model (for easy loading)
        complete_model_path = model_dir / "complete_model.pth"
        torch.save(pipeline.model, complete_model_path)
        
        # Save model configuration
        config_path = model_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config.__dict__, f, indent=2, default=str)
        
        # Save data configuration
        data_config_path = model_dir / "data_config.json"
        with open(data_config_path, 'w') as f:
            json.dump(data_config.__dict__, f, indent=2, default=str)
        
        # Save trial information
        trial_info = {
            "trial_number": trial.number,
            "accuracy": accuracy,
            "params": trial.params,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "complete_model_path": str(complete_model_path)
        }
        
        trial_info_path = model_dir / "trial_info.json"
        with open(trial_info_path, 'w') as f:
            json.dump(trial_info, f, indent=2, default=str)
        
        # Update best model path
        self.best_model_path = model_dir
        
        # Create a symlink/copy to "best_model" for easy access
        best_model_link = self.outputs_dir / "best_model"
        if best_model_link.exists():
            if best_model_link.is_symlink():
                best_model_link.unlink()
            else:
                shutil.rmtree(best_model_link)
        
        # Copy instead of symlink for better compatibility
        shutil.copytree(model_dir, best_model_link)
        
        self.logger.info(f"💾 New best model saved! Accuracy: {accuracy:.4f}, Trial: {trial.number}")
        self.logger.info(f"   Model saved to: {model_dir}")
        self.logger.info(f"   Best model link: {best_model_link}")
    
    def optimize(self) -> optuna.Study:
        """Run hyperparameter optimization.
        
        Returns:
            Completed study object
        """
        # Create sampler and pruner
        sampler = TPESampler(n_startup_trials=10, n_ei_candidates=24)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=1)
        
        # Create or load study
        if self.storage_url:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner
            )
        
        self.logger.info(f"Starting optimization with {self.n_trials} trials...")
        if self.save_best_model:
            self.logger.info(f"Best models will be saved to: {self.outputs_dir}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[self._callback]
        )
        
        # Save results and final best model info
        self._save_study_results(study)
        
        return study
    
    def _callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback function called after each trial.
        
        Args:
            study: Current study
            trial: Completed trial
        """
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.logger.info(f"Trial {trial.number} completed with value: {trial.value:.4f}")
            self.logger.info(f"Best value so far: {study.best_value:.4f}")
            self.logger.info(f"Best params so far: {study.best_params}")
    
    def _save_trial_results(
        self,
        trial: optuna.Trial,
        accuracy: float,
        model_config: ModelConfig,
        data_config: DataConfig
    ) -> None:
        """Save individual trial results.
        
        Args:
            trial: Trial object
            accuracy: Trial accuracy
            model_config: Model configuration used
            data_config: Data configuration used
        """
        trial_data = {
            "trial_number": trial.number,
            "accuracy": accuracy,
            "params": trial.params,
            "model_config": model_config.__dict__,
            "data_config": data_config.__dict__,
            "datetime": datetime.now().isoformat(),
            "is_best_model": accuracy == self.best_accuracy
        }
        
        trial_file = self.results_dir / f"trial_{trial.number:03d}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_data, f, indent=2, default=str)
    
    def _save_study_results(self, study: optuna.Study) -> None:
        """Save complete study results.
        
        Args:
            study: Completed study
        """
        # Save study summary
        study_summary = {
            "study_name": self.study_name,
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "datetime": datetime.now().isoformat(),
            "best_model_info": {
                "accuracy": self.best_accuracy,
                "trial_number": self.best_trial_number,
                "model_path": str(self.best_model_path) if self.best_model_path else None,
                "saved": self.save_best_model
            }
        }
        
        summary_file = self.results_dir / f"{self.study_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(study_summary, f, indent=2, default=str)
        
        # Save trials dataframe
        df = study.trials_dataframe()
        df.to_csv(self.results_dir / f"{self.study_name}_trials.csv", index=False)
        
        # Create optimization plots
        self._create_optimization_plots(study)
        
        # Save final best model summary
        if self.save_best_model and self.best_model_path:
            self._create_best_model_summary()
        
        self.logger.info(f"Results saved to {self.results_dir}")
        if self.save_best_model:
            self.logger.info(f"Best model saved to {self.outputs_dir}/best_model")
    
    def _create_best_model_summary(self) -> None:
        """Create a summary of the best model for easy reference."""
        summary = {
            "best_model_summary": {
                "accuracy": self.best_accuracy,
                "trial_number": self.best_trial_number,
                "model_directory": str(self.best_model_path),
                "files": {
                    "model_state_dict": "model.pth",
                    "complete_model": "complete_model.pth",
                    "model_config": "model_config.json",
                    "data_config": "data_config.json",
                    "trial_info": "trial_info.json"
                },
                "usage_instructions": {
                    "load_model_state_dict": "torch.load('outputs/best_model/model.pth')",
                    "load_complete_model": "torch.load('outputs/best_model/complete_model.pth')",
                    "load_config": "json.load(open('outputs/best_model/model_config.json'))"
                },
                "timestamp": datetime.now().isoformat()
            }
        }
        
        summary_file = self.outputs_dir / "best_model_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _create_optimization_plots(self, study: optuna.Study) -> None:
        """Create optimization visualization plots.
        
        Args:
            study: Completed study
        """
        try:
            # Plot optimization history
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_html(self.results_dir / f"{self.study_name}_history.html")
            
            # Plot parameter importances
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_html(self.results_dir / f"{self.study_name}_importances.html")
            
            # Plot parallel coordinate
            fig3 = optuna.visualization.plot_parallel_coordinate(study)
            fig3.write_html(self.results_dir / f"{self.study_name}_parallel.html")
            
            # Plot slice
            fig4 = optuna.visualization.plot_slice(study)
            fig4.write_html(self.results_dir / f"{self.study_name}_slice.html")
            
            self.logger.info("Optimization plots created successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not create plots: {e}")
    
    def load_best_config(self, study: optuna.Study) -> Tuple[ModelConfig, DataConfig]:
        """Load the best configuration from a study.
        
        Args:
            study: Completed study
            
        Returns:
            Best model and data configurations
        """
        best_params = study.best_params
        
        # Create configs with best parameters
        model_config = ModelConfig()
        data_config = DataConfig()
        
        # Apply best parameters
        if "learning_rate" in best_params:
            model_config.LEARNING_RATE = best_params["learning_rate"]
        if "weight_decay" in best_params:
            model_config.WEIGHT_DECAY = best_params["weight_decay"]
        if "n_shot" in best_params:
            model_config.N_SHOT = best_params["n_shot"]
        if "n_query" in best_params:
            model_config.N_QUERY = best_params["n_query"]
        if "n_training_episodes" in best_params:
            model_config.N_TRAINING_EPISODES = best_params["n_training_episodes"]
        if "validation_frequency" in best_params:
            model_config.VALIDATION_FREQUENCY = best_params["validation_frequency"]
        if "backbone_name" in best_params:
            model_config.BACKBONE_NAME = best_params["backbone_name"]
        if "freeze_backbone" in best_params:
            model_config.FREEZE_BACKBONE = best_params["freeze_backbone"]
        if "batch_size" in best_params:
            model_config.BATCH_SIZE = best_params["batch_size"]
        if "augmentation_strategy" in best_params:
            model_config.AUGMENTATION_STRATEGY = best_params["augmentation_strategy"]
        
        return model_config, data_config
    
    def load_best_model(self, device: Optional[str] = None) -> torch.nn.Module:
        """Load the best model from the outputs directory.
        
        Args:
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        if not self.best_model_path or not self.best_model_path.exists():
            # Try to load from best_model directory
            best_model_dir = self.outputs_dir / "best_model"
            if not best_model_dir.exists():
                raise FileNotFoundError("No best model found. Run optimization first.")
            self.best_model_path = best_model_dir
        
        # Load complete model
        complete_model_path = self.best_model_path / "complete_model.pth"
        if complete_model_path.exists():
            model = torch.load(complete_model_path, map_location=device)
            self.logger.info(f"Loaded complete model from {complete_model_path}")
            return model
        
        # Fallback: load state dict (requires model architecture)
        model_path = self.best_model_path / "model.pth"
        if model_path.exists():
            # Load configuration to recreate model
            config_path = self.best_model_path / "model_config.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Recreate model config
            model_config = ModelConfig()
            for key, value in config_dict.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            
            # Create model using factory
            from models import ModelFactory
            model = ModelFactory.create_prototypical_network(model_config)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
            self.logger.info(f"Loaded model state dict from {model_path}")
            return model
        
        raise FileNotFoundError(f"No model files found in {self.best_model_path}")


def run_hyperparameter_optimization(
    n_trials: int = 50,
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
    save_best_model: bool = True
) -> optuna.Study:
    """Run hyperparameter optimization.
    
    Args:
        n_trials: Number of trials to run
        timeout: Timeout in seconds
        study_name: Name for the study
        save_best_model: Whether to save the best model
        
    Returns:
        Completed study
    """
    optimizer = HyperparameterOptimizer(
        n_trials=n_trials,
        timeout=timeout,
        study_name=study_name,
        save_best_model=save_best_model
    )
    
    study = optimizer.optimize()
    
    print(f"\nOptimization completed!")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    if save_best_model:
        print(f"\n💾 Best model saved to: outputs/best_model/")
        print(f"   Model files:")
        print(f"   - complete_model.pth (full model)")
        print(f"   - model.pth (state dict)")
        print(f"   - model_config.json (configuration)")
        print(f"   - trial_info.json (trial details)")
    
    return study


if __name__ == "__main__":
    # Example usage
    study = run_hyperparameter_optimization(
        n_trials=20, 
        study_name="few_shot_optimization_with_model_saving",
        save_best_model=True
    ) 