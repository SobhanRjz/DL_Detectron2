"""Hyperparameter optimization using Weights & Biases sweeps."""

import wandb
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

from config import ModelConfig, DataConfig
from main_few_shot_learning import FewShotLearningPipeline


class WandBOptimizer:
    """Hyperparameter optimization using Weights & Biases."""
    
    def __init__(self, project_name: str = "few-shot-learning", entity: Optional[str] = None):
        """Initialize the W&B optimizer.
        
        Args:
            project_name: W&B project name
            entity: W&B entity (username/team)
        """
        self.project_name = project_name
        self.entity = entity
        
    def create_sweep_config(self) -> Dict[str, Any]:
        """Create sweep configuration for W&B.
        
        Returns:
            Sweep configuration dictionary
        """
        sweep_config = {
            'method': 'bayes',  # Can be 'grid', 'random', 'bayes'
            'metric': {
                'name': 'accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-2
                },
                'weight_decay': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-6,
                    'max': 1e-2
                },
                'n_shot': {
                    'values': [1, 2, 3, 5, 8, 10]
                },
                'n_query': {
                    'values': [1, 2, 3, 5]
                },
                'n_training_episodes': {
                    'values': [100, 200, 300, 500, 800, 1000]
                },
                'validation_frequency': {
                    'values': [20, 30, 50, 80, 100]
                },
                'backbone_name': {
                    'values': ['resnet18', 'resnet34', 'resnet50']
                },
                'freeze_backbone': {
                    'values': [True, False]
                },
                'batch_size': {
                    'values': [64, 128, 256]
                }
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 3,
                'max_iter': 50,
                's': 2
            }
        }
        
        return sweep_config
    
    def train_with_sweep(self):
        """Training function for W&B sweep."""
        # Initialize wandb run
        with wandb.init() as run:
            # Get hyperparameters from sweep
            config = wandb.config
            
            # Create model and data configs
            model_config = ModelConfig()
            data_config = DataConfig()
            
            # Apply sweep parameters
            model_config.LEARNING_RATE = config.learning_rate
            model_config.WEIGHT_DECAY = config.weight_decay
            model_config.N_SHOT = config.n_shot
            model_config.N_QUERY = config.n_query
            model_config.N_TRAINING_EPISODES = config.n_training_episodes
            model_config.VALIDATION_FREQUENCY = config.validation_frequency
            model_config.BACKBONE_NAME = config.backbone_name
            model_config.FREEZE_BACKBONE = config.freeze_backbone
            model_config.BATCH_SIZE = config.batch_size
            
            # Log configuration
            wandb.config.update({
                "model_config": model_config.__dict__,
                "data_config": data_config.__dict__
            })
            
            try:
                # Create and train pipeline
                pipeline = FewShotLearningPipeline(model_config, data_config)
                
                # Train with logging
                self._train_with_logging(pipeline)
                
                # Final evaluation
                accuracy = pipeline.evaluate()
                
                # Log final results
                wandb.log({
                    "accuracy": accuracy,
                    "final_accuracy": accuracy
                })
                
                print(f"Final accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"Training failed: {e}")
                wandb.log({"error": str(e)})
                raise
            finally:
                plt.close('all')
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _train_with_logging(self, pipeline: FewShotLearningPipeline):
        """Train pipeline with W&B logging.
        
        Args:
            pipeline: Few-shot learning pipeline
        """
        train_loader = pipeline.data_manager.get_train_loader()
        val_loader = pipeline.data_manager.get_validation_loader()
        
        # Modify trainer to log to wandb
        original_train = pipeline.trainer.train
        
        def logged_train(train_loader, val_loader):
            # Store original validation method
            original_validate = pipeline.trainer._validate_model
            
            def logged_validate(val_loader, episode):
                accuracy = original_validate(val_loader, episode)
                wandb.log({
                    "episode": episode,
                    "validation_accuracy": accuracy
                })
                return accuracy
            
            # Replace validation method
            pipeline.trainer._validate_model = logged_validate
            
            # Run training
            result = original_train(train_loader, val_loader)
            
            # Restore original method
            pipeline.trainer._validate_model = original_validate
            
            return result
        
        # Replace train method and run
        pipeline.trainer.train = logged_train
        pipeline.trainer.train(train_loader, val_loader)
    
    def run_sweep(self, count: int = 20) -> str:
        """Run hyperparameter sweep.
        
        Args:
            count: Number of runs in the sweep
            
        Returns:
            Sweep ID
        """
        # Create sweep configuration
        sweep_config = self.create_sweep_config()
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=self.project_name,
            entity=self.entity
        )
        
        print(f"Starting sweep with ID: {sweep_id}")
        print(f"View at: https://wandb.ai/{self.entity or 'your-username'}/{self.project_name}/sweeps/{sweep_id}")
        
        # Run sweep
        wandb.agent(sweep_id, self.train_with_sweep, count=count)
        
        return sweep_id
    
    def save_sweep_config(self, filename: str = "wandb_sweep_config.yaml"):
        """Save sweep configuration to file.
        
        Args:
            filename: Output filename
        """
        config = self.create_sweep_config()
        
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Sweep configuration saved to {filename}")
        print(f"You can run it manually with: wandb sweep {filename}")


def run_wandb_optimization(
    project_name: str = "few-shot-learning",
    entity: Optional[str] = None,
    count: int = 20
) -> str:
    """Run hyperparameter optimization with W&B.
    
    Args:
        project_name: W&B project name
        entity: W&B entity
        count: Number of runs
        
    Returns:
        Sweep ID
    """
    optimizer = WandBOptimizer(project_name, entity)
    
    # Save configuration for manual use
    optimizer.save_sweep_config()
    
    # Run sweep
    sweep_id = optimizer.run_sweep(count)
    
    print(f"\nSweep completed!")
    print(f"Sweep ID: {sweep_id}")
    print(f"View results at: https://wandb.ai/{entity or 'your-username'}/{project_name}")
    
    return sweep_id


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run W&B hyperparameter sweep")
    parser.add_argument("--project", default="few-shot-learning", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--count", type=int, default=10, help="Number of runs")
    
    args = parser.parse_args()
    
    run_wandb_optimization(args.project, args.entity, args.count) 