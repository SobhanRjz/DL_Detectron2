#!/usr/bin/env python3
"""Example script for running hyperparameter optimization."""

import argparse
import sys
from pathlib import Path

from hyperparameter_optimizer import run_hyperparameter_optimization, HyperparameterOptimizer
from main_few_shot_learning import FewShotLearningPipeline
from load_best_model import print_best_model_info, load_and_evaluate_best_model


def main():
    """Main function to run hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for few-shot learning")
    
    parser.add_argument(
        "--trials", 
        type=int, 
        default=20, 
        help="Number of optimization trials (default: 20)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=None, 
        help="Timeout in seconds (default: None)"
    )
    parser.add_argument(
        "--study-name", 
        type=str, 
        default=None, 
        help="Name for the study (default: auto-generated)"
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true", 
        help="Run a quick test with only 3 trials"
    )
    parser.add_argument(
        "--no-save-model", 
        action="store_true", 
        help="Don't save the best model during optimization"
    )
    parser.add_argument(
        "--evaluate-best", 
        action="store_true", 
        help="Evaluate the best model after optimization"
    )
    parser.add_argument(
        "--show-best-info", 
        action="store_true", 
        help="Show information about the current best model"
    )
    
    args = parser.parse_args()
    
    # Handle show best info option
    if args.show_best_info:
        try:
            print_best_model_info()
            return
        except FileNotFoundError:
            print("❌ No best model found. Run optimization first.")
            return
    
    # Adjust for quick test
    if args.quick_test:
        args.trials = 3
        print("Running quick test with 3 trials...")
    
    save_best_model = not args.no_save_model
    
    try:
        print(f"Starting hyperparameter optimization...")
        print(f"Number of trials: {args.trials}")
        print(f"Timeout: {args.timeout}")
        print(f"Study name: {args.study_name or 'auto-generated'}")
        print(f"Save best model: {save_best_model}")
        print("-" * 50)
        
        # Run optimization
        study = run_hyperparameter_optimization(
            n_trials=args.trials,
            timeout=args.timeout,
            study_name=args.study_name,
            save_best_model=save_best_model
        )
        
        # Print results
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Best accuracy: {study.best_value:.4f}")
        print(f"Best parameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        # Show model saving information
        if save_best_model:
            print("\n" + "="*50)
            print("SAVED MODEL INFORMATION")
            print("="*50)
            outputs_dir = Path("outputs")
            best_model_dir = outputs_dir / "best_model"
            
            if best_model_dir.exists():
                print(f"✅ Best model saved to: {best_model_dir}")
                print(f"📁 Model files:")
                print(f"   - complete_model.pth (full model for easy loading)")
                print(f"   - model.pth (state dict)")
                print(f"   - model_config.json (model configuration)")
                print(f"   - data_config.json (data configuration)")
                print(f"   - trial_info.json (optimization trial details)")
                
                # Show best model summary
                try:
                    print("\n📊 Best Model Summary:")
                    print_best_model_info()
                except Exception as e:
                    print(f"⚠️ Could not load model info: {e}")
            else:
                print("⚠️ Best model directory not found")
        
        # Evaluate best model if requested
        if args.evaluate_best and save_best_model:
            print("\n" + "="*50)
            print("EVALUATING BEST MODEL")
            print("="*50)
            try:
                accuracy = load_and_evaluate_best_model()
                print(f"✅ Best model evaluation completed with accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"❌ Failed to evaluate best model: {e}")
        
        # Show how to use the best configuration
        print("\n" + "="*50)
        print("HOW TO USE BEST CONFIGURATION")
        print("="*50)
        
        if save_best_model:
            print("To load and use the best model:")
            print("```python")
            print("from load_best_model import BestModelLoader")
            print("")
            print("# Load the best model")
            print("loader = BestModelLoader()")
            print("model = loader.load_model()")
            print("pipeline = loader.create_pipeline()")
            print("")
            print("# Evaluate the model")
            print("accuracy = loader.evaluate_best_model()")
            print("")
            print("# Or use quick functions")
            print("from load_best_model import load_and_evaluate_best_model")
            print("accuracy = load_and_evaluate_best_model()")
            print("```")
        else:
            print("To use the best configuration in your pipeline:")
            print("```python")
            print("from hyperparameter_optimizer import HyperparameterOptimizer")
            print("from main_few_shot_learning import FewShotLearningPipeline")
            print("")
            print("# Load best configuration")
            print("optimizer = HyperparameterOptimizer()")
            print(f'# study = optuna.load_study(study_name="{study.study_name}")')
            print("best_model_config, best_data_config = optimizer.load_best_config(study)")
            print("")
            print("# Run pipeline with best configuration")
            print("pipeline = FewShotLearningPipeline(best_model_config, best_data_config)")
            print("accuracy = pipeline.run_complete_pipeline()")
            print("```")
        
        # Show results location
        results_dir = Path("optimization_results")
        print(f"\n📈 Optimization results saved to: {results_dir.absolute()}")
        print("Check the HTML files for interactive visualizations!")
        
        if save_best_model:
            outputs_dir = Path("outputs")
            print(f"💾 Best model saved to: {outputs_dir.absolute()}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 