"""Training script with configurable dataset selection and hyperparameter optimization."""

import argparse
import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Set up paths and imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src import Pipeline
from src.core import HyperparameterOptimizer
from src.core.config import DataConfig, ModelConfig

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train few-shot learning models with configurable datasets and hyperparameter optimization"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="roots",
        choices=["roots", "deposit", "crack_fracture", "joint_disp"],
        help="Configuration module to use (default: roots)"
    )

    parser.add_argument(
        "--custom-config",
        type=str,
        help="Path to custom config module (e.g., 'configs.my_custom_config'). "
             "If specified, overrides --config option."
    )

    parser.add_argument(
        "--optimize-hyperparams",
        action="store_true",
        help="Run hyperparameter optimization before training"
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of optimization trials (only used with --optimize-hyperparams)"
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study (default: auto-generated)"
    )

    parser.add_argument(
        "--show-dataset-info",
        action="store_true",
        help="Show detailed dataset information and exit"
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI (default: local './mlruns')"
    )

    parser.add_argument(
        "--load-best-from-mlflow",
        action="store_true",
        help="Load best model configuration from MLflow 'best_model' run and train with 500 episodes"
    )

    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the final trained model in MLflow Model Registry"
    )

    parser.add_argument(
        "--load-best-from-mlflow-run-id",
        type=str,
        help="Load best model configuration from MLflow run ID"
    )
    parser.add_argument(
        "--best-from-mlflow-run-id",
        type=str,
        default="eb2ded5a8a8c4f38bdcc0374fab884ef",
        help="Best model run ID from MLflow"
    )

    return parser.parse_args()


def make_study_name(args) -> str:
    """Create a unique study name with timestamp."""
    if args.study_name:
        return args.study_name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.config.upper()}_{ts}"


def load_config_module(config_name: str) -> Any:
    """Dynamically load config module."""
    try:
        if "." in config_name:
            # Custom config path like "configs.my_custom_config"
            module = importlib.import_module(config_name)
        else:
            # Built-in config - import from configs package
            module = importlib.import_module(f"configs.{config_name}")
        return module
    except ImportError as e:
        print(f"Error loading config module '{config_name}': {e}")
        print("Available configs: roots, deposit")
        sys.exit(1)


def optimize_hyperparameters(
    model_config: ModelConfig,
    data_config: DataConfig,
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run hyperparameter optimization."""
    logger.info("Running Hyperparameter Optimization")
    logger.info(f"Running {args.n_trials} trials...")

    optimizer = HyperparameterOptimizer(
        base_model_config=model_config,
        data_config=data_config,
        backbone_options=['resnet18'],  # Can be expanded
        efficiency_threshold=80.0,
        mlflow_tracking_uri=args.mlflow_tracking_uri
    )

    # Show dataset constraints
    constraints = optimizer.dataset_constraints
    train_const = constraints['train']
    test_const = constraints['test']
    logger.info(f"Dataset: {train_const['n_classes']} train/{test_const['n_classes']} test classes")
    logger.info(f"Constraints: n_shot=5-{test_const['robust_max_n_shot']}, n_query=5-{test_const['robust_max_n_query']} (test min {test_const['min_samples_per_class']} samples/class)")

    study_name = make_study_name(args)
    results = optimizer.optimize(n_trials=args.n_trials, study_name=study_name)

    logger.info("Optimization completed!")
    logger.info("Best Configuration Found:")
    best_accuracy = results.get('best_accuracy', 0)
    logger.info(f"   Final Accuracy: {best_accuracy:.2f}%")
    logger.info(f"   Learning Rate: {results['best_params']['learning_rate']:.2e}")
    logger.info(f"   Weight Decay: {results['best_params']['weight_decay']:.2e}")
    logger.info(f"   N-Shot: {results['best_params']['n_shot']}, N-Query: {results['best_params']['n_query']}")
    logger.info(f"   Backbone: {results['best_params']['backbone']} (freeze: {results['best_params']['freeze_backbone']})")

    return results

def load_params_from_run_id(run_id: str, tracking_uri: str) -> Dict[str, Any]:
    """
    Load hyperparameters from a specific MLflow run.

    Args:
        run_id: MLflow run ID to load parameters from
        tracking_uri: MLflow tracking URI

    Returns:
        Dictionary containing the hyperparameters

    Raises:
        ValueError: If run not found or parameters are missing
    """
    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params

        # Convert and validate parameters
        best_params = {
            "learning_rate": float(params["learning_rate"]),
            "weight_decay": float(params["weight_decay"]),
            "backbone": params["backbone"],
            "n_training_episodes": 500,
            "freeze_backbone": str(params["freeze_backbone"]).lower() == "true",
            "n_shot": int(params["n_shot"]),
            "n_query": int(params["n_query"]),
            "n_way": int(params.get("n_way", 4)),
        }

        return best_params

    except KeyError as e:
        raise ValueError(f"Missing required parameter in run {run_id}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load parameters from run {run_id}: {e}")

def load_best_config_from_mlflow(
    model_config: ModelConfig,
    data_config: DataConfig,
    tracking_uri: str,
    logger: logging.Logger
) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """
    Load the best model configuration from MLflow's latest 'best_model' run.

    Args:
        model_config: Base model configuration
        data_config: Data configuration
        tracking_uri: MLflow tracking URI
        logger: Logger instance

    Returns:
        Tuple of (best_params, best_run_id, parent_run_id)

    Raises:
        ValueError: If experiment or best_model run not found
    """
    mlflow.set_tracking_uri(tracking_uri)

    dataset_name = Path(data_config.train_root).parent.name
    experiment_name = f"FewShot_{model_config.n_way}way_{dataset_name}"

    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")

    # Find the latest best_model run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'best_model'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError("No 'best_model' run found in MLflow")

    best_run_id = runs.iloc[0]["run_id"]
    parent_run_id = runs.iloc[0].get("tags.mlflow.parentRunId")

    logger.info(f"Found best model run: {best_run_id}")
    logger.info(f"Parent HPO session run: {parent_run_id}")

    # Extract parameters from the best run
    try:
        client = MlflowClient()
        run = client.get_run(best_run_id)
        params = run.data.params

        best_params = {
            "learning_rate": float(params["learning_rate"]),
            "weight_decay": float(params["weight_decay"]),
            "n_training_episodes": 500,  # Override for final training
            "backbone": params["backbone"],
            "freeze_backbone": str(params["freeze_backbone"]).lower() == "true",
            "n_shot": int(params["n_shot"]),
            "n_query": int(params["n_query"]),
        }

        return best_params, best_run_id, parent_run_id

    except KeyError as e:
        raise ValueError(f"Missing required parameter in best model run: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load best model configuration: {e}")


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def setup_training_config(args: argparse.Namespace) -> Tuple[ModelConfig, DataConfig]:
    """
    Set up the training configuration based on command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (model_config, data_config)
    """
    # Determine config module to use
    config_name = args.custom_config or args.config
    config_module = load_config_module(config_name)

    # Get base configuration
    model_config, data_config = config_module.get_config()

    return model_config, data_config


def load_best_parameters(
    args: argparse.Namespace,
    model_config: ModelConfig,
    data_config: DataConfig,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Load the best parameters from MLflow based on arguments.

    Args:
        args: Command line arguments
        model_config: Base model configuration
        data_config: Data configuration
        logger: Logger instance

    Returns:
        Best parameters dictionary
    """
    if args.load_best_from_mlflow_run_id:
        # Load from specific run ID
        run_id = args.best_from_mlflow_run_id  # TODO: Make this configurable
        return load_params_from_run_id(run_id, args.mlflow_tracking_uri)
    else:
        # Load latest best model from experiment
        best_params, _, _ = load_best_config_from_mlflow(
            model_config, data_config, args.mlflow_tracking_uri, logger
        )
        return best_params


def run_hyperparameter_optimization(
    model_config: ModelConfig,
    data_config: DataConfig,
    args: argparse.Namespace,
    logger: logging.Logger
) -> ModelConfig:
    """
    Run hyperparameter optimization and return updated model config.

    Args:
        model_config: Base model configuration
        data_config: Data configuration
        args: Command line arguments
        logger: Logger instance

    Returns:
        Updated model configuration with best parameters
    """
    logger.info("Running hyperparameter optimization...")
    results = optimize_hyperparameters(model_config, data_config, args, logger)

    # Apply best parameters to config
    optimizer = HyperparameterOptimizer(model_config, data_config)
    results['best_params']['n_training_episodes'] = 100
    updated_config = optimizer.apply_best_params_to_config(results['best_params'])

    logger.info("Training with optimized hyperparameters")
    return updated_config


def run_final_training(
    pipeline: Pipeline,
    model_config: ModelConfig,
    data_config: DataConfig,
    args: argparse.Namespace,
    logger: logging.Logger
) -> float:
    """
    Run final training with best parameters and log to MLflow.

    Args:
        pipeline: Training pipeline
        model_config: Model configuration
        data_config: Data configuration
        args: Command line arguments
        logger: Logger instance

    Returns:
        Final accuracy achieved
    """
    dataset_name = Path(data_config.train_root).parent.name
    model_name = f"fewshot_{dataset_name}_classifier"

    from datetime import datetime
    run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"final_train_500ep_{run_time}"
    with mlflow.start_run(run_name=run_name):
        logger.info("Starting final 500-episode training...")

        # Set run metadata
        mlflow.set_tag("run_type", "final_train")
        mlflow.set_tag("dataset", dataset_name)

        # Log hyperparameters
        mlflow.log_params({
            "learning_rate": model_config.learning_rate,
            "weight_decay": model_config.weight_decay,
            "n_training_episodes": model_config.n_training_episodes,
            "backbone": model_config.backbone,
            "freeze_backbone": model_config.freeze_backbone,
            "n_shot": model_config.n_shot,
            "n_query": model_config.n_query,
            "n_way": model_config.n_way,
            "training_type": "final_500_episodes",
            "loaded_from_mlflow": True
        })

        # Train the model
        final_accuracy = pipeline.run()

        # Log final metrics
        mlflow.log_metrics({
            "final_accuracy": final_accuracy,
            "training_episodes": model_config.n_training_episodes
        })

        # Save model locally
        save_model_locally(pipeline, data_config, logger)

        # Log model to MLflow artifacts
        mlflow.pytorch.log_model(
            pipeline.model,
            artifact_path="final_model_500ep"
        )

        if args.register_model:
            logger.info(f"Model registered as: {model_name}")
        else:
            logger.info("Model logged to MLflow artifacts")

        logger.info(f"Final training completed. Accuracy: {final_accuracy:.2f}%")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

        return final_accuracy


def run_regular_training(
    pipeline: Pipeline,
    data_config: DataConfig,
    logger: logging.Logger
) -> float:
    """
    Run regular training without MLflow logging.

    Args:
        pipeline: Training pipeline
        data_config: Data configuration
        logger: Logger instance

    Returns:
        Final accuracy achieved
    """
    logger.info("Starting regular training...")
    final_accuracy = pipeline.run()

    save_model_locally(pipeline, data_config, logger)

    logger.info(f"Training completed. Accuracy: {final_accuracy:.2f}%")
    return final_accuracy


def save_model_locally(pipeline: Pipeline, data_config: DataConfig, logger: logging.Logger) -> str:
    """
    Save the trained model to local filesystem.

    Args:
        pipeline: Training pipeline with trained model
        data_config: Data configuration
        logger: Logger instance

    Returns:
        Path where model was saved
    """
    output_path = os.path.join(data_config.output_root, "best_model")
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "model.pth")

    pipeline.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    return model_path


def main() -> None:
    """
    Main entry point for the training script.

    Orchestrates the training workflow based on command line arguments:
    - Hyperparameter optimization
    - Final training with best parameters from MLflow
    - Regular training with default parameters
    """
    logger = setup_logging()
    args = parse_arguments()

    # Override with specific configuration (TODO: Make this configurable)
    args.config = "joint_disp" # deposit, roots, crack_fracture, joint_disp
    args.study_name = ""
    args.n_trials = 50
    args.mlflow_tracking_uri = "http://127.0.0.1:5000"
    args.load_best_from_mlflow = True
    args.optimize_hyperparams = True
    args.register_model = False
    args.load_best_from_mlflow_run_id = True
    args.best_from_mlflow_run_id = "0339728f33b24a769a44c5286a7d6c21"

    try:
        # Set up configuration
        model_config, data_config = setup_training_config(args)
        logger.info(f"Using configuration: {args.custom_config or args.config}")

        # Show dataset info if requested
        if args.show_dataset_info:
            temp_pipeline = Pipeline(model_config, data_config)
            temp_pipeline.print_dataset_info()
            return

        # Determine training mode and configure model
        if args.load_best_from_mlflow:
            # Load best parameters from MLflow
            best_params = load_best_parameters(args, model_config, data_config, logger)

            # Apply best parameters to config
            optimizer = HyperparameterOptimizer(model_config, data_config)
            model_config = optimizer.apply_best_params_to_config(best_params)
            logger.info("Training with best parameters from MLflow")

        elif args.optimize_hyperparams:
            # Run hyperparameter optimization
            model_config = run_hyperparameter_optimization(
                model_config, data_config, args, logger
            )
        else:
            logger.info("Training with default configuration")

        # Create and run training pipeline
        pipeline = Pipeline(model_config, data_config)

        if args.load_best_from_mlflow:
            run_final_training(pipeline, model_config, data_config, args, logger)
        else:
            run_regular_training(pipeline, data_config, logger)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

