"""Hyperparameter optimization using Optuna for few-shot learning."""

from typing import Dict, Any, Tuple, Optional, List
import time
import torch
import optuna
from optuna import Trial
from optuna.study import Study
from pathlib import Path

from .config import ModelConfig, DataConfig
from ..pipeline import Pipeline
import math
import logging
import os
import tempfile
import mlflow
import mlflow.pytorch


class HyperparameterOptimizer:
    """Class for hyperparameter optimization using Optuna with efficiency consideration.

    This optimizer balances model accuracy and computational efficiency by penalizing
    larger/slower models when high accuracy is achieved (>= 95%).
    """

    def __init__(
        self,
        base_model_config: ModelConfig,
        data_config: DataConfig,
        backbone_options: Optional[List[str]] = None,
        efficiency_threshold: float = 80.0,
        param_penalty_scale: float = 10_000_000,
        speed_penalty_scale: float = 100,
        mlflow_tracking_uri: str = "http://localhost:5000"
    ):
        """Initialize the hyperparameter optimizer.

        Args:
            base_model_config: Base model configuration
            data_config: Data configuration
            backbone_options: List of available backbone architectures
            efficiency_threshold: Accuracy threshold above which efficiency is considered
            param_penalty_scale: Scaling factor for parameter count penalty
            speed_penalty_scale: Scaling factor for inference speed penalty
        """
        self.base_model_config = base_model_config
        self.data_config = data_config
        self.backbone_options = backbone_options or ['resnet18']
        self.efficiency_threshold = efficiency_threshold
        self.param_penalty_scale = param_penalty_scale
        self.speed_penalty_scale = speed_penalty_scale
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.study: Optional[Study] = None

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Set up MLflow tracking
        dataset_name = Path(self.data_config.train_root).parent.name  # Get dataset name (Deposit/Root)
        self.mlflow_experiment_name = f"FewShot_{self.base_model_config.n_way}way_{dataset_name}"
        mlflow.set_experiment(self.mlflow_experiment_name)
        

        # Analyze dataset constraints for hyperparameter ranges
        self.dataset_constraints = self._analyze_dataset_constraints()

    def _analyze_dataset_constraints(self) -> Dict[str, Any]:
        """
        Analyze dataset to determine *valid* episodic few-shot ranges.

        Key correctness points vs your version:
        - Computes constraints PER SPLIT (train vs test), not mixing them.
        - Uses the true feasibility rule: n_shot + n_query <= samples_per_class.
        - Provides both a strict bound (min) and a robust bound (10th percentile),
        because a single rare class can make min() overly restrictive.
        - Returns a structure you can use directly in Optuna to sample valid pairs.
        """
        from ..core import DatasetManager

        dataset_manager = DatasetManager(self.data_config, self.base_model_config)

        def _count_per_class(samples):
            counts: Dict[int, int] = {}
            for _, label in samples:
                counts[label] = counts.get(label, 0) + 1
            return counts

        def _safe_min(values):
            return min(values) if values else 0

        def _percentile(values, q: float) -> int:
            """
            q in [0, 1]. Returns an int percentile (floor).
            For small lists, this is stable and avoids numpy dependency.
            """
            if not values:
                return 0
            v = sorted(values)
            idx = int(math.floor(q * (len(v) - 1)))
            return int(v[idx])

        # Count samples per class in each split
        train_counts = _count_per_class(dataset_manager.train_set.samples)
        test_counts  = _count_per_class(dataset_manager.test_set.samples)

        train_values = list(train_counts.values())
        test_values  = list(test_counts.values())

        # Strict bounds (guaranteed valid if you allow all classes)
        train_min = _safe_min(train_values)
        test_min  = _safe_min(test_values)

        # Robust bounds (recommended): avoid one tiny class forcing everything down
        # You can change 0.10 to 0.20 if your class imbalance is extreme.
        train_p10 = _percentile(train_values, 0.10)
        test_p10  = _percentile(test_values, 0.10)

        # Caps to keep search reasonable
        SHOT_CAP = 10
        QUERY_CAP = 10

        def _split_constraints(min_samples: int, p10_samples: int, n_classes: int) -> Dict[str, Any]:
            """
            Derive feasible ranges. True constraint is:
            n_shot + n_query <= samples_per_class
            """
            # Need at least 2 samples per class to have 1-shot + 1-query
            feasible_strict = max(0, min_samples)
            feasible_robust = max(0, p10_samples)

            # Total per-class budget for (shot+query)
            strict_total = feasible_strict
            robust_total = feasible_robust

            # Max shot/query if the other is at least 1
            strict_max_shot = max(1, min(SHOT_CAP, strict_total - 1)) if strict_total >= 2 else 1
            strict_max_query = max(1, min(QUERY_CAP, strict_total - 1)) if strict_total >= 2 else 1

            robust_max_shot = max(1, min(SHOT_CAP, robust_total - 1)) if robust_total >= 2 else 1
            robust_max_query = max(1, min(QUERY_CAP, robust_total - 1)) if robust_total >= 2 else 1

            return {
                "n_classes": n_classes,
                # Strict: safe if you want to allow *all* classes
                "min_samples_per_class": feasible_strict,
                "max_total_shot_query": strict_total,   # n_shot + n_query <= this
                "max_n_shot": strict_max_shot,
                "max_n_query": strict_max_query,
                # Robust: recommended for HPO if there are rare classes
                "p10_samples_per_class": feasible_robust,
                "robust_max_total_shot_query": robust_total,
                "robust_max_n_shot": robust_max_shot,
                "robust_max_n_query": robust_max_query,
            }

        train_constraints = _split_constraints(train_min, train_p10, len(train_counts))
        test_constraints  = _split_constraints(test_min,  test_p10,  len(test_counts))

        # Global n_way upper bounds
        n_way_max = min(len(train_counts), len(test_counts))

        constraints: Dict[str, Any] = {
            "train": train_constraints,
            "test": test_constraints,
            "n_way_max": n_way_max,
        }

        self.logger.info("Dataset constraints analyzed:")
        self.logger.info(f"  Train classes: {len(train_counts)} | Test classes: {len(test_counts)} | n_way_max: {n_way_max}")
        self.logger.info(f"  Train min samples/class: {train_min} | p10: {train_p10}")
        self.logger.info(f"  Test  min samples/class: {test_min} | p10: {test_p10}")
        self.logger.info("  True feasibility rule: n_shot + n_query <= samples_per_class (per split)")

        return constraints

    def calculate_model_efficiency(self, pipeline: Pipeline) -> Tuple[int, float]:
        """Calculate model efficiency metrics.

        Args:
            pipeline: Trained pipeline

        Returns:
            Tuple of (num_parameters, inference_time_per_sample)
        """
        # Count parameters
        n_params = sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad)

        # Measure inference speed (rough estimate)
        pipeline.model.eval()
        device = pipeline.device

        # Create a dummy batch to measure inference time
        dummy_support = torch.randn(4, 3, 224, 224).to(device)  # 4 classes
        dummy_support_labels = torch.randint(0, 4, (4,)).to(device)
        dummy_query = torch.randn(4, 3, 224, 224).to(device)  # 4 query samples

        with torch.no_grad():
            # Warm up
            for _ in range(5):
                _ = pipeline.model(dummy_support, dummy_support_labels, dummy_query)

            # Measure time
            start_time = time.time()
            n_iterations = 50
            for _ in range(n_iterations):
                _ = pipeline.model(dummy_support, dummy_support_labels, dummy_query)
            end_time = time.time()

            avg_inference_time = (end_time - start_time) / n_iterations

        return n_params, avg_inference_time

    def create_model_config_from_trial(self, trial: Trial) -> ModelConfig:
        """Create model configuration from Optuna trial suggestions.

        Args:
            trial: Optuna trial object

        Returns:
            Updated model configuration with suggested hyperparameters
        """
        # Create a copy of base config
        config = ModelConfig(**self.base_model_config.__dict__)

        # Define hyperparameter search space
        config.learning_rate = trial.suggest_float(
            'learning_rate', 1e-6, 1e-2, log=True
        )

        config.weight_decay = trial.suggest_float(
            'weight_decay', 1e-6, 1e-2, log=True
        )

        config.n_training_episodes = trial.suggest_int(
            'n_training_episodes', 50, 100, step=25
        )

        # Backbone selection
        backbone_name = trial.suggest_categorical('backbone', self.backbone_options)
        config.backbone = backbone_name

        # Freeze backbone decision
        config.freeze_backbone = trial.suggest_categorical(
            'freeze_backbone', [True, False]
        )

        # Few-shot learning parameters (dataset-aware ranges)
        # Use test constraints since that's what gets evaluated
        test_const = self.dataset_constraints['test']
        max_n_shot = test_const['robust_max_n_shot']
        max_n_query = test_const['robust_max_n_query']

        # Ensure minimums are reasonable
        min_n_shot = min(1, max_n_shot)
        min_n_query = min(1, max_n_query)

        config.n_shot = trial.suggest_int('n_shot', min_n_shot, max_n_shot)
        config.n_query = trial.suggest_int('n_query', min_n_query, max_n_query)

        # Validate that n_shot + n_query doesn't exceed available samples
        total_needed = config.n_shot + config.n_query
        test_min_samples = self.dataset_constraints['test']['min_samples_per_class']
        if total_needed > test_min_samples:
            # This should not happen with the constrained ranges, but just in case
            self.logger.warning(f"n_shot({config.n_shot}) + n_query({config.n_query}) = {total_needed} > min_samples({test_min_samples})")

        # Adjust learning rate based on backbone freeze status
        if not config.freeze_backbone and config.backbone == 'resnet50':
            config.learning_rate *= 0.1

        return config

    def objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization.

        Returns:
            Composite score balancing accuracy and efficiency (higher is better)
        """
        
        # Start MLflow run for this trial using clean context manager pattern
        try:
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True, parent_run_id=self.parent_run_id): #mlflow.active_run() is not None
                mlflow.set_tag("run_type", "trial")
                mlflow.set_tag("study_name", self.study.study_name if self.study else "unknown")
                mlflow.set_tag("dataset", Path(self.data_config.train_root).parent.name)
                mlflow.set_tag("n_way", str(self.base_model_config.n_way))

                try:
                    # Create model config from trial suggestions
                    model_config = self.create_model_config_from_trial(trial)

                    # Log hyperparameters to MLflow
                    mlflow.log_params({
                        "learning_rate": model_config.learning_rate,
                        "weight_decay": model_config.weight_decay,
                        "n_training_episodes": model_config.n_training_episodes,
                        "backbone": model_config.backbone,
                        "freeze_backbone": model_config.freeze_backbone,
                        "n_shot": model_config.n_shot,
                        "n_query": model_config.n_query,
                        "n_way": model_config.n_way,
                        "trial_number": trial.number
                    })

                    # Create pipeline with suggested hyperparameters
                    pipeline = Pipeline(model_config, self.data_config)

                    # Run training and get final accuracy
                    final_acc = pipeline.run()

                    # Calculate model efficiency metrics
                    n_params, inference_time = self.calculate_model_efficiency(pipeline)

                    # Store metrics for analysis
                    trial.set_user_attr('n_params', n_params)
                    trial.set_user_attr('inference_time', inference_time)
                    trial.set_user_attr('accuracy', final_acc)

                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        "final_accuracy": final_acc,
                        "n_parameters": n_params,
                        "inference_time": inference_time,
                        "training_episodes": model_config.n_training_episodes
                    })

                    # Report intermediate result
                    trial.report(final_acc, step=0)

                    # Handle pruning
                    if trial.should_prune():
                        mlflow.log_metric("pruned", 1)
                        raise optuna.TrialPruned()

                    # Calculate composite score
                    if final_acc >= self.efficiency_threshold:
                        # High accuracy achieved - balance with efficiency
                        param_penalty = n_params / self.param_penalty_scale
                        speed_penalty = inference_time * self.speed_penalty_scale

                        # Composite score: accuracy - penalties for complexity
                        composite_score = final_acc - param_penalty - speed_penalty

                        mlflow.log_metrics({
                            "composite_score": composite_score,
                            "param_penalty": param_penalty,
                            "speed_penalty": speed_penalty
                        })

                        self.logger.info(f"Trial {trial.number}: {final_acc:.1f}% → Score: {composite_score:.2f}")
                    else:
                        # Lower accuracy - focus on improving accuracy first
                        composite_score = final_acc
                        mlflow.log_metric("composite_score", composite_score)
                        self.logger.info(f"Trial {trial.number}: {final_acc:.1f}% (accuracy focus)")

                    return composite_score

                except ValueError as e:
                    # Handle cases where hyperparameter combination is invalid for the dataset
                    # (e.g., not enough samples for n_shot + n_query)
                    if "samples" in str(e).lower() and ("shot" in str(e).lower() or "query" in str(e).lower()):
                        self.logger.warning(f"Trial {trial.number}: Skipped (insufficient samples)")
                        # Log to MLflow within the run context
                        mlflow.log_param("status", "skipped_insufficient_samples")
                        mlflow.log_metric("error", 1)
                        # Return a very low score to discourage this combination
                        return -1000.0
                    else:
                        # Log error and re-raise
                        mlflow.log_param("status", "error")
                        mlflow.log_param("error_message", str(e))
                        mlflow.log_metric("error", 1)
                        raise e
                except Exception as e:
                    # Handle other unexpected errors
                    self.logger.warning(f"Trial {trial.number}: Skipped (error)")
                    # Log to MLflow within the run context
                    mlflow.log_param("status", "error")
                    mlflow.log_param("error_message", str(e))
                    mlflow.log_metric("error", 1)
                    # Return a very low score for other errors too
                    return -1000.0
        except Exception as ex:
            # This top-level try catch ensures that if anything happens in MLflow context or unexpected outer error, we catch and log to self.logger.
            self.logger.error(f"Top-level error during trial {trial.number}: {ex}")
            # Optionally: You can also log to MLflow here if it is active
            try:
                if mlflow.active_run():
                    mlflow.log_param("status", "outer_error")
                    mlflow.log_param("error_message", str(ex))
                    mlflow.log_metric("error", 1)
            except Exception:
                pass
            return -1000.0

    def optimize(
        self,
        n_trials: int = 30,
        study_name: str = "few_shot_optimization",
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
            timeout: Timeout in seconds

        Returns:
            Dictionary with best hyperparameters and results
        """
        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)

        test_const = self.dataset_constraints['test']
        self.logger.info(f"Starting hyperparameter optimization ({n_trials} trials)")
        self.logger.info(f"Dataset constraints: n_shot=1-{test_const['robust_max_n_shot']}, n_query=1-{test_const['robust_max_n_query']}")
        self.logger.info(f"Based on test set min {test_const['min_samples_per_class']} samples per class")

        # Wrap entire optimization in parent MLflow run
        with mlflow.start_run(run_name=f"{study_name}_hpo_session") as parent_run:
            self.parent_run_id = parent_run.info.run_id
            mlflow.set_tag("run_type", "hpo_session")
            mlflow.set_tag("study_name", study_name)
            mlflow.set_tag("dataset", Path(self.data_config.train_root).parent.name)
            mlflow.set_tag("n_way", str(self.base_model_config.n_way))
            
            # Create Optuna study
            self.study = optuna.create_study(
                study_name=study_name,
                direction="maximize",  # We want to maximize composite score
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            # Run optimization (IMPORTANT: use n_jobs=1 for SQLite backend)
            # Run optimization (IMPORTANT: use n_jobs=1 for SQLite backend)
            self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout, n_jobs=4)
            # Log clean results summary
            self.logger.info("="*60)
            self.logger.info("HYPERPARAMETER OPTIMIZATION COMPLETE")
            self.logger.info("="*60)

            best_trial = self.study.best_trial
            best_accuracy = best_trial.user_attrs.get('accuracy', 0)

            self.logger.info(f"Best Final Accuracy: {best_accuracy:.2f}%")
            self.logger.info(f"Best Composite Score: {self.study.best_value:.2f}")
            self.logger.info(f"Best Trial Number: {best_trial.number}")
            self.logger.info(f"Total Trials Completed: {len(self.study.trials)}")

            # Show efficiency metrics for best trial
            if 'n_params' in best_trial.user_attrs:
                n_params = best_trial.user_attrs['n_params']
                inference_time = best_trial.user_attrs['inference_time']
                self.logger.info(f"Model Parameters: {n_params:,}")
                self.logger.info(f"Inference Time: {inference_time:.4f}s")

            self.logger.info("Best Hyperparameters:")
            self.logger.info(f"   Learning Rate: {self.study.best_params['learning_rate']:.2e}")
            self.logger.info(f"   Weight Decay: {self.study.best_params['weight_decay']:.2e}")
            self.logger.info(f"   Training Episodes: {self.study.best_params['n_training_episodes']}")
            self.logger.info(f"   Backbone: {self.study.best_params['backbone']}")
            self.logger.info(f"   Freeze Backbone: {self.study.best_params['freeze_backbone']}")
            self.logger.info(f"   N-Shot: {self.study.best_params['n_shot']}")
            self.logger.info(f"   N-Query: {self.study.best_params['n_query']}")

            self.logger.info("="*60)

            # Log best summary on parent run
            mlflow.log_metric("best_value", float(self.study.best_value))
            best_accuracy = float(best_trial.user_attrs.get('accuracy', 0))
            mlflow.log_metric("best_accuracy", best_accuracy)
            mlflow.log_param("best_trial_number", best_trial.number)

            # Log best model run (nested under parent)
            self._log_best_model_to_mlflow(best_trial, study_name=study_name, parent_run_id=parent_run.info.run_id)

            # Extract best accuracy from trial user attributes
            best_accuracy = best_trial.user_attrs.get('accuracy', 0)

        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'best_accuracy': best_accuracy,
            'study': self.study,
            'best_trial': best_trial
        }

    def _log_best_model_to_mlflow(self, best_trial, study_name: str = None, parent_run_id: str = None):
        """Log the best model to MLflow with a dedicated run."""
        try:
            # Create a new run specifically for the best model
            with mlflow.start_run(run_name="best_model", nested=parent_run_id is not None):

                # Get best model config
                best_config = self.create_model_config_from_trial(best_trial)

                # Log best hyperparameters
                mlflow.log_params({
                    "learning_rate": best_config.learning_rate,
                    "weight_decay": best_config.weight_decay,
                    "n_training_episodes": best_config.n_training_episodes,
                    "backbone": best_config.backbone,
                    "freeze_backbone": best_config.freeze_backbone,
                    "n_shot": best_config.n_shot,
                    "n_query": best_config.n_query,
                    "n_way": best_config.n_way,
                    "trial_number": best_trial.number,
                    "is_best_model": True
                })

                # Log best metrics
                best_accuracy = best_trial.user_attrs.get('accuracy', 0)
                n_params = best_trial.user_attrs.get('n_params', 0)
                inference_time = best_trial.user_attrs.get('inference_time', 0)

                mlflow.log_metrics({
                    "final_accuracy": best_accuracy,
                    "n_parameters": n_params,
                    "inference_time": inference_time,
                    "training_episodes": best_config.n_training_episodes,
                    "composite_score": best_trial.value
                })

                # Train and log the best model
                self.logger.info("Training best model for MLflow logging...")

                # Create pipeline with best config
                pipeline = Pipeline(best_config, self.data_config)

                # Train the model
                best_model_accuracy = pipeline.run()

                # Log the trained model to MLflow
                dataset_name = Path(self.data_config.train_root).parent.name
                registered_model_name = f"fewshot_{dataset_name}_be#t"

                ## register model
                # mlflow.pytorch.log_model(
                #     pipeline.model,
                #     "model",
                #     registered_model_name=registered_model_name
                # )

                # Log additional metrics and artifacts
                mlflow.log_metric("best_model_accuracy", best_model_accuracy)
                mlflow.log_param("model_architecture", best_config.backbone)
                mlflow.log_param("dataset", dataset_name)

                self.logger.info("Best model logged to MLflow successfully!")

        except Exception as e:
            self.logger.error(f"Failed to log best model to MLflow: {e}")
            # Don't raise - optimization was successful, logging failed

    def apply_best_params_to_config(self, best_params: Dict[str, Any]) -> ModelConfig:
        """Apply best hyperparameters to model configuration.

        Args:
            best_params: Best parameters from optimization

        Returns:
            Updated model configuration
        """
        config = ModelConfig(**self.base_model_config.__dict__)

        config.learning_rate = best_params['learning_rate']
        config.weight_decay = best_params['weight_decay']
        config.n_training_episodes = best_params['n_training_episodes']
        config.backbone = best_params['backbone']
        config.freeze_backbone = best_params['freeze_backbone']
        config.n_shot = best_params['n_shot']
        config.n_query = best_params['n_query']

        # Adjust learning rate for unfrozen larger models
        if not config.freeze_backbone and config.backbone == 'resnet50':
            config.learning_rate *= 0.1

        return config

    def get_study_results(self) -> Optional[Dict[str, Any]]:
        """Get current study results if optimization has been run.

        Returns:
            Study results or None if no study exists
        """
        if self.study is None:
            return None

        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }


# Backward compatibility functions
def calculate_model_efficiency(pipeline: Pipeline) -> Tuple[int, float]:
    """Backward compatibility function."""
    optimizer = HyperparameterOptimizer(None, None)  # type: ignore
    return optimizer.calculate_model_efficiency(pipeline)


def create_model_config_from_trial(
    trial: Trial,
    base_config: ModelConfig,
    backbone_options: Optional[list] = None
) -> ModelConfig:
    """Backward compatibility function."""
    optimizer = HyperparameterOptimizer(base_config, None, backbone_options)  # type: ignore
    return optimizer.create_model_config_from_trial(trial)


def objective_function(
    trial: Trial,
    base_model_config: ModelConfig,
    data_config: DataConfig,
    backbone_options: Optional[list] = None
) -> float:
    """Backward compatibility function."""
    optimizer = HyperparameterOptimizer(base_model_config, data_config, backbone_options)
    return optimizer.objective(trial)


def apply_best_params_to_config(
    base_config: ModelConfig,
    best_params: Dict[str, Any]
) -> ModelConfig:
    """Backward compatibility function."""
    optimizer = HyperparameterOptimizer(base_config, None)  # type: ignore
    return optimizer.apply_best_params_to_config(best_params)
