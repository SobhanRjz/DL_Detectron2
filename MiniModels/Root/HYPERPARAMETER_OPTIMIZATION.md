# Hyperparameter Optimization for Few-Shot Learning

This directory contains comprehensive hyperparameter optimization tools for your few-shot learning pipeline. Choose the approach that best fits your needs.

## 🚀 Quick Start

### Option 1: Optuna (Recommended)

**Best for:** Most users, especially those wanting production-ready optimization with great visualizations.

```bash
# Install dependencies
pip install -r requirements_hyperopt.txt

# Run quick test (3 trials)
python run_optimization.py --quick-test

# Run full optimization (20 trials)
python run_optimization.py --trials 20

# Run with custom parameters
python run_optimization.py --trials 50 --study-name "my_optimization"
```

### Option 2: Weights & Biases

**Best for:** Teams wanting cloud-based experiment tracking and collaboration.

```bash
# Install W&B
pip install wandb

# Login to W&B (one-time setup)
wandb login

# Run sweep
python wandb_optimizer.py --count 20 --project "my-few-shot-project"
```

## 📊 Available Hyperparameters

The optimization searches over these parameters:

| Parameter | Type | Range/Options | Description |
|-----------|------|---------------|-------------|
| `learning_rate` | float | 1e-5 to 1e-2 (log scale) | Optimizer learning rate |
| `weight_decay` | float | 1e-6 to 1e-2 (log scale) | L2 regularization strength |
| `n_shot` | int | 1, 2, 3, 5, 8, 10 | Number of support examples per class |
| `n_query` | int | 1, 2, 3, 5 | Number of query examples per class |
| `n_training_episodes` | int | 100 to 1000 | Number of training episodes |
| `validation_frequency` | int | 20 to 100 | Episodes between validations |
| `backbone_name` | categorical | resnet18, resnet34, resnet50 | CNN backbone architecture |
| `freeze_backbone` | boolean | True, False | Whether to freeze backbone weights |
| `batch_size` | categorical | 64, 128, 256 | Training batch size |

## 🔧 Installation

### Core Requirements
```bash
# Install core dependencies first
pip install torch torchvision matplotlib pandas numpy scikit-learn

# For Optuna optimization
pip install -r requirements_hyperopt.txt

# For W&B optimization  
pip install wandb pyyaml
```

## 📖 Detailed Usage

### Optuna Optimization

#### Basic Usage
```python
from hyperparameter_optimizer import run_hyperparameter_optimization

# Run optimization
study = run_hyperparameter_optimization(
    n_trials=50,
    study_name="few_shot_optimization"
)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

#### Advanced Usage
```python
from hyperparameter_optimizer import HyperparameterOptimizer
import optuna

# Create optimizer with custom settings
optimizer = HyperparameterOptimizer(
    n_trials=100,
    timeout=7200,  # 2 hours
    study_name="production_optimization"
)

# Run optimization
study = optimizer.optimize()

# Load best configuration
best_model_config, best_data_config = optimizer.load_best_config(study)

# Use best configuration
from main_few_shot_learning import FewShotLearningPipeline
pipeline = FewShotLearningPipeline(best_model_config, best_data_config)
final_accuracy = pipeline.run_complete_pipeline()
```

#### Resume Previous Study
```python
import optuna

# Load previous study
study = optuna.load_study(study_name="my_study_name")

# Continue optimization
study.optimize(objective_function, n_trials=20)
```

### Weights & Biases Optimization

#### Basic Usage
```python
from wandb_optimizer import run_wandb_optimization

# Run W&B sweep
sweep_id = run_wandb_optimization(
    project_name="few-shot-learning",
    count=30
)
```

#### Manual Sweep Configuration
```bash
# Generate sweep configuration
python wandb_optimizer.py --count 0  # Just generate config

# Run manual sweep
wandb sweep wandb_sweep_config.yaml
wandb agent <sweep_id>
```

#### View Results
- Visit your W&B dashboard: https://wandb.ai/your-username/your-project
- Interactive plots and comparisons available
- Easy sharing with team members

## 📈 Understanding Results

### Optuna Results

After optimization, check the `optimization_results/` directory:

- `*_summary.json`: Best parameters and overall results
- `*_trials.csv`: Detailed trial data
- `*_history.html`: Optimization progress over time
- `*_importances.html`: Parameter importance analysis
- `*_parallel.html`: Parallel coordinate plot
- `*_slice.html`: Parameter slice analysis

### Key Visualizations

1. **Optimization History**: Shows how the best value improves over trials
2. **Parameter Importances**: Which hyperparameters matter most
3. **Parallel Coordinates**: Relationships between parameters and performance
4. **Slice Plots**: How individual parameters affect performance

## ⚡ Performance Tips

### Speed Optimization
1. **Start Small**: Begin with 10-20 trials to get initial insights
2. **Use GPU**: Ensure CUDA is available for faster training
3. **Reduce Episodes**: Lower `n_training_episodes` for initial exploration
4. **Early Stopping**: Optuna automatically prunes poor trials

### Search Strategy
1. **Coarse-to-Fine**: Start with wide ranges, then narrow down
2. **Focus on Important Parameters**: Use importance plots to guide next searches
3. **Domain Knowledge**: Adjust ranges based on your specific dataset

### Resource Management
```python
# For limited computational resources
optimizer = HyperparameterOptimizer(
    n_trials=20,
    timeout=3600  # 1 hour limit
)

# For distributed optimization
# Use Ray Tune (see requirements_hyperopt.txt)
```

## 🎯 Best Practices

### 1. Systematic Approach
```python
# Phase 1: Quick exploration (10-20 trials)
study_1 = run_hyperparameter_optimization(n_trials=15, study_name="exploration")

# Phase 2: Focused search based on results
# Adjust parameter ranges based on Phase 1 results
study_2 = run_hyperparameter_optimization(n_trials=30, study_name="focused")

# Phase 3: Fine-tuning around best parameters
study_3 = run_hyperparameter_optimization(n_trials=20, study_name="fine_tune")
```

### 2. Validation Strategy
- Use separate validation set for hyperparameter tuning
- Final test set only for final evaluation
- Cross-validation for small datasets

### 3. Experiment Tracking
```python
# Always name your studies
study_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M')}_lr_search"

# Document your experiments
experiment_notes = {
    "objective": "Find optimal learning rate range",
    "dataset": "Root classification",
    "changes": "Added weight decay to search space"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size range: `[32, 64, 128]`
   - Lower number of training episodes
   - Use `torch.cuda.empty_cache()` between trials

2. **Slow Convergence**
   - Increase number of startup trials: `n_startup_trials=20`
   - Use different sampler: `RandomSampler()` for exploration

3. **No Improvement**
   - Check parameter ranges are reasonable
   - Verify objective function is working correctly
   - Try different search algorithms

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test single trial
from hyperparameter_optimizer import HyperparameterOptimizer
optimizer = HyperparameterOptimizer(n_trials=1)
study = optimizer.optimize()
```

## 📚 Advanced Topics

### Custom Objective Functions
```python
def custom_objective(trial):
    # Custom parameter suggestions
    backbone = trial.suggest_categorical("backbone", ["efficientnet", "mobilenet"])
    
    # Multi-objective optimization
    accuracy = train_and_evaluate(trial.params)
    model_size = get_model_size(trial.params)
    
    # Combine objectives
    score = accuracy - 0.1 * model_size  # Penalty for large models
    return score
```

### Database Storage
```python
# For persistent studies across runs
optimizer = HyperparameterOptimizer(
    storage_url="sqlite:///my_study.db",
    study_name="persistent_study"
)
```

### Multi-GPU Optimization
```python
# Distribute trials across GPUs (requires setup)
# See Ray Tune documentation for distributed optimization
```

## 🤝 Contributing

To add new hyperparameters:

1. Update `suggest_hyperparameters()` in `hyperparameter_optimizer.py`
2. Add corresponding parameter application in `objective()`
3. Update this documentation

## 📄 License

This optimization framework is part of your few-shot learning project and follows the same license terms. 