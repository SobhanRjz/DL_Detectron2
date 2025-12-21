# Few-Shot Learning Library

Clean, extensible architecture for few-shot classification using Prototypical Networks.

## Structure

```
Root/
├── fsl/                    # Core library
│   ├── core/              # Core components
│   │   ├── config.py      # Configuration classes
│   │   ├── model.py       # Prototypical Network
│   │   ├── dataset.py     # Data management
│   │   ├── trainer.py     # Training logic
│   │   └── evaluator.py   # Evaluation logic
│   ├── pipeline.py        # End-to-end pipeline
│   └── utils.py           # Utilities
├── configs/               # Model-specific configs
│   ├── roots.py          # Root classification
│   └── cracks.py         # Crack classification
└── scripts/              # Entry points
    ├── train.py          # Training script
    ├── evaluate.py       # Evaluation script
    └── inference.py      # Inference script
```

## Usage

### Training a Model

```python
from fsl import Pipeline
from configs import roots

model_config, data_config = roots.get_config()
pipeline = Pipeline(model_config, data_config)

accuracy = pipeline.run()
pipeline.save_model("outputs/model.pth")
```

Or via command line:
```bash
python scripts/train.py --config roots
```

### Hyperparameter Optimization with MLflow Tracking

The library includes built-in hyperparameter optimization with MLflow experiment tracking:

```bash
# Install MLflow
pip install mlflow mlflow[pytorch]

# Run hyperparameter optimization with MLflow tracking
python scripts/train.py --config deposit --optimize-hyperparams --n-trials 50

# Specify MLflow tracking server (optional)
python scripts/train.py --config roots --optimize-hyperparams \
    --mlflow-tracking-uri "http://localhost:5000"
```

## Run mlflow server
mlflow server --host 127.0.0.1 --port 5000 `
  --backend-store-uri "sqlite:///mlflow/mlflow.db" `
  --default-artifact-root "file:///mlflow/mlartifacts"


### MLflow Features

- **Automatic experiment tracking**: Each optimization trial is logged as an MLflow run
- **Parameter and metric logging**: All hyperparameters, accuracy, model size, and inference time
- **Best model logging**: Only the best-performing model is saved to MLflow model registry
- **Run comparison**: Compare all trials in the MLflow UI
- **Experiment organization**: Automatic experiment naming based on dataset and n-way

### Viewing Results

Start MLflow UI to view experiments:
```bash
mlflow ui
```

Navigate to `http://localhost:5000` to see:
- All optimization trials with their parameters and metrics
- Best model performance comparison
- Model artifacts and metadata
- Interactive plots and comparisons

### Adding a New Model

Create a new config file in `configs/`:

```python
# configs/my_model.py
import os
from fsl.core import ModelConfig, DataConfig

def get_config() -> tuple[ModelConfig, DataConfig]:
    model_config = ModelConfig(
        n_way=5,  # Number of classes
        n_shot=3,
        # ... other parameters
    )
    
    data_config = DataConfig(
        train_root="path/to/train",
        test_root="path/to/test",
    )
    
    return model_config, data_config
```

Then use it:
```python
from configs import my_model

model_config, data_config = my_model.get_config()
pipeline = Pipeline(model_config, data_config)
```

## Design Principles

1. **Separation of Concerns**: Core library, configs, and scripts are separate
2. **Reusability**: Import `fsl` package anywhere
3. **Extensibility**: Add new models via config files
4. **Clarity**: Clear naming (dataset.py, model.py, pipeline.py)

