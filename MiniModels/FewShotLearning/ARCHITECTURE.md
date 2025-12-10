# Architecture Overview

## Directory Structure

```
Root/
├── fsl/                          # Core Library (reusable)
│   ├── core/                     # Core components
│   │   ├── config.py            # ModelConfig, DataConfig
│   │   ├── model.py             # PrototypicalNetwork, ModelFactory
│   │   ├── dataset.py           # DatasetManager
│   │   ├── trainer.py           # Trainer
│   │   └── evaluator.py         # Evaluator
│   ├── pipeline.py              # Pipeline orchestration
│   └── utils.py                 # Utilities (seed, device)
│
├── configs/                      # Model-specific configurations
│   ├── roots.py                 # Root Mass/Tap/Fine (3-way)
│   └── cracks.py                # Crack/Fracture (2-way)
│
├── scripts/                      # Entry point scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference script
│
├── Data/                         # Dataset (your data)
│   ├── Train/
│   └── Test/
│
├── outputs/                      # Model checkpoints
│
└── README.md                     # Usage documentation
```

## Design Benefits

### 1. Clean Separation
- **Core library** (`fsl/`): Reusable, framework-agnostic
- **Configs** (`configs/`): Model-specific parameters
- **Scripts** (`scripts/`): Entry points, can be replaced

### 2. Multi-Model Support
To train different models, just create a new config:

```python
# configs/new_task.py
from fsl.core import ModelConfig, DataConfig

def get_config():
    model_config = ModelConfig(n_way=4, ...)
    data_config = DataConfig(train_root="...", test_root="...")
    return model_config, data_config
```

### 3. Clear Naming
- `model.py` instead of `models.py`
- `dataset.py` instead of `data_manager.py`
- `pipeline.py` instead of `main_few_shot_learning.py`

### 4. Minimal Surface Area
Each module exposes only what's necessary:
- `config.py`: 2 classes (ModelConfig, DataConfig)
- `model.py`: 2 classes (PrototypicalNetwork, ModelFactory)
- `trainer.py`: 1 class (Trainer)
- `evaluator.py`: 1 class (Evaluator)
- `dataset.py`: 1 class (DatasetManager)

## Removed Files

Deleted unnecessary files:
- ❌ `speed_test.py` - not core functionality
- ❌ `debug_accuracy_differences.py` - debug script
- ❌ `hyperparameter_optimizer.py` - optimization tool
- ❌ `wandb_optimizer.py` - optimization tool
- ❌ `run_optimization.py` - optimization runner
- ❌ `load_best_model.py` - now in pipeline

## Usage Patterns

### Pattern 1: Direct Library Use
```python
from fsl import Pipeline
from configs import roots

model_config, data_config = roots.get_config()
pipeline = Pipeline(model_config, data_config)
pipeline.run()
```

### Pattern 2: Custom Training
```python
from fsl.core import ModelFactory, DatasetManager, Trainer, Evaluator
from configs import roots

model_config, data_config = roots.get_config()
device = torch.device("cuda")

# Build components
model = ModelFactory.create_model(model_config)
data_manager = DatasetManager(data_config, model_config)
trainer = Trainer(model, model_config, device)

# Custom training
train_loader = data_manager.get_train_loader()
trainer.train(train_loader)
```

### Pattern 3: Command Line
```bash
# Train
python scripts/train.py

# Evaluate
python scripts/evaluate.py

# Inference
python scripts/inference.py
```

