# Quick Start Guide

## Architecture at a Glance

```
┌─────────────────────────────────────────────┐
│           Few-Shot Learning Library         │
├─────────────────────────────────────────────┤
│                                             │
│  fsl/                 (Core Library)        │
│  ├── core/                                  │
│  │   ├── config.py      ModelConfig        │
│  │   ├── model.py       PrototypicalNet    │
│  │   ├── dataset.py     DatasetManager     │
│  │   ├── trainer.py     Trainer            │
│  │   └── evaluator.py   Evaluator          │
│  ├── pipeline.py        Pipeline           │
│  └── utils.py           Utilities          │
│                                             │
│  configs/             (Model Configs)       │
│  ├── roots.py          3-way classification │
│  └── cracks.py         2-way classification │
│                                             │
│  scripts/             (Entry Points)        │
│  ├── train.py          Training script     │
│  ├── evaluate.py       Evaluation script   │
│  └── inference.py      Inference script    │
│                                             │
└─────────────────────────────────────────────┘
```

## 3-Step Usage

### Step 1: Import
```python
from fsl import Pipeline
from configs import roots  # or cracks, or your custom config
```

### Step 2: Configure
```python
model_config, data_config = roots.get_config()
pipeline = Pipeline(model_config, data_config)
```

### Step 3: Train
```python
accuracy = pipeline.run()
pipeline.save_model("outputs/model.pth")
```

## Add New Model in 30 Seconds

Create `configs/my_model.py`:
```python
from fsl.core import ModelConfig, DataConfig

def get_config():
    return (
        ModelConfig(n_way=4, n_shot=5),
        DataConfig(train_root="path/to/train", test_root="path/to/test")
    )
```

Use it:
```python
from configs import my_model
model_config, data_config = my_model.get_config()
```

## Design Philosophy

### Before (❌ Bad)
- Files had misleading names (`data_manager.py`)
- Everything in one folder (no organization)
- Debug/test scripts mixed with core code
- Hard to add new models (edit core files)

### After (✅ Good)
- Clear names (`dataset.py`, `model.py`, `pipeline.py`)
- Organized structure (core/configs/scripts)
- Clean separation of concerns
- Easy to add models (just add config file)

## File Organization Logic

```
fsl/                   # Library (import anywhere)
  core/                # Core components (don't edit often)
  pipeline.py          # Orchestration
  utils.py             # Helpers

configs/               # Model definitions (edit here for new models)
  roots.py
  cracks.py
  custom.py            # Add more as needed

scripts/               # Entry points (convenience wrappers)
  train.py
  evaluate.py
  inference.py
```

## Multi-Model Workflow

```python
# Model 1: Roots
from configs import roots
model_cfg1, data_cfg1 = roots.get_config()
pipeline1 = Pipeline(model_cfg1, data_cfg1)
pipeline1.run()
pipeline1.save_model("outputs/roots_model.pth")

# Model 2: Cracks
from configs import cracks
model_cfg2, data_cfg2 = cracks.get_config()
pipeline2 = Pipeline(model_cfg2, data_cfg2)
pipeline2.run()
pipeline2.save_model("outputs/cracks_model.pth")
```

## Command Line Usage

```bash
# Activate environment first
.venv\Scripts\activate

# Train
python scripts/train.py

# Evaluate
python scripts/evaluate.py

# Custom training with different config
# Edit scripts/train.py to use different config
```

## What Got Removed

- ❌ `speed_test.py` - benchmarking (not core)
- ❌ `debug_accuracy_differences.py` - debugging (not core)
- ❌ `hyperparameter_optimizer.py` - optimization (not core)
- ❌ `wandb_optimizer.py` - optimization (not core)
- ❌ `load_best_model.py` - now in pipeline
- ❌ Old core files - replaced with clean versions

**Result: ~2,748 lines of non-essential code removed**

## Core Principles

1. **Minimal**: Only what's necessary
2. **Clear**: Obvious naming
3. **Extensible**: Add models via configs
4. **Reusable**: Core library is framework
5. **Professional**: Industry-standard structure

