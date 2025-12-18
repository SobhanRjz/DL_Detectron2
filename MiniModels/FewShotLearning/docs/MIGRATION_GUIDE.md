# Migration Guide

## Old vs New Architecture

### Old Structure (Removed)
```
❌ main_few_shot_learning.py  → fsl/pipeline.py
❌ config.py                   → fsl/core/config.py
❌ models.py                   → fsl/core/model.py
❌ data_manager.py             → fsl/core/dataset.py
❌ trainer.py                  → fsl/core/trainer.py
❌ evaluator.py                → fsl/core/evaluator.py
❌ utils.py                    → fsl/utils.py
```

### New Structure
```
✅ fsl/                        # Organized library
   ├── core/                   # Core components
   └── pipeline.py             # Main pipeline

✅ configs/                    # Separate configs
   ├── roots.py
   └── cracks.py

✅ scripts/                    # Entry points
   ├── train.py
   ├── evaluate.py
   └── inference.py
```

## Code Changes

### Before
```python
from main_few_shot_learning import FewShotLearningPipeline
from config import ModelConfig, DataConfig

config = ModelConfig()
data_config = DataConfig()
pipeline = FewShotLearningPipeline(config, data_config)
```

### After
```python
from fsl import Pipeline
from configs import roots

model_config, data_config = roots.get_config()
pipeline = Pipeline(model_config, data_config)
```

## Key Changes

1. **Imports**: Use `from fsl import ...` instead of importing from individual files
2. **Class Names**: `PrototypicalNetwork` (singular) instead of `PrototypicalNetworks`
3. **Config**: Configs moved to `configs/` directory, organized by task
4. **Pipeline**: Renamed from `FewShotLearningPipeline` to `Pipeline`

## Adding New Models

### Old Approach
Edit `config.py` and change hardcoded paths:
```python
# Not flexible - hardcoded for one task
TRAIN_IMAGE_ROOT: str = "path/to/roots/train"
```

### New Approach
Create new config file for each task:
```python
# configs/cracks.py
def get_config():
    model_config = ModelConfig(n_way=2, ...)
    data_config = DataConfig(
        train_root="path/to/cracks/train",
        test_root="path/to/cracks/test"
    )
    return model_config, data_config
```

Then use it:
```python
from configs import cracks

model_config, data_config = cracks.get_config()
pipeline = Pipeline(model_config, data_config)
```

## Benefits

1. ✅ **Clear naming**: `dataset.py`, `model.py`, `pipeline.py`
2. ✅ **Reusable**: Import `fsl` anywhere
3. ✅ **Multi-model**: Add configs without touching core
4. ✅ **Clean**: Removed debug/test/optimization scripts
5. ✅ **Organized**: Separation of concerns (core/configs/scripts)

