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
python scripts/train.py
```

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

