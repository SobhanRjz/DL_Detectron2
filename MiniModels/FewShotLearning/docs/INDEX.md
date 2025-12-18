# Documentation Index

## Quick Access

### 📚 Start Here
- **[QUICK_START.md](QUICK_START.md)** - 3-step guide to get started
- **[README.md](README.md)** - Overview and basic usage

### 🏗️ Architecture
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture design
- **[BEFORE_AFTER.md](BEFORE_AFTER.md)** - Complete comparison with old structure

### 🔄 Migration
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - How to migrate from old code
- **[SUMMARY.md](SUMMARY.md)** - What changed and why

## File Structure

```
Root/
├── 📖 Documentation
│   ├── INDEX.md              (This file)
│   ├── QUICK_START.md        (Quick reference)
│   ├── README.md             (Overview)
│   ├── ARCHITECTURE.md       (Design details)
│   ├── BEFORE_AFTER.md       (Comparison)
│   ├── MIGRATION_GUIDE.md    (Migration help)
│   └── SUMMARY.md            (Summary)
│
├── 📦 fsl/                   (Core Library)
│   ├── core/
│   │   ├── config.py         (ModelConfig, DataConfig)
│   │   ├── model.py          (PrototypicalNetwork)
│   │   ├── dataset.py        (DatasetManager)
│   │   ├── trainer.py        (Trainer)
│   │   ├── evaluator.py      (Evaluator)
│   │   └── __init__.py       (Exports)
│   ├── pipeline.py           (Pipeline orchestration)
│   ├── utils.py              (Utilities)
│   └── __init__.py           (Package exports)
│
├── ⚙️ configs/               (Model Configurations)
│   ├── roots.py              (Root Mass/Tap/Fine - 3-way)
│   ├── cracks.py             (Crack/Fracture - 2-way)
│   └── __init__.py           (Config exports)
│
├── 🚀 scripts/               (Entry Points)
│   ├── train.py              (Training script)
│   ├── evaluate.py           (Evaluation script)
│   └── inference.py          (Inference script)
│
├── 📊 Data/                  (Your dataset)
│   ├── Train/
│   └── Test/
│
└── 💾 outputs/               (Model checkpoints)
```

## Quick Navigation

### I want to...

#### Train a model
→ See [QUICK_START.md](QUICK_START.md) - Step 3
→ Run `python scripts/train.py`

#### Add a new model
→ See [QUICK_START.md](QUICK_START.md) - "Add New Model in 30 Seconds"
→ Create `configs/your_model.py`

#### Understand the design
→ See [ARCHITECTURE.md](ARCHITECTURE.md)
→ See [BEFORE_AFTER.md](BEFORE_AFTER.md) for comparison

#### Migrate from old code
→ See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

#### Know what changed
→ See [SUMMARY.md](SUMMARY.md)
→ See [BEFORE_AFTER.md](BEFORE_AFTER.md)

## Core Components

### fsl/core/config.py
```python
ModelConfig  # Model and training parameters
DataConfig   # Dataset paths
```

### fsl/core/model.py
```python
PrototypicalNetwork  # Few-shot learning model
ModelFactory         # Model creation utilities
```

### fsl/core/dataset.py
```python
DatasetManager  # Data loading and preprocessing
```

### fsl/core/trainer.py
```python
Trainer  # Training logic
```

### fsl/core/evaluator.py
```python
Evaluator  # Evaluation logic
```

### fsl/pipeline.py
```python
Pipeline  # End-to-end orchestration
```

## Configuration Files

### configs/roots.py
3-way classification: Root Mass, Root Tap, Root Fine
```python
from configs import roots
model_config, data_config = roots.get_config()
```

### configs/cracks.py
2-way classification: Crack, Fracture
```python
from configs import cracks
model_config, data_config = cracks.get_config()
```

## Scripts

### scripts/train.py
Entry point for training
```bash
python scripts/train.py
```

### scripts/evaluate.py
Entry point for evaluation
```bash
python scripts/evaluate.py
```

### scripts/inference.py
Entry point for inference
```bash
python scripts/inference.py
```

## Documentation Purpose

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **INDEX.md** | Navigation hub | Finding what you need |
| **QUICK_START.md** | Get started fast | First time setup |
| **README.md** | Basic overview | Understanding basics |
| **ARCHITECTURE.md** | Design details | Understanding structure |
| **BEFORE_AFTER.md** | Comparison | Understanding changes |
| **MIGRATION_GUIDE.md** | Code updates | Migrating old code |
| **SUMMARY.md** | What changed | Quick reference |

## Key Concepts

### Multi-Model Support
One architecture, multiple models through config files:
- `configs/roots.py` - Root classification
- `configs/cracks.py` - Crack classification  
- `configs/custom.py` - Your custom model

### Separation of Concerns
- **fsl/** - Core library (don't edit often)
- **configs/** - Model definitions (edit to add models)
- **scripts/** - Entry points (convenience wrappers)

### Clean Design
- Clear naming: `dataset.py`, `model.py`, `pipeline.py`
- Minimal API: Only essential classes exposed
- Extensible: Add models without touching core
- Reusable: Import `fsl` anywhere

## Getting Help

1. Check [QUICK_START.md](QUICK_START.md) for common tasks
2. See [ARCHITECTURE.md](ARCHITECTURE.md) for design questions
3. Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) if migrating
4. Review [BEFORE_AFTER.md](BEFORE_AFTER.md) to understand changes

## Statistics

- **Core Library**: ~676 lines
- **Configurations**: ~58 lines
- **Scripts**: ~114 lines
- **Total**: ~848 lines (clean, focused code)
- **Removed**: ~2,748 lines of non-essential code
- **Reduction**: 77% less code overall

