# FastFlow Anomaly Detection - Professional Implementation

An unofficial PyTorch implementation of [_FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows_](https://arxiv.org/abs/2111.07677) (Jiawei Yu et al.) with professional Object-Oriented Programming (OOP) design.

## ✨ Features

- **Professional OOP Architecture**: Modular, maintainable, and extensible codebase
- **Separate Training & Inference**: Clear separation of concerns with dedicated scripts
- **Flexible Inference Options**: Single image, batch processing, and folder processing
- **Comprehensive Metrics**: AUROC evaluation, detailed statistics, and visualization
- **Easy Configuration**: YAML-based configuration management
- **Checkpoint Management**: Automatic checkpoint saving and resuming
- **Production Ready**: Clean API, proper error handling, and documentation

## 📁 Project Structure

```
FastFlow_AnomalyDetection/
├── src/                          # Source code package
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── config_manager.py     # YAML config loader and validator
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   └── fastflow.py           # FastFlow model and OOP wrapper
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py            # Dataset class
│   │   └── dataloader.py         # DataLoader manager
│   ├── training/                 # Training pipeline
│   │   ├── __init__.py
│   │   └── trainer.py            # Trainer class with full training loop
│   ├── inference/                # Inference pipeline
│   │   ├── __init__.py
│   │   └── inference_engine.py   # Inference engine with batch support
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── helpers.py            # Helper utilities (AverageMeter, etc.)
│       └── metrics.py            # Evaluation metrics (ROC-AUC)
├── configs/                      # Model configurations
│   ├── resnet18.yaml
│   ├── wide_resnet50_2.yaml
│   ├── deit.yaml
│   └── cait.yaml
├── data/                         # Dataset directory
├── main_train.py                 # Training entry point
├── inference.py                  # Inference entry point
├── constants.py                  # Global constants
├── requirements.txt              # Python dependencies
├── LICENSE                       # Apache 2.0 License
└── README.md                     # This file
```

## 🚀 Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

We use [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset structure:

```
data/
└── anomaly_data/
    └── split_data/
        └── [category_name]/
            ├── train/
            │   └── good/          # Normal training images
            └── test/
                ├── good/          # Normal test images
                └── defect_*/      # Anomaly test images
```

## 🎓 Training

### Basic Training

```bash
python main_train.py \
    --config configs/resnet18.yaml \
    --data data \
    --category pipe_anomaly \
    --epochs 100
```

### Advanced Training Options

```bash
python main_train.py \
    --config configs/resnet18.yaml \
    --data data \
    --category pipe_anomaly \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-3 \
    --weight-decay 1e-5 \
    --eval-interval 10 \
    --checkpoint-interval 10 \
    --device cuda \
    --num-workers 0
```

### Resume Training

```bash
python main_train.py \
    --config configs/resnet18.yaml \
    --data data \
    --category pipe_anomaly \
    --resume checkpoints/exp0/50.pt
```

## 🔍 Inference

### Single Image Inference

```bash
python inference.py \
    --config configs/resnet18.yaml \
    --checkpoint checkpoints/exp0/best.pt \
    --image path/to/image.jpg \
    --threshold 1000 \
    --visualize
```

### Batch Inference

```bash
python inference.py \
    --config configs/resnet18.yaml \
    --checkpoint checkpoints/exp0/best.pt \
    --batch image1.jpg image2.jpg image3.jpg \
    --threshold 1000 \
    --output results.json
```

### Folder Inference

```bash
python inference.py \
    --config configs/resnet18.yaml \
    --checkpoint checkpoints/exp0/best.pt \
    --folder path/to/images/ \
    --batch-size 8 \
    --threshold 1000 \
    --output results.csv
```

### Save Results and Visualization

```bash
python inference.py \
    --config configs/resnet18.yaml \
    --checkpoint checkpoints/exp0/best.pt \
    --image test_image.jpg \
    --threshold 1000 \
    --visualize \
    --save-viz visualization.png \
    --output result.json
```

## 💻 Programmatic Usage

### Training Example

```python
from src.config import ConfigManager
from src.models import FastFlowModel
from src.data import DataLoaderManager
from src.training import FastFlowTrainer

# Load configuration
config = ConfigManager("configs/resnet18.yaml")

# Setup data
data_manager = DataLoaderManager(
    data_root="data",
    category="pipe_anomaly",
    input_size=config['input_size'],
    batch_size=4
)
train_loader, test_loader = data_manager.get_loaders()

# Build model
model = FastFlowModel(config, device='cuda')

# Train
trainer = FastFlowTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device='cuda'
)
trainer.train(num_epochs=100)
```

### Inference Example

```python
from src.inference import FastFlowInference

# Initialize inference engine
engine = FastFlowInference(
    config_path="configs/resnet18.yaml",
    checkpoint_path="checkpoints/exp0/best.pt",
    device='cuda',
    threshold=1000
)

# Single image inference
result = engine.predict_single("test_image.jpg", return_map=True)
print(f"Anomaly Score: {result['anomaly_score']:.4f}")
print(f"Is Anomaly: {result['is_anomaly']}")

# Visualize result
engine.visualize_result(result, save_path="visualization.png")

# Batch inference
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = engine.predict_batch(image_paths, batch_size=8)

# Print statistics
engine.print_statistics(results)

# Save results
engine.save_results(results, "results.json")
```

## ⚙️ Configuration

Configuration files are in YAML format. Example (`configs/resnet18.yaml`):

```yaml
backbone_name: resnet18
input_size: 256
conv3x3_only: False
hidden_ratio: 1.0
flow_step: 8
```

### Supported Backbones

- `resnet18`
- `wide_resnet50_2`
- `deit_base_distilled_patch16_384`
- `cait_m48_448`

## 📈 Performance

As reported in the original paper:

| AUROC (Mean)      | wide-resnet-50 | resnet18  | DeiT      | CaiT      |
| ----------------- | -------------- | --------- | --------- | --------- |
| MVTec-AD (Paper)  | 0.981          | 0.972     | 0.981     | 0.985     |

## 🏗️ Architecture Highlights

### OOP Design Benefits

1. **Modularity**: Each component (model, data, training, inference) is independent
2. **Reusability**: Components can be easily reused in other projects
3. **Maintainability**: Clear structure makes code easy to understand and modify
4. **Extensibility**: New features can be added without breaking existing code
5. **Testability**: Components can be tested independently

### Key Classes

- **ConfigManager**: Loads and validates YAML configurations
- **FastFlowModel**: OOP wrapper for FastFlow with checkpoint management
- **DataLoaderManager**: Manages train and test data loaders
- **FastFlowTrainer**: Complete training pipeline with evaluation
- **FastFlowInference**: Production-ready inference engine with batch support

## 📝 Command Line Arguments

### Training (`main_train.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to config file | `configs/resnet18.yaml` |
| `--data` | Dataset root path | `data` |
| `--category` | Dataset category | `pipe_anomaly` |
| `--epochs` | Number of epochs | `100` |
| `--batch-size` | Batch size | `4` |
| `--lr` | Learning rate | `1e-3` |
| `--weight-decay` | Weight decay | `1e-5` |
| `--checkpoint-dir` | Checkpoint directory | `_fastflow_experiment_checkpoints` |
| `--resume` | Resume from checkpoint | `None` |
| `--eval-interval` | Evaluation interval | `10` |
| `--checkpoint-interval` | Checkpoint save interval | `10` |
| `--device` | Device (cuda/cpu) | `cuda` |
| `--num-workers` | DataLoader workers | `0` |

### Inference (`inference.py`)

| Argument | Description | Required |
|----------|-------------|----------|
| `--config` | Path to config file | Yes |
| `--checkpoint` | Path to checkpoint | Yes |
| `--image` | Single image path | Yes* |
| `--folder` | Folder path | Yes* |
| `--batch` | List of image paths | Yes* |
| `--device` | Device (cuda/cpu) | No |
| `--threshold` | Anomaly threshold | No |
| `--batch-size` | Batch size | No |
| `--output` | Output file path | No |
| `--visualize` | Show visualization | No |
| `--save-viz` | Save visualization | No |
| `--no-verbose` | Disable verbose output | No |

*One of `--image`, `--folder`, or `--batch` is required

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper: [FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows](https://arxiv.org/abs/2111.07677)
- Based on the community implementation with enhancements for production use
- MVTec-AD dataset: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## 📧 Contact

For questions and issues, please open an issue on GitHub.

---

**Note**: This is an unofficial implementation with professional OOP refactoring for production use.
