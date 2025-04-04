# Detectron2 Object Detection and Segmentation

A robust implementation of object detection and instance segmentation using Facebook AI Research's Detectron2 framework.

## Overview

This project provides a modular, optimized pipeline for training, evaluating, and deploying computer vision models with Detectron2. The implementation follows best practices for performance optimization, data processing, and model training in production environments.

## Project Structure

- **base_config.py**: Core configuration settings and path management
- **detectron_config.py**: Detectron2-specific configurations and model setup
- **dataset_config.py**: Dataset registration, preprocessing, and augmentation
- **main.py**: Entry point for training and evaluation workflows

## Features

- Configurable model architectures (Faster R-CNN, Mask R-CNN, RetinaNet)
- Optimized data loading and preprocessing pipeline
- Comprehensive evaluation metrics and visualization tools
- Production-ready implementation with performance optimizations

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Detectron2
- Additional dependencies in `requirements.txt`

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python main.py
   ```

## Dataset Preparation

Place your datasets in the `Data` directory with the following structure:

```
Data/
├── train/
│   ├── images/
│   └── annotations/
├── valid/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

## Usage

### Training

```bash
# Basic training with default settings
python main.py --mode train

# Training with custom configuration
python main.py --mode train --config configs/mask_rcnn_R_50_FPN.yaml
```

### Evaluation

```bash
# Evaluate a trained model
python main.py --mode eval --model_path output/model_final.pth

# Evaluate with visualization
python main.py --mode eval --model_path output/model_final.pth --visualize
```

### Inference

```bash
# Run inference on a single image
python main.py --mode infer --model_path output/model_final.pth --input path/to/image.jpg

# Run inference on a directory of images
python main.py --mode infer --model_path output/model_final.pth --input path/to/images/
```

## Performance Optimization

The implementation includes several optimizations:
- Vectorized data preprocessing using NumPy and OpenCV
- Efficient data loading with custom DataLoader
- GPU acceleration and mixed precision training
- Caching mechanisms for faster training iterations

## Outputs

All outputs including trained models, evaluation results, and visualizations are saved to the `output/` directory with the following structure:

```
output/
├── models/
│   ├── model_final.pth
│   └── checkpoints/
├── eval/
│   ├── metrics.json
│   └── visualizations/
└── logs/
    └── training_log.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Facebook AI Research](https://github.com/facebookresearch/detectron2) for the Detectron2 framework
