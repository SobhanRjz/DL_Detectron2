"""

# Project Documentation

This project uses Detectron2 for object detection and segmentation.

## Structure

- **base_config.py**: Handles base paths and settings.
- **detectron_config.py**: Configuration for Detectron2.
- **dataset_config.py**: Dataset registration and preprocessing.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python main.py
   ```

## Datasets

- Place datasets in the `Data` folder structured as:
  ```
  Data/
  ├── train/
  ├── valid/
  └── test/
  ```

## Outputs

- Generated outputs will be saved in the `output/` directory.

"""
