# main.py
"""
Entry point for the project.
Handles dataset preprocessing, configuration and model training.
"""
import os
import logging
from pathlib import Path
import torch
from Config.basic_config import BASE_PATH, OUTPUT_PATH, DEVICE
from Config.basic_config import detectron2_logger as logger
from Config.detectron2_config import DetectronConfig
from Config.dataset_config import DatasetConfig
from DataSets.preprocess_COCO import COCOJsonProcessor
from Train import mainTrain
from detectron2.engine import launch


def process_datasets():
    """Handle dataset preprocessing and registration"""
    # Use pathlib for more robust path handling
    json_paths = [
        Path(BASE_PATH) / "DataSets" / "images" / split / "_annotations.coco.json"
        for split in ["train", "test", "valid"]
    ]
    
    logger.info("Processing COCO JSON files...")
    coco_processor = COCOJsonProcessor([str(p) for p in json_paths])
    coco_processor.process_files()



    logger.info("Registering datasets...")
    DatasetConfig().register_datasets()

    from collections import Counter
    from detectron2.data import DatasetCatalog
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')  # Use Tkinter backend
    # Load dataset
    dataset_dicts = DatasetCatalog.get("my_dataset_train")

    # Count instances per class
    class_counts = Counter()
    for data in dataset_dicts:
        for annotation in data["annotations"]:
            class_id = annotation["category_id"]
            class_counts[class_id] += 1

    # Create bar plot
    plt.figure(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(classes, counts)
    plt.title('Distribution of Classes in Training Dataset')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Instances')
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(classes[i], count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(str(Path(OUTPUT_PATH) / 'class_distribution.png'))
    plt.close()

    print("Class Distribution:", class_counts)
    print("Distribution plot saved to:", str(Path(OUTPUT_PATH) / 'class_distribution.png'))


def setup_training():
    """Configure training settings and get GPU count"""
    logger.info("Loading Detectron2 configuration...")

    
    detectron_config = DetectronConfig()
    cfg = detectron_config.get_cfg()

    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {num_gpus}")
    
    return cfg, num_gpus

def main():
    """Main execution function"""
    logger.info("Initializing project...")
    
    # Process and register datasets
    process_datasets()
    
    # Setup training configuration
    cfg, num_gpus = setup_training()

    # Launch distributed training
    logger.info("Launching training...")
    launch(
        mainTrain,
        num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto", 
        args=(cfg,"custom"),
    )
    # Save final model
    logger.info("Saving final model...")
    final_model_path = os.path.join(OUTPUT_PATH, "model_final.pth")
    torch.save(cfg.MODEL.STATE_DICT(), final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Configuration loaded successfully with device: {DEVICE}")

if __name__ == "__main__":
    main()
