# main.py
"""
Entry point for the project.
"""
from Config.basic_config import BASE_PATH, OUTPUT_PATH, DEVICE
from Config.detectron2_config import DetectronConfig
from Config.dataset_config import *
from DataSets.preprocess_COCO import COCOJsonProcessor
from Train.main_train import mainTrain
from detectron2.engine import launch
def main():
    print("Initializing project...")

    # Paths to COCO JSON files for training, testing, and validation
    list_json = [
        f"{BASE_PATH}/DataSets/images/{split}/_annotations.coco.json"
        for split in ["train", "test", "valid"]
    ]

    # Preprocess COCO JSON files
    print("Processing COCO JSON files...")
    coco_processor = COCOJsonProcessor(list_json)
    coco_processor.process_files()

    # Register datasets
    print("Registering datasets...")
    dataset_config = DatasetConfig()
    dataset_config.register_datasets()

    # Load Detectron2 configuration
    print("Loading Detectron2 configuration...")
    detectron_config = DetectronConfig()
    cfg = detectron_config.get_cfg()

    # Dynamically detect the number of GPUs
    num_gpus = get_num_gpus()

    # Launch the main training function
    print("Launching main training function...")
    invoke_main(cfg, num_gpus)

    print(f"Configuration loaded successfully with device: {DEVICE}")

def get_num_gpus():
    """
    Returns the number of GPUs available in the system.
    """
    import torch
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    return num_gpus

def invoke_main(cfg, num_gpus) -> None:
    launch(
        mainTrain,
        num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(cfg,),
    )

if __name__ == "__main__":
    main()

