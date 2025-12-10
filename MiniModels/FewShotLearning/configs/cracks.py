"""Configuration for crack classification (Crack vs Fracture)."""

import os
from fsl.core import ModelConfig, DataConfig


def get_config() -> tuple[ModelConfig, DataConfig]:
    """Get configuration for crack classification task.
    
    Returns:
        (ModelConfig, DataConfig) tuple
    """
    model_config = ModelConfig(
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=True,
        n_way=2,  # Crack vs Fracture
        n_shot=2,
        n_query=1,
        n_training_episodes=500,
        learning_rate=0.0001,
        random_seed=42,
    )
    
    base_dir = r"C:\Users\sobha\Desktop\detectron2\Code\Implement Detectron 2\MiniModels"
    data_config = DataConfig(
        train_root=os.path.join(base_dir, "datasets", "Crack", "Train"),
        test_root=os.path.join(base_dir, "datasets", "Crack", "Test"),
        output_root="output/crack",
    )
    
    return model_config, data_config

