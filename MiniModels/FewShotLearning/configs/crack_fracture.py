"""Configuration for crack fracture classification."""

import os
from src.core import ModelConfig, DataConfig


def get_config() -> tuple[ModelConfig, DataConfig]:
    """Get configuration for crack fracture classification task.

    Returns:
        (ModelConfig, DataConfig) tuple
    """
    model_config = ModelConfig(
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=True,
        n_way=4, # CF_S (spiral), CF_M (multiple), CF_L (longitudinal), CF_C (circular)
        n_shot=3,
        n_query=1,
        n_training_episodes=500,
        learning_rate=0.0001,
        random_seed=42,
    )

    base_dir = r"C:\Users\sobha\Desktop\detectron2\Code\Implement Detectron 2\MiniModels\FewShotLearning"
    data_config = DataConfig(
        train_root=os.path.join(base_dir, "datasets", "Crack_Fracture", "Train"),
        test_root=os.path.join(base_dir, "datasets", "Crack_Fracture", "Test"),
        output_root="MiniModels\\FewShotLearning\\output\\crack_fracture",
    )

    return model_config, data_config
