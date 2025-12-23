"""Configuration for joint displacement classification (JD vs OJ)."""

import os
from src.core import ModelConfig, DataConfig


def get_config(n_shot: int = 2) -> tuple[ModelConfig, DataConfig]:
    """Get configuration for joint displacement classification task.

    Returns:
        (ModelConfig, DataConfig) tuple
    """
    model_config = ModelConfig(
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=True,
        n_way=2,  # JD, OJ
        n_shot=n_shot,
        n_query=1,
        n_training_episodes=500,
        learning_rate=0.0001,
        random_seed=42,
    )

    base_dir = r"C:\Users\sobha\Desktop\detectron2\Code\Implement Detectron 2\MiniModels\FewShotLearning"
    data_config = DataConfig(
        train_root=os.path.join(base_dir, "datasets", "JointDisp", "Train"),
        test_root=os.path.join(base_dir, "datasets", "JointDisp", "Test"),
        output_root="MiniModels\\FewShotLearning\\output\\joint_disp",
    )

    return model_config, data_config
