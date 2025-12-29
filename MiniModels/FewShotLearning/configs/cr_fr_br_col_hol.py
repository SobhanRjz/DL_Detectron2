"""Configuration for 5-way crack, fracture, break, collision, hole classification."""

import os
from src.core import ModelConfig, DataConfig


def get_config() -> tuple[ModelConfig, DataConfig]:
    """Get configuration for 5-way Cr_Fr_Br_Xr_Hol classification task.

    Returns:
        (ModelConfig, DataConfig) tuple
    """
    model_config = ModelConfig(
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=True,
        n_way=5,  # Cr (Crack), Fr (Fracture), Br (Break), Xr (Collision), Hol (Hole)
        n_shot=1,  # Reduced due to limited test samples
        n_query=1,  # Reduced due to limited test samples
        n_training_episodes=150,
        learning_rate=0.0001,
        random_seed=42,
    )

    base_dir = r"C:\Users\sobha\Desktop\detectron2\Code\Implement Detectron 2\MiniModels\FewShotLearning"
    data_config = DataConfig(
        train_root=os.path.join(base_dir, "datasets", "Cr_Fr_Br_Xr_Hol", "Train"),
        test_root=os.path.join(base_dir, "datasets", "Cr_Fr_Br_Xr_Hol", "Test"),
        output_root="MiniModels\\FewShotLearning\\output\\Cr_Fr_Br_Xr_Hol",
    )

    return model_config, data_config
