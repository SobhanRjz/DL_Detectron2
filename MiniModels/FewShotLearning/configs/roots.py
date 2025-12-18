"""Configuration for root classification (Root mass, Root tap, Root fine)."""

import os
from fsl.core import ModelConfig, DataConfig


def get_config(n_shot: int = 2) -> tuple[ModelConfig, DataConfig]:
    """Get configuration for root classification task.
    
    Returns:
        (ModelConfig, DataConfig) tuple
    """
    model_config = ModelConfig(
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=True,
        n_way=3, # root mass, root tap, root fine
        n_shot=n_shot,  # 2 shots per class
        n_query=1,
        n_training_episodes=500,
        learning_rate=0.0001,
        random_seed=42,
    )
    
    base_dir = r"C:\Users\sobha\Desktop\detectron2\Code\Implement Detectron 2\MiniModels\FewShotLearning"
    data_config = DataConfig(
        train_root=os.path.join(base_dir, "datasets",  "Root", "Train"),
        test_root=os.path.join(base_dir, "datasets", "Root",  "Test"),
        output_root="MiniModels/output/root",
    )
    
    return model_config, data_config

