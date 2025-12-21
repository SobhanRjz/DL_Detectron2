"""Evaluation script."""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src import Pipeline
from configs import roots


def main() -> None:
    """Run evaluation."""
    model_config, data_config = roots.get_config()
    
    pipeline = Pipeline(model_config, data_config)
    pipeline.load_model("outputs/model.pth")
    
    accuracy = pipeline.evaluate()
    print(f"\nTest accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

