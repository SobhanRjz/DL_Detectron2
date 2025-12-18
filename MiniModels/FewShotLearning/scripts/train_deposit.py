"""Training script for Deposit classification."""

import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from fsl import Pipeline
from configs import deposit


def main() -> None:
    """Run training for deposit classification."""
    model_config, data_config = deposit.get_config()

    pipeline = Pipeline(model_config, data_config)
    final_acc = pipeline.run()

    output_path = os.path.join(data_config.output_root, "best_model")
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "model.pth")
    pipeline.save_model(model_path)
    print(f"\nFinal accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()