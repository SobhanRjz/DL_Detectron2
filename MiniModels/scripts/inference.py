"""Inference script for single images."""

import sys
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from fsl.core import ModelFactory
from fsl.utils import get_device
from configs import roots


class Predictor:
    """Handles model inference."""
    
    def __init__(self, model_path: str) -> None:
        """Initialize predictor.
        
        Args:
            model_path: Path to saved model
        """
        model_config, _ = roots.get_config()
        
        self.device = get_device(model_config.device)
        self.model = ModelFactory.create_model(model_config)
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(model_config.image_size),
            transforms.ToTensor(),
        ])
    
    def predict(
        self,
        support_images: list[str],
        support_labels: list[int],
        query_image: str
    ) -> int:
        """Predict class for query image.
        
        Args:
            support_images: Paths to support images
            support_labels: Labels for support images
            query_image: Path to query image
            
        Returns:
            Predicted class label
        """
        support_tensors = torch.stack([
            self.transform(Image.open(img).convert('RGB'))
            for img in support_images
        ]).to(self.device)
        
        support_labels_tensor = torch.tensor(support_labels).to(self.device)
        
        query_tensor = self.transform(
            Image.open(query_image).convert('RGB')
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            scores = self.model(support_tensors, support_labels_tensor, query_tensor)
            _, pred = torch.max(scores, 1)
        
        return pred.item()


def main() -> None:
    """Run inference demo."""
    predictor = Predictor("MiniModels/output/root/best_model/model.pth")
    
    # Demo: provide support set and query (2 support images per class = 6 total)
    support_imgs = [
        "MiniModels/datasets/Root/Support_Set/root_mass_1.jpg",
        "MiniModels/datasets/Root/Support_Set/root_mass_2.jpg",
        "MiniModels/datasets/Root/Support_Set/root_tap_1.jpg",
        "MiniModels/datasets/Root/Support_Set/root_tap_2.jpg",
        "MiniModels/datasets/Root/Support_Set/root_fine_1.jpg",
        "MiniModels/datasets/Root/Support_Set/root_fine_2.jpg",
    ]
    support_lbls = [0, 0, 1, 1, 2, 2]  # 2 images for each of 3 classes
    query_img = r"C:\Users\sobha\Desktop\detectron2\Code\Implement Detectron 2\MiniModels\datasets\Root\Train\RM\(4.aaaaaaaa.v2i.coco)_Root_21.jpg"
    
    pred = predictor.predict(support_imgs, support_lbls, query_img)
    print(f"Predicted class: {pred}")


if __name__ == "__main__":
    main()

