"""
Download and save pretrained backbone models locally.
Run this script once to download models, then use them offline.
"""
import timm
import torch
import os

def download_and_save_backbone(backbone_name, save_dir="pretrained_backbones"):
    """Download a backbone model and save it locally."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading {backbone_name}...")
    
    # Create model with pretrained weights
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        features_only=True,
        out_indices=[1, 2, 3],
    )
    
    # Save the model state dict
    save_path = os.path.join(save_dir, f"{backbone_name}.pth")
    torch.save(model.state_dict(), save_path)
    
    print(f"✓ Saved to: {save_path}")
    
    # Also save feature info for later use
    feature_info = {
        'channels': model.feature_info.channels(),
        'reduction': model.feature_info.reduction(),
    }
    info_path = os.path.join(save_dir, f"{backbone_name}_info.pth")
    torch.save(feature_info, info_path)
    
    print(f"✓ Feature info saved to: {info_path}")
    return save_path


if __name__ == "__main__":
    # Download commonly used backbones
    backbones = [
        "resnet18"
    ]
    
    print("="*70)
    print("Downloading Pretrained Backbones for Offline Use")
    print("="*70)
    
    for backbone in backbones:
        try:
            download_and_save_backbone(backbone)
            print()
        except Exception as e:
            print(f"✗ Error downloading {backbone}: {e}")
            print()
    
    print("="*70)
    print("Download Complete!")
    print("You can now use these models offline.")
    print("="*70)

