"""Prototypical Networks implementation."""

import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50

from .config import ModelConfig


class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot classification."""
    
    def __init__(self, backbone: nn.Module, freeze_backbone: bool = True) -> None:
        """Initialize network.
        
        Args:
            backbone: Feature extraction network
            freeze_backbone: Freeze backbone weights except final layers
        """
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self) -> None:
        """Freeze all layers except layer4 and final layers."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = 'layer4' in name or 'fc' in name or 'classifier' in name

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification scores.
        
        Args:
            support_images: [n_support, C, H, W]
            support_labels: [n_support]
            query_images: [n_query, C, H, W]
            
        Returns:
            Classification scores [n_query, n_way]
        """
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)

        n_way = len(torch.unique(support_labels))
        
        prototypes = []
        for label in range(n_way):
            class_mask = (support_labels == label)
            prototypes.append(z_support[class_mask].mean(dim=0))
        
        z_proto = torch.stack(prototypes, dim=0)
        dists = torch.cdist(z_query, z_proto)
        
        return -dists


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_backbone(config: ModelConfig) -> nn.Module:
        """Create backbone network.
        
        Args:
            config: Model configuration
            
        Returns:
            Backbone with identity final layer
        """
        backbones = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
        }
        
        if config.backbone not in backbones:
            raise ValueError(f"Unsupported backbone: {config.backbone}")
        
        backbone = backbones[config.backbone](pretrained=config.pretrained)
        backbone.fc = nn.Identity()
        return backbone
    
    @staticmethod
    def create_model(config: ModelConfig) -> PrototypicalNetwork:
        """Create complete model.
        
        Args:
            config: Model configuration
            
        Returns:
            PrototypicalNetwork instance
        """
        backbone = ModelFactory.create_backbone(config)
        return PrototypicalNetwork(backbone, config.freeze_backbone)

