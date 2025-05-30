"""Neural network models for few-shot learning."""

import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50

from config import ModelConfig


class PrototypicalNetworks(nn.Module):
    """Prototypical Networks for few-shot learning."""
    
    def __init__(self, backbone: nn.Module, freeze_backbone: bool = True) -> None:
        """Initialize Prototypical Networks.
        
        Args:
            backbone: Feature extraction backbone network
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters except the last few layers."""
        # Freeze all layers except the last residual block and final layers
        for name, param in self.backbone.named_parameters():
            # Keep layer4 (final residual block) and any fc/classifier layers trainable
            if not ('layer4' in name or 'fc' in name or 'classifier' in name):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for prototypical networks.
        
        Args:
            support_images: Support set images [n_support, C, H, W]
            support_labels: Support set labels [n_support]
            query_images: Query set images [n_query, C, H, W]
            
        Returns:
            Classification scores for query images [n_query, n_way]
        """
        # Extract features
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)

        # Calculate prototypes
        n_way = len(torch.unique(support_labels))
        
        # Create prototypes by averaging support features for each class
        prototypes = []
        for label in range(n_way):
            # Get indices of support samples for this class
            class_mask = (support_labels == label)
            # Average the features for this class
            class_prototype = z_support[class_mask].mean(dim=0)
            prototypes.append(class_prototype)
        
        # Stack prototypes to create [n_way, feature_dim] tensor
        z_proto = torch.stack(prototypes, dim=0)

        # Compute distances and scores
        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        
        return scores


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_backbone(config: ModelConfig) -> nn.Module:
        """Create backbone network.
        
        Args:
            config: Model configuration
            
        Returns:
            Backbone network
        """
        if config.BACKBONE_NAME == "resnet18":
            backbone = resnet18(pretrained=config.PRETRAINED)
            backbone.fc = nn.Identity()
            return backbone
        elif config.BACKBONE_NAME == "resnet34":
            backbone = resnet34(pretrained=config.PRETRAINED)
            backbone.fc = nn.Identity()
            return backbone
        elif config.BACKBONE_NAME == "resnet50":
            backbone = resnet50(pretrained=config.PRETRAINED)
            backbone.fc = nn.Identity()
            return backbone
        else:
            raise ValueError(f"Unsupported backbone: {config.BACKBONE_NAME}")
    @staticmethod
    def create_prototypical_network(config: ModelConfig) -> PrototypicalNetworks:
        """Create Prototypical Networks model.
        
        Args:
            config: Model configuration
            
        Returns:
            PrototypicalNetworks instance
        """
        backbone = ModelFactory.create_backbone(config)
        return PrototypicalNetworks(backbone, freeze_backbone=config.FREEZE_BACKBONE) 