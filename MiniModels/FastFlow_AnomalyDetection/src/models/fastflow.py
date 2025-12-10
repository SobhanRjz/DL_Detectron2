"""FastFlow model architecture and wrapper"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.constants as const

def subnet_conv_func(kernel_size, hidden_ratio):
    """Create subnet function for normalizing flow
    
    Args:
        kernel_size: Convolution kernel size
        hidden_ratio: Hidden channel ratio
        
    Returns:
        Function that creates subnet
    """
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )
    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    """Create FastFlow normalizing flow network
    
    Args:
        input_chw: Input shape (channels, height, width)
        conv3x3_only: Whether to use only 3x3 convolutions
        hidden_ratio: Hidden channel ratio
        flow_steps: Number of flow steps
        clamp: Affine clamping value
        
    Returns:
        FrEIA SequenceINN
    """
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    """FastFlow anomaly detection model"""
    
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
        pretrained_backbone_path=None,
    ):
        """Initialize FastFlow model
        
        Args:
            backbone_name: Name of backbone (e.g., 'resnet18')
            flow_steps: Number of normalizing flow steps
            input_size: Input image size
            conv3x3_only: Use only 3x3 convolutions
            hidden_ratio: Hidden channel ratio
            pretrained_backbone_path: Path to pretrained backbone weights
        """
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            # Load from local if path provided, otherwise download
            if pretrained_backbone_path and os.path.exists(pretrained_backbone_path):
                print(f"Loading backbone from local: {pretrained_backbone_path}")
                self.feature_extractor = timm.create_model(
                    backbone_name,
                    pretrained=False,
                    features_only=True,
                    out_indices=[1, 2, 3],
                )
                state_dict = torch.load(pretrained_backbone_path, map_location='cpu')
                self.feature_extractor.load_state_dict(state_dict)
            else:
                os.environ["http_proxy"] = "http://127.0.0.1:10808"
                os.environ["https_proxy"] = "http://127.0.0.1:10808"
                # INSERT_YOUR_CODE

                print(f"Downloading backbone: {backbone_name}")
                self.feature_extractor = timm.create_model(
                    backbone_name,
                    pretrained=True,
                    features_only=True,
                    out_indices=[1, 2, 3],
                )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # For transformers, use their pretrained norm w/o grad
            # For resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary with 'loss' and optionally 'anomaly_map'
        """
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            n_elements = output.size(1) * output.size(2) * output.size(3)
            loss += torch.mean(
                (0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets) / n_elements
            )
            outputs.append(output)
        ret = {"loss": loss}

        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                a_map = torch.mean(output**2, dim=1, keepdim=True)
                a_map = F.interpolate(
                    a_map,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map
        return ret


class FastFlowModel:
    """OOP wrapper for FastFlow model with easier interface"""
    
    def __init__(self, config, device='cuda'):
        """Initialize FastFlow model wrapper
        
        Args:
            config: ConfigManager or dict with model configuration
            device: Device to run model on
        """
        self.config = config if isinstance(config, dict) else config.to_dict()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = self._build_model()
        
    def _build_model(self):
        """Build FastFlow model from config"""
        model = FastFlow(
            backbone_name=self.config["backbone_name"],
            flow_steps=self.config["flow_step"],
            input_size=self.config["input_size"],
            conv3x3_only=self.config["conv3x3_only"],
            hidden_ratio=self.config["hidden_ratio"],
            pretrained_backbone_path=self.config.get("pretrained_backbone_path", None),
        )
        model.to(self.device)
        print(
            "Model Parameters: {}".format(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
        )
        return model
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Model loaded from: {checkpoint_path}")
        
    def save_checkpoint(self, checkpoint_path, epoch, optimizer=None):
        """Save model checkpoint
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
            optimizer: Optimizer state (optional)
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
        }
        if optimizer:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        self.model.to(device)
        return self
    
    def __call__(self, *args, **kwargs):
        """Forward pass through model"""
        return self.model(*args, **kwargs)
    
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()

