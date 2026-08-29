"""
Vision Transformer / SegFormer Architecture for Large-Scale Surface Water Delineation
AI-Driven Geospatial Hydrology (Springer Nature)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 8, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class GeospatialSegFormer(nn.Module):
    def __init__(self, in_channels: int = 8, num_classes: int = 1, embed_dim: int = 128):
        """Lightweight Hierarchical Vision Transformer for high-resolution water extraction."""
        super().__init__()
        self.patch_embed = OverlapPatchEmbedding(patch_size=7, stride=4, in_chans=in_channels, embed_dim=embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # All-MLP Decoder Head
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        tokens, h, w = self.patch_embed(x)
        encoded = self.transformer_encoder(tokens)
        
        feature_map = encoded.transpose(1, 2).reshape(x.shape[0], -1, h, w)
        out = self.decoder(feature_map)
        
        return F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
