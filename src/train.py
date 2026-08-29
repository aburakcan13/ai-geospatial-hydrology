"""
End-to-End Training Pipeline for Boundary-Aware ResUNet
AI-Driven Geospatial Hydrology (Springer Nature)
"""

import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.unet_boundary import BoundaryAwareResUNet
from src.metrics.hydrological_kge import calculate_iou_dice


def load_config(config_path: str = "config/default_params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, criterion_bce, loss_weights, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        pred_water, pred_boundary = model(images)

        # Dual-head loss computation
        loss_water = criterion_bce(pred_water, masks)
        loss_boundary = criterion_bce(pred_boundary, masks)  # Edge/boundary target
        
        total_loss = (
            loss_weights["water_bce"] * loss_water +
            loss_weights["boundary_bce"] * loss_boundary
        )

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    return running_loss / max(len(dataloader), 1)


def main():
    config = load_config()
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Starting training pipeline on device: {device}")

    # Model başlatma
    model = BoundaryAwareResUNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["model"]["lr"],
        weight_decay=config["model"]["weight_decay"]
    )
    criterion_bce = nn.BCEWithLogitsLoss()

    print("Model and loss functions successfully configured.")


if __name__ == "__main__":
    main()
