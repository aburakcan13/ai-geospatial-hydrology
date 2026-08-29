"""
Unit Tests for Deep Learning Architectures and Loss Pipelines
AI-Driven Geospatial Hydrology (Springer Nature)
"""

import pytest
import torch
import numpy as np
from src.models.unet_boundary import BoundaryAwareResUNet
from src.metrics.hydrological_kge import calculate_iou_dice, kling_gupta_efficiency


def test_boundary_aware_unet_forward():
    batch_size = 2
    in_channels = 8
    height, width = 128, 128
    dummy_input = torch.randn(batch_size, in_channels, height, width)

    model = BoundaryAwareResUNet(in_channels=in_channels, num_classes=1)
    water_out, boundary_out = model(dummy_input)

    assert water_out.shape == (batch_size, 1, height, width)
    assert boundary_out.shape == (batch_size, 1, height, width)


def test_iou_dice_computation():
    pred = torch.tensor([[10.0, 10.0], [-10.0, -10.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    iou, dice = calculate_iou_dice(pred, target)
    assert pytest.approx(iou, rel=1e-2) == 1.0
    assert pytest.approx(dice, rel=1e-2) == 1.0


def test_kling_gupta_efficiency_perfect_match():
    obs = np.array([1.2, 2.5, 3.8, 4.1, 5.0])
    sim = np.array([1.2, 2.5, 3.8, 4.1, 5.0])

    kge, r, gamma, beta = kling_gupta_efficiency(obs, sim)
    assert pytest.approx(kge, rel=1e-3) == 1.0
    assert pytest.approx(r, rel=1e-3) == 1.0
