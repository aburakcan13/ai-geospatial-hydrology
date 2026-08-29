"""
Unit Tests for Deep Learning Architectures and Loss Pipelines
AI-Driven Geospatial Hydrology (Springer Nature)
"""

import pytest
import torch
import numpy as np
from src.models.unet_boundary import BoundaryAwareResUNet
from src.models.vision_transformers import GeospatialSegFormer
from src.models.temporal_models import HydroConvLSTM
from src.metrics.hydrological_kge import calculate_iou_dice, kling_gupta_efficiency, nash_sutcliffe_efficiency


def test_boundary_aware_unet_forward():
    batch_size = 2
    in_channels = 8
    height, width = 128, 128
    dummy_input = torch.randn(batch_size, in_channels, height, width)

    model = BoundaryAwareResUNet(in_channels=in_channels, num_classes=1)
    water_out, boundary_out = model(dummy_input)

    assert water_out.shape == (batch_size, 1, height, width)
    assert boundary_out.shape == (batch_size, 1, height, width)


def test_geospatial_segformer_forward():
    batch_size = 2
    in_channels = 8
    height, width = 128, 128
    dummy_input = torch.randn(batch_size, in_channels, height, width)

    model = GeospatialSegFormer(in_channels=in_channels, num_classes=1, embed_dim=64)
    pred_water = model(dummy_input)

    assert pred_water.shape == (batch_size, 1, height, width)


def test_hydro_convlstm_forward():
    timesteps = 4
    batch_size = 2
    in_channels = 4
    height, width = 64, 64
    dummy_seq = torch.randn(timesteps, batch_size, in_channels, height, width)

    model = HydroConvLSTM(in_channels=in_channels, hidden_channels=16)
    pred_map = model(dummy_seq)

    assert pred_map.shape == (batch_size, 1, height, width)


def test_kge_and_nse_metrics():
    obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    sim = np.array([10.5, 19.8, 30.2, 39.5, 50.1])

    kge, r, gamma, beta = kling_gupta_efficiency(obs, sim)
    nse = nash_sutcliffe_efficiency(obs, sim)

    assert kge > 0.95
    assert nse > 0.95
