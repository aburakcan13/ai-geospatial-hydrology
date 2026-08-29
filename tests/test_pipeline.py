import pytest
import torch
from src.models.unet_boundary import BoundaryAwareResUNet
from src.models.temporal_models import HydroLSTMForecaster
from src.metrics.hydrological_kge import calculate_kge

def test_unet_forward_pass():
    model = BoundaryAwareResUNet(in_channels=8, num_classes=1)
    dummy_input = torch.randn(2, 8, 128, 128)
    water_mask, edge_mask = model(dummy_input)
    assert water_mask.shape == (2, 1, 128, 128)
    assert edge_mask.shape == (2, 1, 128, 128)

def test_lstm_forecaster_forward():
    model = HydroLSTMForecaster(in_features=6, hidden_dim=32, forecast_steps=12)
    dummy_seq = torch.randn(4, 24, 6)
    out = model(dummy_seq)
    assert out.shape == (4, 12)

def test_kge_metric_calculation():
    obs = torch.tensor([10.0, 12.0, 15.0, 14.0, 11.0])
    sim = torch.tensor([9.8, 12.2, 14.7, 13.9, 11.2])
    kge_val = calculate_kge(sim, obs)
    assert kge_val > 0.85
