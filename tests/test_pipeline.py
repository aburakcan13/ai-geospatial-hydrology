import pytest
import torch
import numpy as np
from src.models.unet_boundary import BoundaryAwareResUNet
from src.forecasting.lstm_model import HydroLSTMForecaster
from src.spectral.unmixing import linear_spectral_unmixing

def test_unet_forward_pass():
    model = BoundaryAwareResUNet(in_channels=8, num_classes=1)
    dummy_input = torch.randn(2, 8, 128, 128)
    water_mask, edge_mask = model(dummy_input)
    assert water_mask.shape == (2, 1, 128, 128)
    assert edge_mask.shape == (2, 1, 128, 128)

def test_lstm_forecaster_forward():
    model = HydroLSTMForecaster(in_features=6, hidden_dim=32, forecast_steps=12)
    dummy_seq = torch.randn(4, 24, 6) # (batch, seq_len, features)
    out = model(dummy_seq)
    assert out.shape == (4, 12)

def test_fcls_spectral_unmixing():
    endmembers = np.array([
        [0.02, 0.34], # SWIR1: pure water vs pure soil
        [0.05, 0.20]  # NIR
    ])
    # 50% mixture test
    pixel = np.array([0.18, 0.125])
    fractions = linear_spectral_unmixing(pixel, endmembers)
    assert pytest.approx(fractions.sum(), 0.01) == 1.0
    assert pytest.approx(fractions[0], 0.05) == 0.50
