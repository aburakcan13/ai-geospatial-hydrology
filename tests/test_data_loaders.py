"""
Unit Tests for Data Loaders & Topographic Computations
AI-Driven Geospatial Hydrology (Springer Nature)
"""

import numpy as np
import pytest
from src.data_loaders.dem_loader import DEMTerrainProcessor


def test_slope_calculation():
    # 30m hücre boyutu ile yatayda sabit yükselen eğim düzlemi
    cell_size = 30.0
    elevation = np.array([
        [0.0, 30.0, 60.0],
        [0.0, 30.0, 60.0],
        [0.0, 30.0, 60.0]
    ])
    
    slope = DEMTerrainProcessor.calculate_slope(elevation, cell_size=cell_size)
    assert slope.shape == (3, 3)
    assert np.all(slope >= 0.0)


def test_twi_computation():
    slope_deg = np.array([[5.0, 10.0], [15.0, 20.0]])
    flow_acc = np.array([[100.0, 50.0], [25.0, 10.0]])
    
    twi = DEMTerrainProcessor.compute_topographic_wetness_index(slope_deg, flow_acc, cell_size=30.0)
    assert twi.shape == (2, 2)
    assert not np.isnan(twi).any()
