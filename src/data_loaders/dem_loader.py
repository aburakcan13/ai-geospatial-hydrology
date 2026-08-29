"""
Copernicus DEM (GLO-30) Data Ingestion & Topographic Index Extraction
AI-Driven Geospatial Hydrology (Springer Nature)
"""

from typing import Tuple, Optional
import numpy as np
import rasterio
import rioxarray
import xarray as xr


class DEMTerrainProcessor:
    def __init__(self, dem_path: str):
        """Initializes with path to Copernicus DEM raster tile (GeoTIFF/Cloud-Optimized GeoTIFF)."""
        self.dem_path = dem_path

    def load_clipped_dem(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> xr.DataArray:
        """
        Loads and optionally clips elevation raster.
        bbox: (minx, miny, maxx, maxy) in target CRS.
        """
        dem = rioxarray.open_rasterio(self.dem_path, masked=True)
        if bbox:
            dem = dem.rio.clip_box(*bbox)
        return dem

    @staticmethod
    def calculate_slope(elevation: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
        """Calculates topographic slope in degrees using 2D finite difference gradients."""
        dy, dx = np.gradient(elevation, cell_size)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        return np.degrees(slope_rad)

    @staticmethod
    def compute_topographic_wetness_index(slope_deg: np.ndarray, flow_accumulation: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
        """
        Computes Topographic Wetness Index: TWI = ln(a / tan(beta))
        where 'a' is specific catchment area and 'beta' is slope in radians.
        """
        slope_rad = np.radians(np.clip(slope_deg, 0.001, 89.9))
        specific_catchment_area = (flow_accumulation * cell_size) + 1.0
        twi = np.log(specific_catchment_area / np.tan(slope_rad))
        return twi
