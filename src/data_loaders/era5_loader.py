"""
ECMWF ERA5 Climate Reanalysis Data Loader & NetCDF Preprocessor
AI-Driven Geospatial Hydrology (Springer Nature)
"""

from pathlib import Path
from typing import List, Tuple, Optional
import cdsapi
import xarray as xr
import numpy as np


class ERA5ClimateLoader:
    def __init__(self, output_dir: str = "data/sample_inputs"):
        """Initializes Copernicus CDS API Client."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = cdsapi.Client()

    def download_hourly_hydro_variables(
        self,
        year: str,
        month: str,
        days: List[str],
        bbox: Tuple[float, float, float, float],
        filename: str = "era5_reanalysis_sample.nc"
    ) -> Path:
        """
        Downloads total precipitation, 2m temperature, and surface runoff.
        bbox: (North, West, South, East) in decimal degrees.
        """
        target_file = self.output_dir / filename

        self.client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "2m_temperature",
                    "total_precipitation",
                    "surface_runoff"
                ],
                "year": year,
                "month": month,
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": list(bbox),
            },
            str(target_file)
        )
        return target_file

    @staticmethod
    def extract_time_series(nc_path: Path, lat: float, lon: float) -> xr.Dataset:
        """Extracts localized precipitation and runoff time series from NetCDF."""
        ds = xr.open_dataset(nc_path)
        point_data = ds.sel(latitude=lat, longitude=lon, method="nearest")
        return point_data
