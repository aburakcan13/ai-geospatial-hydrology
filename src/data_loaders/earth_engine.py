"""
Earth Engine Data Ingestion and Preprocessing Pipeline
AI-Driven Geospatial Hydrology (Springer Nature)
"""

from typing import List, Tuple, Optional
import ee
import geemap
import geopandas as gpd
import xarray as xr


class SentinelHydrologyLoader:
    def __init__(self, project_id: Optional[str] = None):
        """Initializes Google Earth Engine API."""
        try:
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project_id)

    def get_surface_water_collection(
        self,
        roi: ee.Geometry,
        start_date: str,
        end_date: str,
        cloud_percentage: float = 20.0
    ) -> ee.ImageCollection:
        """
        Extracts Sentinel-2 L2A Harmonized surface reflectance images
        filtered by ROI, cloud coverage, and calculated spectral indices.
        """
        def compute_indices(image: ee.Image) -> ee.Image:
            # MNDWI = (Green - SWIR1) / (Green + SWIR1) -> (B3 - B11) / (B3 + B11)
            mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
            # NDWI = (Green - NIR) / (Green + NIR) -> (B3 - B8) / (B3 + B8)
            ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
            # AWEIsh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
            awei = image.expression(
                "B2 + 2.5 * B3 - 1.5 * (B8 + B11) - 0.25 * B12",
                {
                    "B2": image.select("B2"),
                    "B3": image.select("B3"),
                    "B8": image.select("B8"),
                    "B11": image.select("B11"),
                    "B12": image.select("B12"),
                }
            ).rename("AWEIsh")
            return image.addBands([mndwi, ndwi, awei])

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_percentage))
            .map(compute_indices)
        )
        return collection

    @staticmethod
    def export_subpatch_to_geotiff(
        image: ee.Image,
        region: ee.Geometry,
        output_path: str,
        scale: int = 10,
        bands: Optional[List[str]] = None
    ) -> None:
        """Exports a processed multi-band sub-region to local GeoTIFF."""
        if bands is None:
            bands = ["B2", "B3", "B4", "B8", "B11", "MNDWI", "AWEIsh"]
        
        selected_img = image.select(bands)
        geemap.ee_export_image(
            selected_img,
            filename=output_path,
            scale=scale,
            region=region,
            file_per_band=False
        )
