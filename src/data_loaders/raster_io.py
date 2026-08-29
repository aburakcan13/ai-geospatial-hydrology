"""
GeoTIFF Raster I/O, Patch Extraction, and Windowed Reading
AI-Driven Geospatial Hydrology (Springer Nature)
"""

from typing import Generator, Tuple, List
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
from torch.utils.data import Dataset


class GeoTIFFPatchExtractor:
    def __init__(self, raster_path: str, patch_size: int = 256, stride: int = 224):
        self.raster_path = raster_path
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self) -> Generator[Tuple[np.ndarray, Window], None, None]:
        """Yields sliding window array patches with spatial metadata."""
        with rasterio.open(self.raster_path) as src:
            width, height = src.width, src.height

            for y in range(0, height - self.patch_size + 1, self.stride):
                for x in range(0, width - self.patch_size + 1, self.stride):
                    window = Window(x, y, self.patch_size, self.patch_size)
                    patch = src.read(window=window)
                    # NaN temizleme ve normalizasyon
                    patch = np.nan_to_num(patch, nan=0.0)
                    yield patch, window


class HydrologyDataset(Dataset):
    def __init__(self, patches: List[np.ndarray], masks: List[np.ndarray]):
        self.patches = patches
        self.masks = masks

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.patches[idx]).float()
        y = torch.from_numpy(self.masks[idx]).float().unsqueeze(0)
        return x, y
