"""
Interactive Map Overlays & Delineation Visualization
AI-Driven Geospatial Hydrology (Springer Nature)
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_water_extraction_comparison(
    rgb_image: np.ndarray,
    ground_truth: np.ndarray,
    predicted_mask: np.ndarray,
    boundary_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> None:
    """
    Renders side-by-side comparison of RGB composite, ground truth mask,
    predicted water body, and predicted boundary delineations.
    """
    num_plots = 4 if boundary_mask is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    # 1. RGB Composite
    axes[0].imshow(rgb_image)
    axes[0].set_title("Sentinel-2 True Color (RGB)", fontsize=11)
    axes[0].axis("off")

    # 2. Ground Truth
    axes[1].imshow(ground_truth, cmap="Blues", interpolation="nearest")
    axes[1].set_title("Ground Truth Water Mask", fontsize=11)
    axes[1].axis("off")

    # 3. Model Prediction
    axes[2].imshow(predicted_mask, cmap="Blues", interpolation="nearest")
    axes[2].set_title("Boundary-Aware Prediction", fontsize=11)
    axes[2].axis("off")

    # 4. Boundary Head Output (Optional)
    if boundary_mask is not None:
        axes[3].imshow(boundary_mask, cmap="magma", interpolation="nearest")
        axes[3].set_title("Extracted Shoreline Boundary", fontsize=11)
        axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
