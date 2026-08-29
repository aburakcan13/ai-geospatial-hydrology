"""
Hydrological & Semantic Segmentation Evaluation Metrics
AI-Driven Geospatial Hydrology (Springer Nature)
"""

import numpy as np
import torch


def calculate_iou_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6):
    """Calculates Intersection over Union (IoU) and Dice coefficient for binary masks."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (pred_bin.sum() + target_bin.sum() + smooth)

    return iou.item(), dice.item()


def kling_gupta_efficiency(observed: np.ndarray, simulated: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes Kling-Gupta Efficiency (KGE - Gupta et al., 2009)
    along with its three components: correlation (r), variability ratio (gamma), and bias ratio (beta).
    """
    if len(observed) != len(simulated):
        raise ValueError("Observed and simulated series must have the same length.")

    # Correlation coefficient (r)
    r = np.corrcoef(observed, simulated)[0, 1]

    # Variability ratio (gamma) = (std_sim / mean_sim) / (std_obs / mean_obs)
    gamma = (np.std(simulated) / np.mean(simulated)) / (np.std(observed) / np.mean(observed))

    # Bias ratio (beta) = mean_sim / mean_obs
    beta = np.mean(simulated) / np.mean(observed)

    # KGE formula
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2)

    return float(kge), float(r), float(gamma), float(beta)


def nash_sutcliffe_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Computes Nash-Sutcliffe Efficiency (NSE)."""
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1.0 - (numerator / denominator)
    return float(nse)
