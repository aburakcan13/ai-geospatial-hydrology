from typing import Tuple
import torch
import numpy as np

def calculate_kge(simulated, observed) -> float:
    if isinstance(simulated, torch.Tensor):
        sim = simulated.detach().cpu().numpy()
    else:
        sim = np.asarray(simulated)
        
    if isinstance(observed, torch.Tensor):
        obs = observed.detach().cpu().numpy()
    else:
        obs = np.asarray(observed)
        
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-7)
    beta = np.mean(sim) / (np.mean(obs) + 1e-7)
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    return float(kge)

def kling_gupta_efficiency(observed: np.ndarray, simulated: np.ndarray) -> Tuple[float, float, float, float]:
    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = np.std(simulated) / (np.std(observed) + 1e-7)
    beta = np.mean(simulated) / (np.mean(observed) + 1e-7)
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    return float(kge), float(r), float(alpha), float(beta)

def nash_sutcliffe_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return float(1.0 - (numerator / (denominator + 1e-7)))

def calculate_iou_dice(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = float(intersection / (union + 1e-7))
    dice = float((2.0 * intersection) / (pred.sum() + target.sum() + 1e-7))
    return iou, dice
