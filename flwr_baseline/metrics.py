# metrics.py
import numpy as np

def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2 * inter / denom) if denom > 0 else np.nan)  # <-- NaN, not 1.0

def brats_dice_regions(pred: np.ndarray, gt: np.ndarray) -> dict:
    wt_pred = pred > 0
    wt_gt   = gt > 0

    # TC = NCR/NET (1) + ET (3)
    tc_pred = np.logical_or(pred == 1, pred == 3)
    tc_gt   = np.logical_or(gt == 1, gt == 3)

    # ET = 3
    et_pred = pred == 3
    et_gt   = gt == 3

    return {
        "dice_WT": dice_bool(wt_pred, wt_gt),
        "dice_TC": dice_bool(tc_pred, tc_gt),
        "dice_ET": dice_bool(et_pred, et_gt),
    }