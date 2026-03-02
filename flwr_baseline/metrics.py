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

    tc_pred = np.logical_or(pred == 3, pred == 4)
    tc_gt   = np.logical_or(gt == 3, gt == 4)

    et_pred = pred == 4
    et_gt   = gt == 4

    return {
        "dice_WT": dice_bool(wt_pred, wt_gt),
        "dice_TC": dice_bool(tc_pred, tc_gt),
        "dice_ET": dice_bool(et_pred, et_gt),
    }