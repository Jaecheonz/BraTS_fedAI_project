from pathlib import Path
import csv
import numpy as np
import nibabel as nib

from brats.preprocessing import preprocess_coreg_sri24reg_bet
from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms, Backends


# ----------------------------
# Suffix-based file resolver (.nii only)
# ----------------------------
SUFFIXES_NII = {
    "t1c": "t1c.nii",
    "t1n": "t1n.nii",
    "t2f": "t2f.nii",
    "t2w": "t2w.nii",
    "seg": "seg.nii",
}

def _pick_best(candidates):
    """
    If multiple files end with the same suffix, pick the most likely one.
    Heuristic: shortest filename (least extra prefix), tie-break lexicographically.
    """
    return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]

def find_by_suffix_nii(case_dir: Path, key: str) -> Path:
    """
    Find a file under case_dir (recursively) whose name ends with '<suffix>.nii'.
    Only supports .nii (not .nii.gz).
    """
    if key not in SUFFIXES_NII:
        raise KeyError(f"Unknown key: {key}")

    suffix = SUFFIXES_NII[key]
    candidates = [p for p in case_dir.rglob("*") if p.is_file() and p.name.endswith(suffix)]

    if not candidates:
        raise FileNotFoundError(f"No file found for '{key}' in {case_dir} ending with '{suffix}'")

    # If you want to fail on duplicates instead of guessing, uncomment:
    # if len(candidates) > 1:
    #     raise RuntimeError(f"Multiple candidates for {key} in {case_dir}: {[c.name for c in candidates]}")

    return _pick_best(candidates)


# ----------------------------
# Metrics
# ----------------------------
def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2 * inter / denom) if denom > 0 else 1.0

def compute_brats_dice(pred_path: Path, gt_path: Path):
    pred_nii = nib.load(str(pred_path))
    gt_nii   = nib.load(str(gt_path))

    pred = pred_nii.get_fdata().astype(np.int16)
    gt   = gt_nii.get_fdata().astype(np.int16)

    # Basic sanity checks (shape mismatch = invalid dice)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    # BraTS labels: 1=NET/NCR, 2=ED, 4=ET
    wt_pred = pred > 0
    wt_gt   = gt > 0

    tc_pred = np.logical_or(pred == 1, pred == 4)
    tc_gt   = np.logical_or(gt == 1, gt == 4)

    et_pred = pred == 4
    et_gt   = gt == 4

    return {
        "WT": float(dice(wt_pred, wt_gt)),
        "TC": float(dice(tc_pred, tc_gt)),
        "ET": float(dice(et_pred, et_gt)),
        "pred_labels": np.unique(pred).tolist(),
        "gt_labels": np.unique(gt).tolist(),
        "gt_ET_vox": int((gt == 4).sum()),
        "pred_ET_vox": int((pred == 4).sum()),
    }


# ----------------------------
# Pipeline steps per case
# ----------------------------
def preprocess_case(raw_case_dir: Path, pre_case_dir: Path):
    pre_case_dir.mkdir(parents=True, exist_ok=True)

    # Resolve raw inputs by suffix (prefix can be anything)
    t1n_in = find_by_suffix_nii(raw_case_dir, "t1n")
    t1c_in = find_by_suffix_nii(raw_case_dir, "t1c")
    t2w_in = find_by_suffix_nii(raw_case_dir, "t2w")
    t2f_in = find_by_suffix_nii(raw_case_dir, "t2f")

    preprocess_coreg_sri24reg_bet(
        t1_input=t1n_in,
        t1c_input=t1c_in,
        t2_input=t2w_in,
        flair_input=t2f_in,
        # write consistent names in preprocessed dir (.nii only)
        t1_output=pre_case_dir / "t1n.nii",
        t1c_output=pre_case_dir / "t1c.nii",
        t2_output=pre_case_dir / "t2w.nii",
        flair_output=pre_case_dir / "t2f.nii",
    )

def infer_case(pre_case_dir: Path, out_pred_path: Path, segmenter):
    out_pred_path.parent.mkdir(parents=True, exist_ok=True)

    segmenter.infer_single(
        t1c=str(pre_case_dir / "t1c.nii"),
        t1n=str(pre_case_dir / "t1n.nii"),
        t2f=str(pre_case_dir / "t2f.nii"),
        t2w=str(pre_case_dir / "t2w.nii"),
        output_file=str(out_pred_path),
        backend=Backends.DOCKER,
    )


# ----------------------------
# Batch runner
# ----------------------------
def main():
    raw_root = Path("data/raw_cases")
    pre_root = Path("data/preprocessed_cases")
    out_root = Path("outputs")
    results_csv = Path("results/dice_summary.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    # init segmenter once (reused for all cases)
    segmenter = AdultGliomaPreAndPostTreatmentSegmenter(
        algorithm=AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_2,
    )

    case_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case folders found under: {raw_root}")

    rows = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        print(f"\n=== Case: {case_id} ===")

        try:
            pre_case_dir = pre_root / case_id
            pred_path = out_root / case_id / "segmentation.nii.gz"

            # Resolve GT by suffix too (prefix can be anything)
            gt_path = find_by_suffix_nii(case_dir, "seg")

            # 1) preprocess
            print("Preprocessing...")
            preprocess_case(case_dir, pre_case_dir)

            # 2) inference
            print("Inference...")
            infer_case(pre_case_dir, pred_path, segmenter)

            # 3) eval
            print("Evaluating Dice...")
            scores = compute_brats_dice(pred_path, gt_path)

            row = {
                "case": case_id,
                "dice_WT": scores["WT"],
                "dice_TC": scores["TC"],
                "dice_ET": scores["ET"],
                "gt_ET_vox": scores["gt_ET_vox"],
                "pred_ET_vox": scores["pred_ET_vox"],
                "gt_labels": str(scores["gt_labels"]),
                "pred_labels": str(scores["pred_labels"]),
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
            }
            rows.append(row)

            print(f"WT={row['dice_WT']:.3f}  TC={row['dice_TC']:.3f}  ET={row['dice_ET']:.3f}")

        except Exception as e:
            # don’t kill the whole batch; record failure
            print(f"FAILED {case_id}: {e}")
            rows.append({
                "case": case_id,
                "dice_WT": "",
                "dice_TC": "",
                "dice_ET": "",
                "gt_ET_vox": "",
                "pred_ET_vox": "",
                "gt_labels": "",
                "pred_labels": "",
                "pred_path": "",
                "gt_path": "",
                "error": str(e),
            })

    # write csv summary
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with results_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote summary: {results_csv}")
    print(f"Processed {len(rows)} cases.")

if __name__ == "__main__":
    main()

    # py -3.11 -m poetry run python scripts\run_batch.py