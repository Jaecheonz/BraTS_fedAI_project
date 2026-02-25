# scripts/batch_run_brats.py
from __future__ import annotations

import sys
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


# ---------------------------------------------------------------------
# Windows workaround:
# brats imports Singularity client (spython) which imports Unix-only `pwd`.
# We stub `pwd` so the import chain doesn't crash on Windows.
# This is fine as long as you use Backends.DOCKER (not Singularity).
# ---------------------------------------------------------------------
if sys.platform.startswith("win"):
    import types
    if "pwd" not in sys.modules:
        sys.modules["pwd"] = types.ModuleType("pwd")


RAW_ROOT = Path("data/brats_raw")
PREP_ROOT = Path("data/brats_preprocessed")
OUT_ROOT = Path("data/brats_outputs")
CSV_PATH = Path("repro/baseline_metrics.csv")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom > 0 else 1.0


def find_one(case_dir: Path, patterns: list[str]) -> Path:
    """Find exactly one file matching any of the patterns (glob)."""
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(case_dir.glob(pat))
    matches = [m for m in matches if m.is_file()]

    if len(matches) == 0:
        raise FileNotFoundError(f"No match in {case_dir} for patterns: {patterns}")

    if len(matches) > 1:
        # If multiple, prefer .nii.gz over .nii
        gz = [m for m in matches if m.name.endswith(".nii.gz")]
        if len(gz) == 1:
            return gz[0]
        raise RuntimeError(f"Multiple matches in {case_dir} for {patterns}: {matches}")

    return matches[0]


def map_modalities(case_dir: Path) -> dict[str, Path]:
    # Dataset naming: *-t1c.nii, *-t1n.nii, *-t2w.nii, *-t2f.nii, *-seg.nii
    t1n = find_one(case_dir, ["*t1n.nii", "*t1n.nii.gz"])
    t1c = find_one(case_dir, ["*t1c.nii", "*t1c.nii.gz", "*t1ce.nii", "*t1ce.nii.gz"])
    t2w = find_one(case_dir, ["*t2w.nii", "*t2w.nii.gz", "*t2.nii", "*t2.nii.gz"])
    t2f = find_one(case_dir, ["*t2f.nii", "*t2f.nii.gz", "*flair.nii", "*flair.nii.gz"])

    seg = None
    try:
        seg = find_one(case_dir, ["*seg.nii", "*seg.nii.gz"])
    except FileNotFoundError:
        pass

    d = {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}
    if seg is not None:
        d["seg"] = seg
    return d


def preprocess_case(case_id: str, files: dict[str, Path]) -> Path:
    """
    Lazy-import preprocess function so the script doesn't crash at import-time
    on Windows if brats pulls Singularity deps.
    """
    from brats.preprocessing import preprocess_coreg_sri24reg_bet

    out_dir = PREP_ROOT / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use .nii.gz outputs for consistency
    t1o = out_dir / "t1n.nii.gz"
    t1co = out_dir / "t1c.nii.gz"
    t2o = out_dir / "t2w.nii.gz"
    flairo = out_dir / "t2f.nii.gz"

    # Idempotency: skip if already there
    if all(p.exists() for p in [t1o, t1co, t2o, flairo]):
        return out_dir

    preprocess_coreg_sri24reg_bet(
        t1_input=files["t1n"],
        t1c_input=files["t1c"],
        t2_input=files["t2w"],
        flair_input=files["t2f"],
        t1_output=t1o,
        t1c_output=t1co,
        t2_output=t2o,
        flair_output=flairo,
    )
    return out_dir


def get_segmenter():
    """
    Create the segmenter with lazy imports.
    Keeping imports here avoids triggering `brats/__init__.py` at file import time.
    """
    from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms
    # Importing from brats root (`from brats import ...`) can trigger __init__.
    # We still try it, but with the pwd stub above this should no longer crash.
    from brats import AdultGliomaPreAndPostTreatmentSegmenter

    return AdultGliomaPreAndPostTreatmentSegmenter(
        algorithm=AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_2
    )


def infer_case(case_id: str, prep_dir: Path, segmenter) -> Path:
    from brats.constants import Backends  # lazy import

    out_dir = OUT_ROOT / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "segmentation.nii.gz"

    if out_file.exists():
        return out_file

    segmenter.infer_single(
        t1c=str(prep_dir / "t1c.nii.gz"),
        t1n=str(prep_dir / "t1n.nii.gz"),
        t2f=str(prep_dir / "t2f.nii.gz"),
        t2w=str(prep_dir / "t2w.nii.gz"),
        output_file=str(out_file),
        backend=Backends.DOCKER,
    )
    return out_file


def eval_case(pred_path: Path, gt_path: Path) -> tuple[float, float, float]:
    pred_img = nib.load(str(pred_path))
    gt_img = nib.load(str(gt_path))

    # First attempt: GT -> Pred
    gt_rs = resample_from_to(gt_img, pred_img, order=0)
    gt = gt_rs.get_fdata().astype(np.int16)
    pred = pred_img.get_fdata().astype(np.int16)

    # If GT vanished, try Pred -> GT
    if (gt > 0).sum() == 0 and (gt_img.get_fdata() > 0).sum() > 0:
        pred_rs = resample_from_to(pred_img, gt_img, order=0)
        pred = pred_rs.get_fdata().astype(np.int16)
        gt = gt_img.get_fdata().astype(np.int16)

    wt_pred = pred > 0
    wt_gt = gt > 0
    tc_pred = np.logical_or(pred == 1, pred == 4)
    tc_gt = np.logical_or(gt == 1, gt == 4)
    et_pred = pred == 4
    et_gt = gt == 4

    return dice(wt_pred, wt_gt), dice(tc_pred, tc_gt), dice(et_pred, et_gt)


def ensure_csv_header():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "dice_wt", "dice_tc", "dice_et", "pred_path", "gt_path", "error"])


def already_done(case_id: str) -> bool:
    if not CSV_PATH.exists():
        return False
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("case_id") == case_id:
                return True
    return False


def append_row(case_id: str, dwt, dtc, det, pred_path: str, gt_path: str, error: str):
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([case_id, dwt, dtc, det, pred_path, gt_path, error])


def main():
    ensure_csv_header()

    case_dirs = sorted([p for p in RAW_ROOT.iterdir() if p.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case folders found under {RAW_ROOT.resolve()}")

    # Build segmenter once
    try:
        segmenter = get_segmenter()
    except Exception as e:
        raise RuntimeError(
            "Failed to create segmenter. This usually means brats import/backends are broken in this env."
        ) from e

    for case_dir in case_dirs:
        case_id = case_dir.name

        if already_done(case_id):
            print(f"[SKIP] {case_id} already in CSV")
            continue

        dwt = dtc = det = float("nan")
        pred_path_str = ""
        gt_path_str = ""
        error_msg = ""

        try:
            files = map_modalities(case_dir)
            gt_path_str = str(files.get("seg", ""))

            prep_dir = preprocess_case(case_id, files)
            pred_path = infer_case(case_id, prep_dir, segmenter)
            pred_path_str = str(pred_path)

            if "seg" in files:
                dwt, dtc, det = eval_case(pred_path, files["seg"])

            print(f"[OK] {case_id} WT={dwt:.3f} TC={dtc:.3f} ET={det:.3f}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {case_id}: {error_msg}")

        append_row(case_id, dwt, dtc, det, pred_path_str, gt_path_str, error_msg)

    print("Done. CSV:", CSV_PATH)


if __name__ == "__main__":
    main()