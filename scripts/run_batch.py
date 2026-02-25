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
    """
    Preprocess modalities to BraTS atlas space AND warp seg.nii into the same space.
    Outputs:
      pre_case_dir/t1c.nii.gz, t1n.nii.gz, t2f.nii.gz, t2w.nii.gz
      pre_case_dir/seg_atlas.nii.gz   (GT warped to atlas space)
      pre_case_dir/transforms/...     (saved transforms)
    """
    from brainles_preprocessing.modality import Modality, CenterModality
    from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
    from brainles_preprocessing.transform import Transform
    from brainles_preprocessing.constants import Atlas

    pre_case_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir = pre_case_dir / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve raw inputs (your suffix matcher) ----
    t1c_in = find_by_suffix_nii(raw_case_dir, "t1c")
    t1n_in = find_by_suffix_nii(raw_case_dir, "t1n")
    t2f_in = find_by_suffix_nii(raw_case_dir, "t2f")
    t2w_in = find_by_suffix_nii(raw_case_dir, "t2w")
    seg_in = find_by_suffix_nii(raw_case_dir, "seg")

    # ---- define output paths (atlas space) ----
    t1c_out = pre_case_dir / "t1c.nii.gz"
    t1n_out = pre_case_dir / "t1n.nii.gz"
    t2f_out = pre_case_dir / "t2f.nii.gz"
    t2w_out = pre_case_dir / "t2w.nii.gz"

    # Center modality (t1c) + moving modalities
    center = CenterModality(
        modality_name="t1c",
        input_path=t1c_in,
        raw_bet_output_path=t1c_out,  # brain-extracted atlas-space output
    )
    moving = [
        Modality(modality_name="t1n", input_path=t1n_in, raw_bet_output_path=t1n_out),
        Modality(modality_name="t2f", input_path=t2f_in, raw_bet_output_path=t2f_out),
        Modality(modality_name="t2w", input_path=t2w_in, raw_bet_output_path=t2w_out),
    ]

    # Run atlas-centric preprocessing (BraTS-like atlas space)
    preprocessor = AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=moving,
        atlas_image_path=Atlas.BRATS_SRI24,  # BraTS-flavored SRI24 atlas :contentReference[oaicite:3]{index=3}
    )

    preprocessor.run(
        save_dir_transformations=transforms_dir,  # <--- this is crucial :contentReference[oaicite:4]{index=4}
    )

    # Warp GT segmentation into atlas space using the saved transforms
    seg_atlas_out = pre_case_dir / "seg_atlas.nii.gz"
    log_file = pre_case_dir / "transform_seg.log"

    tfm = Transform(transformations_dir=transforms_dir)
    tfm.apply(
        target_modality_name="t1c",
        target_modality_img=t1c_out,        # target grid: atlas-space t1c
        moving_image=seg_in,                # source: native-space GT seg
        output_img_path=seg_atlas_out,
        log_file_path=log_file,
        interpolator="genericLabel",        # best for label maps :contentReference[oaicite:5]{index=5}
        inverse=False,                      # native -> atlas
    )

def infer_case(pre_case_dir: Path, out_pred_path: Path, segmenter):
    out_pred_path.parent.mkdir(parents=True, exist_ok=True)

    segmenter.infer_single(
        t1c=str(pre_case_dir / "t1c.nii.gz"),
        t1n=str(pre_case_dir / "t1n.nii.gz"),
        t2f=str(pre_case_dir / "t2f.nii.gz"),
        t2w=str(pre_case_dir / "t2w.nii.gz"),
        output_file=str(out_pred_path),
        backend=Backends.DOCKER,
    )


# ----------------------------
# Batch runner
# ----------------------------
def main():
    raw_root = Path("data/brats_raw")
    pre_root = Path("data/brats_preprocessed")
    out_root = Path("data/brats_outputs")
    results_csv = Path("repro/baseline_metrics.csv")
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

            gt_path = pre_case_dir / "seg_atlas.nii.gz"

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