# dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    t1c: Path
    t1n: Path
    t2f: Path
    t2w: Path
    seg: Path


def list_cases(pre_root: Path) -> List[CasePaths]:
    cases: List[CasePaths] = []
    for case_dir in sorted([p for p in pre_root.iterdir() if p.is_dir()]):
        seg = case_dir / "seg_atlas.nii.gz"
        t1c = case_dir / "t1c.nii.gz"
        t1n = case_dir / "t1n.nii.gz"
        t2f = case_dir / "t2f.nii.gz"
        t2w = case_dir / "t2w.nii.gz"
        if all(p.exists() for p in [seg, t1c, t1n, t2f, t2w]):
            cases.append(CasePaths(case_dir.name, t1c, t1n, t2f, t2w, seg))
    if not cases:
        raise FileNotFoundError(f"No preprocessed cases found under {pre_root}")
    return cases


def list_cases_raw(raw_root: Path) -> List[CasePaths]:
    cases: List[CasePaths] = []
    for case_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        case_id = case_dir.name

        seg = case_dir / f"{case_id}-seg.nii"
        t1c = case_dir / f"{case_id}-t1c.nii"
        t1n = case_dir / f"{case_id}-t1n.nii"
        t2f = case_dir / f"{case_id}-t2f.nii"
        t2w = case_dir / f"{case_id}-t2w.nii"

        if all(p.exists() for p in [seg, t1c, t1n, t2f, t2w]):
            cases.append(CasePaths(case_id, t1c, t1n, t2f, t2w, seg))

    if not cases:
        raise FileNotFoundError(f"No raw cases found under {raw_root}")
    return cases


def partition_cases(cases: List[CasePaths], num_clients: int, cid: int) -> List[CasePaths]:
    # Deterministic round-robin split
    return [c for i, c in enumerate(cases) if (i % num_clients) == cid]


def _load_nii(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def _zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    x: (C, D, H, W)
    z-score per channel using nonzero voxels (common for MRI).
    """
    out = x.copy()
    for c in range(out.shape[0]):
        chan = out[c]
        mask = chan != 0
        if mask.any():
            mu = chan[mask].mean()
            sd = chan[mask].std()
            out[c] = (chan - mu) / (sd + eps)
        else:
            out[c] = 0
    return out


def _random_crop_3d(
    x: np.ndarray, y: np.ndarray, crop: Tuple[int, int, int], rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    # x: (C, D, H, W), y: (D, H, W)
    _, D, H, W = x.shape
    cd, ch, cw = crop
    sd = rng.integers(0, max(1, D - cd + 1))
    sh = rng.integers(0, max(1, H - ch + 1))
    sw = rng.integers(0, max(1, W - cw + 1))
    return (
        x[:, sd : sd + cd, sh : sh + ch, sw : sw + cw],
        y[sd : sd + cd, sh : sh + ch, sw : sw + cw],
    )


def _tumor_biased_crop_3d(
    x: np.ndarray,
    y: np.ndarray,
    crop: Tuple[int, int, int],
    rng: np.random.Generator,
    tumor_prob: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    # x: (C, D, H, W), y: (D, H, W)
    _, D, H, W = x.shape
    cd, ch, cw = crop

    # pick a tumor voxel with prob tumor_prob
    if rng.random() < tumor_prob:
        tumor = np.argwhere(y > 0)  # WT-biased. Can switch to y==3 for ET-biased (after remap {0,2,3,4}->{0,1,2,3}).
        if tumor.size > 0:
            cz, cy, cx = tumor[rng.integers(0, len(tumor))]
            sd = int(np.clip(cz - cd // 2, 0, max(0, D - cd)))
            sh = int(np.clip(cy - ch // 2, 0, max(0, H - ch)))
            sw = int(np.clip(cx - cw // 2, 0, max(0, W - cw)))
            return (
                x[:, sd:sd+cd, sh:sh+ch, sw:sw+cw],
                y[sd:sd+cd, sh:sh+ch, sw:sw+cw],
            )

    # fallback: uniform random crop
    return _random_crop_3d(x, y, crop, rng)


class BratsPatchDataset(Dataset):
    """
    Returns random 3D patches from each case.
    Labels are remapped to contiguous integers (0..3).
    """
    def __init__(
        self,
        cases: List[CasePaths],
        patches_per_case: int = 4,
        crop: Tuple[int, int, int] = (96, 96, 96),
        seed: int = 0,
    ):
        self.cases = cases
        self.patches_per_case = patches_per_case
        self.crop = crop
        self.seed = seed

        # pre-load volumes for simplicity (ok for small datasets; later you can stream)
        self._x: List[np.ndarray] = []
        self._y: List[np.ndarray] = []
        for c in cases:
            t1c = _load_nii(c.t1c)
            t1n = _load_nii(c.t1n)
            t2f = _load_nii(c.t2f)
            t2w = _load_nii(c.t2w)
            seg = nib.load(str(c.seg)).get_fdata().astype(np.int16)

            # nibabel gives (X, Y, Z). We want (D, H, W) for Conv3D: (Z, Y, X)
            # We'll reorder consistently: (Z, Y, X)
            t1c = np.transpose(t1c, (2, 1, 0))
            t1n = np.transpose(t1n, (2, 1, 0))
            t2f = np.transpose(t2f, (2, 1, 0))
            t2w = np.transpose(t2w, (2, 1, 0))
            seg = np.transpose(seg, (2, 1, 0))

            # Remap {0,2,3,4} -> {0,1,2,3} (contiguous) without merging classes
            seg = seg.astype(np.int16)
            seg_remap = np.zeros_like(seg, dtype=np.int16)
            seg_remap[seg == 2] = 1
            seg_remap[seg == 3] = 2
            seg_remap[seg == 4] = 3
            seg = seg_remap

            # Sanity check (fail fast)
            u = np.unique(seg)
            if not set(u).issubset({0, 1, 2, 3}):
                raise ValueError(f"Unexpected labels after remap for {c.case_id}: {u}")

            x = np.stack([t1c, t1n, t2f, t2w], axis=0)  # (C, D, H, W)
            x = _zscore_per_channel(x)

            self._x.append(x.astype(np.float32))
            self._y.append(seg.astype(np.int16))

        self._len = len(self.cases) * self.patches_per_case

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        case_idx = idx // self.patches_per_case
        patch_idx = idx % self.patches_per_case

        rng = np.random.default_rng(self.seed + case_idx * 1000 + patch_idx)
        x = self._x[case_idx]
        y = self._y[case_idx]

        x_crop, y_crop = _tumor_biased_crop_3d(x, y, self.crop, rng, tumor_prob=0.7)

        # torch tensors
        x_t = torch.from_numpy(x_crop)  # (C, D, H, W)
        y_t = torch.from_numpy(y_crop).long()  # (D, H, W)
        return x_t, y_t
    
    
# full-volume dataset for evaluation
class BratsCaseDataset(Dataset):
    """Returns full volumes (x: (C,D,H,W), y: (D,H,W)) for case-level evaluation."""
    def __init__(self, cases: List[CasePaths]):
        self.cases = cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        c = self.cases[idx]

        t1c = _load_nii(c.t1c)
        t1n = _load_nii(c.t1n)
        t2f = _load_nii(c.t2f)
        t2w = _load_nii(c.t2w)
        seg = nib.load(str(c.seg)).get_fdata().astype(np.int16)

        # (X,Y,Z) -> (Z,Y,X) => (D,H,W)
        t1c = np.transpose(t1c, (2, 1, 0))
        t1n = np.transpose(t1n, (2, 1, 0))
        t2f = np.transpose(t2f, (2, 1, 0))
        t2w = np.transpose(t2w, (2, 1, 0))
        seg = np.transpose(seg, (2, 1, 0))

        seg = seg.astype(np.int16)
        seg_remap = np.zeros_like(seg, dtype=np.int16)
        seg_remap[seg == 2] = 1
        seg_remap[seg == 3] = 2
        seg_remap[seg == 4] = 3
        seg = seg_remap
        u = np.unique(seg)
        if not set(u).issubset({0, 1, 2, 3}):
            raise ValueError(f"Unexpected labels after remap for {c.case_id}: {u}")

        x = np.stack([t1c, t1n, t2f, t2w], axis=0).astype(np.float32)
        x = _zscore_per_channel(x)

        x_t = torch.from_numpy(x)          # (C,D,H,W)
        y_t = torch.from_numpy(seg).long() # (D,H,W)
        return x_t, y_t, c.case_id