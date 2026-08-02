"""Loader for the Middlebury optical-flow benchmark data used for speed benchmarking.

Data layout (created by downloading from https://vision.middlebury.edu/flow/data/):
    data/other-data/<Name>/frame10.png, frame11.png   -- two-frame color image pairs
    data/other-gt-flow/<Name>/flow10.flo              -- ground-truth flow (subset only)
"""

from __future__ import annotations

import os

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PAIRS_DIR = os.path.join(DATA_DIR, "other-data")
GT_DIR = os.path.join(DATA_DIR, "other-gt-flow")

_FLO_MAGIC = 202021.25  # Middlebury .flo sanity-check tag


def load_pairs(
    max_pairs: int | None = None,
    gray: bool = True,
    size: tuple[int, int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Load (frame1, frame2, name) tuples of uint8 arrays.

    Args:
        max_pairs: cap on number of pairs (None = all, sorted by name).
        gray: if True return single-channel (H, W); else BGR (H, W, 3).
        size: optional (H, W) to resize frames to.
    """
    if not os.path.isdir(PAIRS_DIR):
        raise FileNotFoundError(
            f"{PAIRS_DIR} not found; download the Middlebury 'other-color-twoframes' set first."
        )
    pairs: list[tuple[np.ndarray, np.ndarray, str]] = []
    for name in sorted(os.listdir(PAIRS_DIR)):
        seq_dir = os.path.join(PAIRS_DIR, name)
        f1_path = os.path.join(seq_dir, "frame10.png")
        f2_path = os.path.join(seq_dir, "frame11.png")
        if not (os.path.isfile(f1_path) and os.path.isfile(f2_path)):
            continue
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        f1 = cv2.imread(f1_path, flag)
        f2 = cv2.imread(f2_path, flag)
        if f1 is None or f2 is None:
            continue
        if size is not None:
            h, w = size
            f1 = cv2.resize(f1, (w, h), interpolation=cv2.INTER_AREA)
            f2 = cv2.resize(f2, (w, h), interpolation=cv2.INTER_AREA)
        pairs.append((f1, f2, name))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def load_flow_gt(name: str) -> np.ndarray | None:
    """Read the Middlebury ground-truth flow for a sequence, or None if absent.

    Returns an (H, W, 2) float32 array of (u, v) displacements. Occluded /
    unknown pixels are marked with values > 1e9 in the raw files.
    """
    path = os.path.join(GT_DIR, name, "flow10.flo")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic.size == 0 or abs(magic[0] - _FLO_MAGIC) > 1e-3:
            raise ValueError(f"{path}: bad .flo magic {magic!r}")
        w, h = np.fromfile(f, np.int32, count=2)
        data = np.fromfile(f, np.float32, count=int(2 * w * h))
    return data.reshape(int(h), int(w), 2)


if __name__ == "__main__":
    pairs = load_pairs(gray=False)
    print(f"Middlebury 'other' two-frame dataset: {len(pairs)} pairs\n")
    print(f"{'name':<12} {'resolution (HxW)':<18} {'GT flow':<10}")
    for f1, f2, name in pairs:
        gt = load_flow_gt(name)
        res = f"{f1.shape[0]}x{f1.shape[1]}"
        gt_str = "-"
        if gt is not None:
            gt_str = f"yes {gt.shape[0]}x{gt.shape[1]}"
        print(f"{name:<12} {res:<18} {gt_str:<10}")
