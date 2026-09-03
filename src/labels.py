"""Annotation aggregation: tensor [T, C, A] -> segment-level soft/hard labels.

Mirrors the Task-4 strategy (per-annotator overlap fraction, mean over annotators for
soft labels, majority vote for hard labels) and the evaluator's majority-vote rule.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import feature_file_paths, load_npz


def soft_labels(annotations):
    """Mean over annotator axis: `annotations [T, C, A]` -> `soft [T, C]` in [0, 1]."""
    return np.asarray(annotations, dtype=np.float32).mean(axis=2)


def hard_labels(soft, threshold: float = 0.5):
    """Binarize soft labels via majority-vote threshold: `soft [T, C]` -> `hard [T, C]`."""
    return (np.asarray(soft, dtype=np.float32) >= threshold).astype(np.int8)


def majority_vote(annotations):
    """Per-segment majority vote matching the evaluator (>= half of annotators agree)."""
    return hard_labels(soft_labels(annotations), threshold=0.5)


def _labels_one_recording(path: Path) -> tuple[str, np.ndarray]:
    """Load hard labels `[T, C]` for one recording path."""
    data = load_npz(path)
    if "annotations" not in data:
        raise KeyError(f"No annotations in {path}")
    return path.stem, majority_vote(data["annotations"])


def stack_hard_labels(
    split_dir: str | Path,
    index=None,
    n_workers: int | None = None,
    cache_path: str | Path | None = None,
    force: bool = False,
) -> np.ndarray:
    """Stack hard labels for a split into `Y [N, C]` aligned to `index`.

    Parameters
    ----------
    split_dir:
        Split directory containing annotated `audio_features/*.npz` files.
    index:
        Optional DataFrame from `data_io.stack_feature_matrix`; if provided, output rows
        follow this exact recording/segment order.
    n_workers:
        Number of parallel loader threads. Defaults to `os.cpu_count()`.
    cache_path:
        Optional `.npy` label cache path.
    force:
        Recompute cached labels when true.

    Returns
    -------
    Y:
        Binary hard labels `[N, C]`.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            return np.load(cache_path)

    paths = feature_file_paths(split_dir)
    workers = n_workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = dict(executor.map(_labels_one_recording, paths))

    if index is None:
        Y = np.vstack([loaded[path.stem] for path in paths]).astype(np.int8, copy=False)
    else:
        idx = pd.DataFrame(index)
        chunks = []
        for recording_id, group in idx.groupby("recording_id", sort=False):
            labels = loaded[str(recording_id)]
            segment_idx = group["segment_idx"].to_numpy(dtype=int)
            chunks.append(labels[segment_idx])
        Y = np.vstack(chunks).astype(np.int8, copy=False)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, Y)
    return Y
