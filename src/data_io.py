"""Loading of challenge data: .npz feature files, metadata.csv, annotations.csv.

Builds per-recording arrays and the stacked feature matrix used by the classifiers.
All array shapes are documented; T = segments, F = features (960), C = classes,
A = annotators.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_feature_matrix, default_feature_keys


def load_npz(path: str | Path) -> dict:
    """Load one feature/annotation `.npz` file into a dict of numpy arrays.

    Returns the raw dict including feature keys (`*_mean/std/min/max`),
    `annotations [T, C, A]`, `start_time [T]`, `end_time [T]`, `class_names`, etc.
    """
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_metadata(split_dir: str | Path):
    """Load `metadata.csv` for a split into a DataFrame (None for the test split)."""
    path = Path(split_dir) / "metadata.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_annotations_csv(split_dir: str | Path):
    """Load `annotations.csv` for a split into a DataFrame, or None when absent."""
    path = Path(split_dir) / "annotations.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def feature_file_paths(split_dir: str | Path) -> list[Path]:
    """Return sorted `.npz` feature paths for one split directory."""
    return sorted((Path(split_dir) / "audio_features").glob("*.npz"))


def list_recordings(split_dir: str | Path) -> list[str]:
    """Return the filename ids present in `<split_dir>/audio_features/`."""
    return [path.stem for path in feature_file_paths(split_dir)]


def _stack_one_recording(path: Path, feature_keys: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Load one recording into `X [T, F]` plus segment index rows."""
    data = load_npz(path)
    X = build_feature_matrix(data, feature_keys)
    start_time = np.asarray(data.get("start_time", np.arange(X.shape[0]) * 0.5), dtype=float)
    end_time = np.asarray(data.get("end_time", start_time + 1.0), dtype=float)
    index = pd.DataFrame(
        {
            "recording_id": path.stem,
            "filename": f"{path.stem}.wav",
            "npz_path": str(path),
            "segment_idx": np.arange(X.shape[0], dtype=int),
            "start_time": start_time,
            "end_time": end_time,
        }
    )
    return X, index


def stack_feature_matrix(
    feature_keys: list[str] | None,
    split_dir: str | Path,
    n_workers: int | None = None,
    cache_path: str | Path | None = None,
    force: bool = False,
):
    """Stack selected features across recordings into `X [N, F]` plus index.

    Parameters
    ----------
    feature_keys:
        `.npz` feature keys to stack. `None` uses the 960-dimensional default set.
    split_dir:
        Split directory containing `audio_features/*.npz`.
    n_workers:
        Number of parallel loader threads. Defaults to `os.cpu_count()`.
    cache_path:
        Optional `.npz` path for `X` and index caching.
    force:
        Recompute cached data when true.

    Returns
    -------
    X, index:
        `X [N, F]` float32 feature matrix and a DataFrame with recording/segment metadata.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            cached = np.load(cache_path, allow_pickle=True)
            return cached["X"], pd.DataFrame(cached["index"].item())

    keys = default_feature_keys() if feature_keys is None else list(feature_keys)
    paths = feature_file_paths(split_dir)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {Path(split_dir) / 'audio_features'}")

    workers = n_workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = list(executor.map(lambda path: _stack_one_recording(path, keys), paths))

    X = np.vstack([item[0] for item in loaded]).astype(np.float32, copy=False)
    index = pd.concat([item[1] for item in loaded], ignore_index=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, index=index.to_dict(orient="list"))
    return X, index


def unstack_probabilities(probabilities, index) -> dict:
    """Map stacked probabilities `[N, C]` back to per-file arrays.

    Returns `{filename: {"probabilities": [T, C], "start_times": [T]}}` preserving the
    segment order stored in `index`.
    """
    prob = np.asarray(probabilities, dtype=np.float32)
    idx = pd.DataFrame(index).reset_index(drop=True)
    if len(idx) != prob.shape[0]:
        raise ValueError(f"Index/probability row mismatch: {len(idx)} vs {prob.shape[0]}")

    result = {}
    for filename, group in idx.groupby("filename", sort=True):
        order = group.sort_values("segment_idx").index.to_numpy()
        result[filename] = {
            "probabilities": prob[order],
            "start_times": group.loc[order, "start_time"].to_numpy(dtype=float),
        }
    return result
