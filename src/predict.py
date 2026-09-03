"""Inference pipeline: per-segment probabilities -> events -> submission CSV.

Produces the `filename,annotation,onset,offset` CSV required by the challenge. Applies
per-class thresholds (tuned on validation only) and optional post-processing, then writes
one row per detected event.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    from .config import EVAL_RESOLUTION_SECONDS
except ImportError:  # pragma: no cover - supports direct notebook execution fallbacks
    from config import EVAL_RESOLUTION_SECONDS


def _threshold_for_class(prob_col, y_col, grid) -> float:
    """Return threshold maximizing binary F1 for one class."""
    y_true = np.asarray(y_col).astype(int)
    probabilities = np.asarray(prob_col, dtype=np.float32)
    if y_true.sum() == 0:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in grid:
        y_pred = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _threshold_vector(thresholds, class_names: list[str]) -> np.ndarray:
    """Convert dict/list/scalar thresholds to vector `[C]`."""
    if isinstance(thresholds, dict):
        return np.asarray([thresholds[name] for name in class_names], dtype=np.float32)
    arr = np.asarray(thresholds, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(len(class_names), float(arr), dtype=np.float32)
    if arr.shape[0] != len(class_names):
        raise ValueError(f"Expected {len(class_names)} thresholds, got {arr.shape[0]}")
    return arr


def predictions_to_intervals(
    predictions: np.ndarray,
    start_times: np.ndarray,
    filename: str,
    class_names: list[str],
    segment_seconds: float = EVAL_RESOLUTION_SECONDS,
) -> list[dict]:
    """Convert binary predictions `[T, C]` into merged event rows.

    Consecutive active per-second segments for the same class are merged into one
    `filename,annotation,onset,offset` interval.
    """
    y_pred = np.asarray(predictions).astype(int)
    times = np.asarray(start_times, dtype=float)
    if y_pred.ndim != 2:
        raise ValueError(f"Expected predictions [T, C], got shape {y_pred.shape}")
    if y_pred.shape[0] != times.shape[0]:
        raise ValueError(f"Prediction/time mismatch: {y_pred.shape[0]} vs {times.shape[0]}")

    rows: list[dict] = []
    for class_idx, class_name in enumerate(class_names):
        in_event = False
        onset = None
        for time, active in zip(times, y_pred[:, class_idx]):
            if active and not in_event:
                onset = float(time)
                in_event = True
            elif not active and in_event:
                rows.append(
                    {
                        "filename": filename,
                        "annotation": class_name,
                        "onset": onset,
                        "offset": float(time),
                    }
                )
                in_event = False
        if in_event:
            rows.append(
                {
                    "filename": filename,
                    "annotation": class_name,
                    "onset": onset,
                    "offset": float(times[-1] + segment_seconds),
                }
            )
    return rows


def tune_thresholds(prob_val, Y_val, class_names: list[str]):
    """Select per-class thresholds maximizing per-class F1 on validation data.

    Parameters
    ----------
    prob_val:
        Validation probabilities `[N, C]`.
    Y_val:
        Validation labels `[N, C]`.
    class_names:
        Class names matching columns of `prob_val` and `Y_val`.

    Returns
    -------
    thresholds:
        Dict `{class_name: threshold}`. Never fit this on the hidden test split.
    """
    probabilities = np.asarray(prob_val, dtype=np.float32)
    labels = np.asarray(Y_val).astype(int)
    if probabilities.shape != labels.shape:
        raise ValueError(f"Probability/label shape mismatch: {probabilities.shape} vs {labels.shape}")
    if probabilities.shape[1] != len(class_names):
        raise ValueError(f"Expected {len(class_names)} classes, got {probabilities.shape[1]}")

    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    return {
        class_name: _threshold_for_class(probabilities[:, idx], labels[:, idx], grid)
        for idx, class_name in enumerate(class_names)
    }


def apply_thresholds(prob, thresholds, class_names: list[str]):
    """Binarize probabilities `[T, C]` with per-class thresholds -> `[T, C]` {0,1}."""
    probabilities = np.asarray(prob, dtype=np.float32)
    threshold_values = _threshold_vector(thresholds, class_names)
    if probabilities.ndim != 2:
        raise ValueError(f"Expected probabilities [T, C], got shape {probabilities.shape}")
    if probabilities.shape[1] != len(class_names):
        raise ValueError(f"Expected {len(class_names)} classes, got {probabilities.shape[1]}")
    return (probabilities >= threshold_values[np.newaxis, :]).astype(np.int8)


def build_submission(prob_by_file, thresholds, class_names: list[str]):
    """Turn per-file per-second probabilities into a submission DataFrame.

    Parameters
    ----------
    prob_by_file:
        Mapping `filename -> probabilities [T, C]` or
        `filename -> {"probabilities": [T, C], "start_times": [T]}`.
    thresholds:
        Per-class thresholds as dict/list/scalar.
    class_names:
        Class names matching probability columns.

    Returns
    -------
    df:
        Submission rows with columns `filename,annotation,onset,offset`.
    """
    rows: list[dict] = []
    for filename, payload in prob_by_file.items():
        if isinstance(payload, dict):
            probabilities = payload.get("probabilities", payload.get("prob"))
            start_times = payload.get("start_times")
        else:
            probabilities = payload
            start_times = None

        probabilities = np.asarray(probabilities, dtype=np.float32)
        if start_times is None:
            start_times = np.arange(probabilities.shape[0], dtype=float) * EVAL_RESOLUTION_SECONDS
        predictions = apply_thresholds(probabilities, thresholds, class_names)
        rows.extend(predictions_to_intervals(predictions, start_times, filename, class_names))

    columns = ["filename", "annotation", "onset", "offset"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def write_submission_csv(df, path: str | Path):
    """Write the submission DataFrame to CSV in the required format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["filename", "annotation", "onset", "offset"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Submission DataFrame missing columns: {missing}")
    df.loc[:, columns].to_csv(output_path, index=False)
    return output_path
