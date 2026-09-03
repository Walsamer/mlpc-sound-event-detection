"""Samuel's post-processing variant helpers.

This module deliberately does not replace Andreas's ``postprocess.py``.  It provides
small, validation-driven experiments for Samuel's notebook so we can test variants
that still fit the Task 5 PDF wording: median filtering and temporal smoothing.

All functions operate on flattened segment arrays plus the segment index produced by
``data_io.stack_feature_matrix``.  The index is used to ensure that filtering never
crosses recording boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _index_frame(index) -> pd.DataFrame:
    """Return a normalized index frame with row positions and recording ids."""
    frame = pd.DataFrame(index).reset_index(drop=True).copy()
    if "recording_id" not in frame.columns:
        if "filename" not in frame.columns:
            raise KeyError("index must contain either 'recording_id' or 'filename'")
        frame["recording_id"] = frame["filename"].astype(str).str.replace(".wav", "", regex=False)
    if "segment_idx" not in frame.columns:
        frame["segment_idx"] = np.arange(len(frame), dtype=int)
    frame["_row_pos"] = np.arange(len(frame), dtype=int)
    return frame


def apply_per_recording(values, index, transform: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply ``transform`` independently to each recording sequence.

    Parameters
    ----------
    values:
        Flat array ``[N, C]`` containing probabilities or binary predictions.
    index:
        Segment index aligned with ``values``. Must contain a recording identifier.
    transform:
        Function receiving one recording's ``[T, C]`` sequence and returning a
        transformed sequence with the same shape.
    """
    array = np.asarray(values)
    out = np.empty_like(array)
    frame = _index_frame(index)

    for _, group in frame.sort_values(["recording_id", "segment_idx"]).groupby("recording_id", sort=False):
        positions = group["_row_pos"].to_numpy(dtype=int)
        transformed = transform(array[positions])
        if transformed.shape != array[positions].shape:
            raise ValueError(f"transform changed sequence shape from {array[positions].shape} to {transformed.shape}")
        out[positions] = transformed
    return out


def median_filter_sequence(pred_seq: np.ndarray, window: int = 3) -> np.ndarray:
    """Median-filter a binary ``[T, C]`` sequence without crossing file boundaries."""
    if window <= 1:
        return np.asarray(pred_seq).copy()
    if window % 2 == 0:
        raise ValueError("median filter window must be odd")

    seq = np.asarray(pred_seq).astype(np.int8)
    radius = window // 2
    padded = np.pad(seq, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(seq)
    for t in range(seq.shape[0]):
        out[t] = np.median(padded[t : t + window], axis=0) >= 0.5
    return out.astype(np.int8)


def smooth_probabilities_sequence(prob_seq: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Smooth probabilities by weighted neighboring segments."""
    weights_arr = np.asarray(weights, dtype=np.float32)
    if weights_arr.ndim != 1 or len(weights_arr) % 2 == 0:
        raise ValueError("weights must be a one-dimensional odd-length sequence")
    weights_arr = weights_arr / weights_arr.sum()

    seq = np.asarray(prob_seq, dtype=np.float32)
    radius = len(weights_arr) // 2
    padded = np.pad(seq, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(seq, dtype=np.float32)
    for offset, weight in enumerate(weights_arr):
        out += weight * padded[offset : offset + len(seq)]
    return out


def close_short_gaps_sequence(pred_seq: np.ndarray, max_gap: int = 1) -> np.ndarray:
    """Fill short zero gaps between positives, but do not delete isolated positives."""
    seq = np.asarray(pred_seq).astype(np.int8)
    out = seq.copy()
    if max_gap <= 0:
        return out

    for c in range(seq.shape[1]):
        positive = np.flatnonzero(seq[:, c] > 0)
        if len(positive) < 2:
            continue
        for left, right in zip(positive[:-1], positive[1:]):
            gap = right - left - 1
            if 0 < gap <= max_gap:
                out[left : right + 1, c] = 1
    return out.astype(np.int8)


def dilate_sequence(pred_seq: np.ndarray, radius: int = 1, columns: Sequence[int] | None = None) -> np.ndarray:
    """Expand positives by ``radius`` segments for selected classes."""
    seq = np.asarray(pred_seq).astype(np.int8)
    out = seq.copy()
    if radius <= 0:
        return out
    if columns is None:
        columns = range(seq.shape[1])

    for c in columns:
        positive = np.flatnonzero(seq[:, c] > 0)
        for t in positive:
            left = max(0, t - radius)
            right = min(seq.shape[0], t + radius + 1)
            out[left:right, c] = 1
    return out.astype(np.int8)


def apply_median_filter(predictions, index, window: int) -> np.ndarray:
    """Apply binary median filtering per recording."""
    return apply_per_recording(predictions, index, lambda seq: median_filter_sequence(seq, window=window))


def apply_temporal_smoothing(probabilities, index, weights: Sequence[float]) -> np.ndarray:
    """Apply probability smoothing per recording."""
    return apply_per_recording(probabilities, index, lambda seq: smooth_probabilities_sequence(seq, weights=weights))


def apply_gap_closing(predictions, index, max_gap: int = 1) -> np.ndarray:
    """Apply short-gap closing per recording."""
    return apply_per_recording(predictions, index, lambda seq: close_short_gaps_sequence(seq, max_gap=max_gap))


def apply_dilation(predictions, index, radius: int = 1, columns: Sequence[int] | None = None) -> np.ndarray:
    """Apply binary dilation per recording."""
    return apply_per_recording(predictions, index, lambda seq: dilate_sequence(seq, radius=radius, columns=columns))


def apply_class_specific_median(predictions, index, class_windows: Mapping[str, int], class_names: Sequence[str]) -> np.ndarray:
    """Apply a potentially different median window to every class."""
    pred = np.asarray(predictions).astype(np.int8)
    out = np.empty_like(pred)
    for class_idx, class_name in enumerate(class_names):
        window = int(class_windows.get(class_name, 1))
        filtered_col = apply_median_filter(pred[:, [class_idx]], index, window=window)
        out[:, class_idx] = filtered_col[:, 0]
    return out.astype(np.int8)


def apply_class_specific_temporal_smoothing(
    probabilities,
    index,
    class_kernels: Mapping[str, Sequence[float]],
    class_names: Sequence[str],
) -> np.ndarray:
    """Apply a potentially different temporal-smoothing kernel to every class."""
    prob = np.asarray(probabilities, dtype=np.float32)
    out = np.empty_like(prob)
    for class_idx, class_name in enumerate(class_names):
        weights = class_kernels[class_name]
        smoothed_col = apply_temporal_smoothing(prob[:, [class_idx]], index, weights=weights)
        out[:, class_idx] = smoothed_col[:, 0]
    return out


def evaluate_binary_predictions(y_true, y_pred, class_names: Sequence[str], label: str) -> tuple[dict, pd.DataFrame]:
    """Return summary and per-class F1 for one binary prediction matrix."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    summary = {
        "variant": label,
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
    }
    per_class = pd.DataFrame(
        {
            "variant": label,
            "class_name": list(class_names),
            "f1": f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0),
        }
    )
    return summary, per_class


def class_specific_best_windows(
    y_true,
    raw_predictions,
    index,
    class_names: Sequence[str],
    windows: Sequence[int] = (1, 3, 5, 7),
) -> tuple[dict[str, int], pd.DataFrame]:
    """Select the best median-filter window independently per class on validation data."""
    rows = []
    pred = np.asarray(raw_predictions).astype(np.int8)
    y = np.asarray(y_true).astype(int)

    for window in windows:
        filtered = apply_median_filter(pred, index, window=window)
        per_class_f1 = f1_score(y, filtered, average=None, zero_division=0)
        for class_name, score in zip(class_names, per_class_f1):
            rows.append({"class_name": class_name, "window": int(window), "validation_f1": float(score)})

    table = pd.DataFrame(rows)
    best = (
        table.sort_values(["class_name", "validation_f1", "window"], ascending=[True, False, True])
        .groupby("class_name", as_index=False)
        .first()
    )
    return dict(zip(best["class_name"], best["window"])), table


def _threshold_array(thresholds: Mapping[str, float], class_names: Sequence[str]) -> np.ndarray:
    """Convert a threshold mapping to a class-aligned vector."""
    return np.asarray([thresholds[class_name] for class_name in class_names], dtype=np.float32)


def threshold_probabilities(probabilities, thresholds: Mapping[str, float], class_names: Sequence[str]) -> np.ndarray:
    """Apply class-wise thresholds to probabilities."""
    return (np.asarray(probabilities, dtype=np.float32) >= _threshold_array(thresholds, class_names)).astype(np.int8)


def allowed_postprocessing_candidates(
    probabilities,
    raw_predictions,
    index,
    thresholds: Mapping[str, float],
    class_names: Sequence[str],
    median_windows: Sequence[int] = (1, 3, 5),
    smoothing_kernels: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, np.ndarray]:
    """Build prediction matrices for allowed post-processing candidates only.

    The candidate set intentionally contains only no-op, median filtering, and temporal
    smoothing.  Gap closing and dilation are excluded because they are not listed as
    valid main post-processing methods in the Task 5 PDF.
    """
    if smoothing_kernels is None:
        smoothing_kernels = {
            "smooth_light": (0.25, 0.50, 0.25),
            "smooth_medium": (0.10, 0.20, 0.40, 0.20, 0.10),
        }

    candidates: dict[str, np.ndarray] = {"none": np.asarray(raw_predictions).astype(np.int8)}
    for window in median_windows:
        if int(window) <= 1:
            candidates["none"] = np.asarray(raw_predictions).astype(np.int8)
        else:
            candidates[f"median_w{int(window)}"] = apply_median_filter(raw_predictions, index, window=int(window))

    for name, weights in smoothing_kernels.items():
        smoothed = apply_temporal_smoothing(probabilities, index, weights=weights)
        candidates[name] = threshold_probabilities(smoothed, thresholds, class_names)

    return candidates


def select_classwise_allowed_postprocessing(
    y_true,
    probabilities,
    raw_predictions,
    index,
    thresholds: Mapping[str, float],
    class_names: Sequence[str],
    median_windows: Sequence[int] = (1, 3, 5),
    smoothing_kernels: Mapping[str, Sequence[float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Select the best allowed post-processing candidate independently per class.

    Returns
    -------
    selection, candidate_table, candidates:
        ``selection`` has one row per class and contains the validation-selected
        candidate. ``candidate_table`` stores all per-class validation F1 values.
        ``candidates`` maps candidate names to full prediction matrices, useful for
        applying the selected class-wise recipe to the same split.
    """
    y = np.asarray(y_true).astype(int)
    candidates = allowed_postprocessing_candidates(
        probabilities,
        raw_predictions,
        index,
        thresholds,
        class_names,
        median_windows=median_windows,
        smoothing_kernels=smoothing_kernels,
    )

    priority = {"none": 0}
    priority.update({f"median_w{int(window)}": 10 + int(window) for window in median_windows if int(window) > 1})
    if smoothing_kernels is None:
        smoothing_names = ["smooth_light", "smooth_medium"]
    else:
        smoothing_names = list(smoothing_kernels)
    priority.update({name: 100 + idx for idx, name in enumerate(smoothing_names)})

    rows = []
    for candidate_name, pred in candidates.items():
        scores = f1_score(y, pred, average=None, zero_division=0)
        for class_name, score in zip(class_names, scores):
            rows.append(
                {
                    "class_name": class_name,
                    "candidate": candidate_name,
                    "method_family": "none"
                    if candidate_name == "none"
                    else "median_filtering"
                    if candidate_name.startswith("median_w")
                    else "temporal_smoothing",
                    "validation_f1": float(score),
                    "tie_break_priority": priority.get(candidate_name, 999),
                }
            )

    candidate_table = pd.DataFrame(rows)
    selection = (
        candidate_table.sort_values(
            ["class_name", "validation_f1", "tie_break_priority"],
            ascending=[True, False, True],
        )
        .groupby("class_name", as_index=False)
        .first()
        .drop(columns=["tie_break_priority"])
    )
    return selection, candidate_table.drop(columns=["tie_break_priority"]), candidates


def apply_classwise_allowed_selection(
    probabilities,
    raw_predictions,
    index,
    selection: pd.DataFrame,
    thresholds: Mapping[str, float],
    class_names: Sequence[str],
    median_windows: Sequence[int] = (1, 3, 5),
    smoothing_kernels: Mapping[str, Sequence[float]] | None = None,
) -> np.ndarray:
    """Apply a validation-selected allowed post-processing recipe class-wise."""
    candidates = allowed_postprocessing_candidates(
        probabilities,
        raw_predictions,
        index,
        thresholds,
        class_names,
        median_windows=median_windows,
        smoothing_kernels=smoothing_kernels,
    )
    selection_by_class = dict(zip(selection["class_name"], selection["candidate"]))

    out = np.empty_like(np.asarray(raw_predictions).astype(np.int8))
    for class_idx, class_name in enumerate(class_names):
        candidate_name = selection_by_class[class_name]
        out[:, class_idx] = candidates[candidate_name][:, class_idx]
    return out.astype(np.int8)
