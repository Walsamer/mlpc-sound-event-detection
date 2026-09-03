"""Temporal post-processing of per-second predictions (Report Section 3).

Choose ONE for the report: temporal smoothing (weighted combination of neighbors) or
median filtering (median within a window). Both operate per class along the time axis.
"""


def median_filter(pred_seq, window: int = 3):
    """Median-filter a per-class prediction sequence `[T, C]` along time with the given
    odd window size; removes short spurious activations, preserves long events."""
    raise NotImplementedError


def temporal_smoothing(prob_seq, weights=None):
    """Smooth per-class probability sequence `[T, C]` with a weighted combination of
    neighboring frames; returns smoothed probabilities before thresholding."""
    raise NotImplementedError
