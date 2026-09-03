"""Segment-based macro/micro F1 — internal evaluator kept in parity with the official
`evaluate.py` provided with the challenge.

Both ground truth and predictions are converted to non-overlapping 1 s multi-label
segments (onset down, offset up), then per-class precision/recall/F1 are computed and
macro-averaged. Use this for all development decisions on validation / non-hidden test.
"""


def segment_f1(gt_segments, pred_segments, class_names: list[str]):
    """Per-class precision/recall/F1 + macro/micro F1 from aligned segment matrices
    `[N, C]`. Returns a results table and the scalar macro/micro scores."""
    raise NotImplementedError


def evaluate_csv(gt_csv, pred_csv, audio_dir=None):
    """Parity wrapper mirroring the official script: read GT + prediction CSVs, apply
    majority voting and rounding, and return the segment-based macro F1."""
    raise NotImplementedError
