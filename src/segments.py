"""Segment geometry: overlapping 1 s segments (0.5 s hop) <-> full-second segments,
and conversions between per-segment predictions and (onset, offset) events.

Rounding rule (must match the evaluator): onset rounded DOWN, offset rounded UP to whole
seconds before expanding into 1 s segments.
"""


def keep_full_second_segments(start_time):
    """Boolean mask selecting segments that start at full seconds (0, 1, 2, ...),
    i.e. discarding the 0.5 s overlapping segments as in the baseline."""
    raise NotImplementedError


def predictions_to_events(pred_per_second, filename: str, class_names: list[str]):
    """Convert a per-second per-class prediction matrix into event rows
    `(filename, annotation, onset, offset)` by merging consecutive active seconds."""
    raise NotImplementedError


def events_to_segments(events, n_seconds: int, class_names: list[str]):
    """Expand `(onset, offset, class)` events into a non-overlapping 1 s multi-label
    segment matrix `[n_seconds, C]` using the onset-down/offset-up rule."""
    raise NotImplementedError
