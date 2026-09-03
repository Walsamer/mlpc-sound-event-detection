# Methodology

## Problem

Segment-based Sound Event Detection (SED) over 15 domestic sound-event classes:

- **Input** — per-recording `.npz` files with per-segment descriptor features and
  per-annotator annotations (see `docs/DATASET.md`).
- **Output** — per-second presence per class, converted into `(onset, offset)` events
  for the official submission format.
- **Metric** — segment-based **macro F1** at 1-second resolution (per-class F1 averaged
  over classes, computed on aligned 1 s multi-label segments).

## Label construction

1. Load per-recording annotation tensor `[T, C, A]` (`A` annotators).
2. Soft labels: mean over the annotator axis → `[T, C]` in [0, 1].
3. Hard labels: majority vote (soft >= 0.5) → `[T, C]` binary, matching the official
   evaluator's rule (`src/labels.py`).

## Feature engineering

- Per 1 s analysis window (0.5 s hop), a compact statistic summary (mean/std/min/max)
  is computed for each descriptor (mel spectrogram, MFCC + deltas, spectral
  centroid/bandwidth/contrast/rolloff/flux/flatness, ZCR, energy, power).
- Descriptor statistics are stacked into a 960-dimensional vector per segment
  (`src/features.py`, `src/data_io.py`).
- An imputation + z-score pipeline is fit on the **training split only** and then
  applied to validation/test — no leakage of scaler statistics.

## Models

Independent per-class binary classifiers (multi-label one-vs-rest):

- **Decision-tree baseline** — supplied with the challenge; reproduced for a reference
  point (non-hidden-test macro F1 ≈ 0.317).
- **Logistic Regression (final)** — one L2-regularized binary classifier per class.
  Hyperparameters were selected on validation: `C ∈ {0.001, 0.01, 0.1, 1, 10}` × class
  weighting `{none, balanced}`. Winner: **C = 0.01, no class weighting** (validation
  macro F1 ≈ 0.523).
- Random forests were explored during model selection in an earlier course stage; LR was
  chosen because it generalized better on validation.

## Class imbalance & thresholds

- Balanced class weighting consistently reduced validation macro F1 (false positives on
  the abundant background outweigh rare-class recall gains).
- Instead, per-class decision thresholds are tuned on **validation only**, scanning a
  0.05–0.95 grid to maximize per-class F1 (`src/predict.py::tune_thresholds`).
- Resulting thresholds are stored with the model artifacts and applied to validation,
  non-hidden test, and (in inference) to the hidden test.

## Temporal post-processing

- Median filtering of per-class probability sequences, applied **per recording** so
  filtering never crosses recording boundaries (`src/postprocess_variants.py`).
- Window size swept on validation (`w ∈ {1, 3, 5, 7, 9, 11}` s). `w = 5` was selected:
  it raises non-hidden-test macro F1 from 0.541 → 0.559 by suppressing isolated spurious
  activations while preserving short genuine events.
- Experiment logs: `results/metrics/02_postprocessing_window_sweep_from_andreas.csv`,
  `results/metrics/03_postprocessing_variants_nht_summary.csv`.

## Evaluation protocol

- Segment-level macro/micro F1 computed with `sklearn.metrics.f1_score` over hard labels
  on aligned 1 s segments (ground truth = majority vote, predictions = thresholded
  probabilities), matching the official metric definition.
- Development metrics come from **validation**; headline README numbers are reported on
  the **non-hidden test** split (labels visible during the course). Scores from the two
  evaluation setups are not directly comparable to hidden-test results.
- All thresholds and hyperparameters were fixed using train/validation only.

## Reproducibility

- `SEED = 42` in `src/config.py`; every learner receives an explicit `random_state`.
- Feature stacking, label stacking, and model sweeps are cache-aware
  (`cache_path`/`force` arguments) so experiments are cheap to re-run.
- Scripts `scripts/prepare_data.py`, `scripts/train.py`, `scripts/evaluate.py` run the
  data check, the final LR training (+ threshold tuning), and evaluation, respectively;
  they fail gracefully without the private dataset.
- Notebooks mirror the real workflow order: data exploration (01), simple classifiers
  (02), baseline reproduction (03), post-processing (04), error analysis (05).
