# Dataset

## Availability

The dataset was created for use within the Machine Learning and Pattern Classification
course at Johannes Kepler University Linz (SS 2026) and is **not redistributable**. It is
therefore **not included** in this repository.

To run the pipeline you need the challenge dataset (`MLPC2026_challenge`). Place it at

```
data/MLPC2026_challenge/
```

or point `MLPC_DATA_DIR` at its parent directory. All paths in `src/config.py` are
relative to the repository root and can be overridden with that environment variable.

## Expected layout

```
data/MLPC2026_challenge/
├── train/
│   ├── audio_features/
│   │   ├── 000001.npz
│   │   └── ...
│   └── metadata.csv            # per-recording metadata (train/validation only)
├── validation/
│   ├── audio_features/...
│   └── metadata.csv
└── test/
    ├── audio_features/...      # hidden test: no annotations / no metadata
    └── metadata.csv
```

## Recording files (`audio_features/*.npz`)

Each `.npz` stores, for one recording:

| key            | shape            | meaning                                              |
| -------------- | ---------------- | ---------------------------------------------------- |
| `annotations`  | `[T, C, A]`      | per-segment, per-class, per-annotator binary labels  |
| `class_names`  | `[C]`            | the 15 sound-event classes                           |
| `start_time`   | `[T]`            | onset of each analysis segment (s)                   |
| `end_time`     | `[T]`            | offset of each analysis segment (s)                  |
| feature keys   | `[T, d]`         | per-segment descriptor statistics (see below)        |

### Features

Per-segment statistics (mean / std / min / max) are computed over 1-second analysis
windows (0.5 s hop) for spectral/temporal descriptors:

- zero-crossing rate
- mel spectrogram
- MFCC, plus first-order (`mfcc_d`) and second-order (`mfcc_d2`) deltas
- spectral flux, flatness, centroid, bandwidth, contrast
- rolloff (low & high), energy, power

`data_io.stack_feature_matrix` stacks all descriptor statistics into one
**960-dimensional vector per segment** (`X` of shape `[N, 960]`).

### Class label space (15 classes)

```
bell_ringing, coffee_machine, cutlery_dishes, door_open_close, footsteps,
keyboard_typing, keychain, light_switch, microwave, phone_ringing,
running_water, toilet_flushing, vacuum_cleaner, wardrobe_drawer_open_close,
window_open_close
```

### Labels

- Soft label per segment/class = mean over annotators.
- Hard label (used for training and for evaluation against the official metric)
  = majority vote, i.e. soft >= 0.5, matching the evaluator's rule.

## Notes

- The hidden `test/` split contains **no** annotations — submissions are event CSVs
  (`filename, annotation, onset, offset`) scored by the official segment-based macro F1.
- Development decisions (thresholds, post-processing windows, model selection) were made
  exclusively on `train/` + `validation/`; the "non-hidden test" numbers quoted in the
  README come from re-scoring on a split whose labels were visible during the course.
- Raw waveforms are only required for bonus/analysis work; the provided features suffice
  for the classical pipeline in this repository.
