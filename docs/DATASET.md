# Dataset

## Original Dataset
The sound event detection challenge used a custom dataset provided by the Machine Learning and Pattern Classification course at Johannes Kepler University Linz, SS 2026. The dataset consists of domestic audio recordings annotated for 15 sound-event classes.

Due to licensing restrictions, the original dataset is **not included** in this repository. The dataset was provided for use within the university course only.

## Expected Dataset Structure
To run the code in this repository, the dataset must be placed in the `data/` directory with the following structure:

```
data/
└── MLPC2026_challenge/
    ├── train/
    │   ├── audio/
    │   │   ├── 000001.wav
    │   │   └── ...
    │   ├── features/
    │   │   ├── 000001.npz
    │   │   └── ...
    │   ├── metadata.csv
    │   └── annotations.csv
    ├── validation/
    │   ├── audio/
    │   │   ├── 000001.wav
    │   │   └── ...
    │   ├── features/
    │   │   ├── 000001.npz
    │   │   └── ...
    │   ├── metadata.csv
    │   └── annotations.csv
    └── test/
        ├── audio/
        │   ├── 000001.wav
        │   │   └── ...
        ├── features/
        │   │   ├── 000001.npz
        │   │   └── ...
        └── metadata.csv
```

## Feature Files
- Each `.npz` file contains a variable `logmel` of shape `(T, 40)` where `T` is the number of time steps (frames) and 40 is the number of mel bands.
- Features are extracted with a hop length of 10 ms and window size of 20 ms (yielding 100 frames per second of audio).
- The provided features are log-mel spectrograms (already compressed to decibel scale).

## Metadata
- `metadata.csv` contains columns: `filename`, `duration`, and possibly other metadata.

## Annotations
- `annotations.csv` contains per-annotator per-segment per-class labels (values 0, 1, or possibly fractional for multiple annotators).
- The labeling pipeline averages over annotators to produce soft labels, then binarizes at 0.5 for classical model training.

## Notes
- The raw audio files are only needed if extracting features from scratch; the provided features suffice for the classical models.
- The test set does not include `annotations.csv`; predictions are submitted for the test set only.
- Feature extraction parameters (if needed): sample rate 16000 Hz, n_mels=40, hop_length=160, win_length=400, fmin=0, fmax=8000.
