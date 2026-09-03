#!/usr/bin/env python3
"""
Data preparation / inspection for the Sound Event Detection project.

The MLPC2026_challenge dataset is *not* redistributable (see docs/DATASET.md), so this
script only verifies the dataset is present in the expected layout and reports basic
statistics. It exits gracefully when the data directory is missing.

Expected layout:

    data/MLPC2026_challenge/
        train/  validation/  test/
            audio_features/*.npz      # features + per-annotator annotations
            metadata.csv              # train/validation (absent for hidden test)
            annotations.csv           # optional; labels usually come from the .npz files
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from src import config
from src.data_io import list_recordings, stack_feature_matrix
from src.labels import stack_hard_labels


def main() -> None:
    if not config.DATA_DIR.exists():
        print(f"[prepare_data] Dataset not found at {config.DATA_DIR}")
        print("[prepare_data] Expected layout is documented in docs/DATASET.md.")
        print("[prepare_data] Place the MLPC2026_challenge dataset under data/ and re-run.")
        print("[prepare_data] (You can point elsewhere via the MLPC_DATA_DIR env var.)")
        return

    for split_name, split_dir in (
        ("train", config.TRAIN_DIR),
        ("validation", config.VALIDATION_DIR),
        ("test", config.TEST_DIR),
    ):
        if not split_dir.exists():
            print(f"[prepare_data] Missing split directory: {split_dir}")
            continue
        recordings = list_recordings(split_dir)
        print(f"[prepare_data] {split_name:>10}: {len(recordings)} recordings")

    # Sanity-check the stacked train matrix + labels so users immediately see expected shapes.
    try:
        X, index = stack_feature_matrix(None, config.TRAIN_DIR)
        print(f"[prepare_data] Stacked train features: X={X.shape} (N segments x F features)")
        Y = stack_hard_labels(config.TRAIN_DIR, index)
        print(f"[prepare_data] Stacked train labels:   Y={Y.shape} (N segments x {config.N_CLASSES} classes)")
    except FileNotFoundError as exc:
        print(f"[prepare_data] Could not stack train split: {exc}")


if __name__ == "__main__":
    main()
