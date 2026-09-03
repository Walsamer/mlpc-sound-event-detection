#!/usr/bin/env python3
"""
Train the final Logistic Regression system (independent per-class binary classifiers).

Reproduces the selected setting from the project: C = 0.01, no class weighting,
per-class thresholds tuned on the validation split. Artifacts (model + scaler +
thresholds) are written to models/ and can be consumed by scripts/evaluate.py.

Requires the private challenge dataset (see docs/DATASET.md); exits gracefully when it
is missing.
"""
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from src import config
from src.data_io import stack_feature_matrix
from src.features import apply_scaler, fit_scaler
from src.labels import stack_hard_labels
from src.models import fit_logistic_artifacts


def main() -> None:
    if not config.TRAIN_DIR.exists() or not config.VALIDATION_DIR.exists():
        print("[train] Dataset not found. Expected layout documented in docs/DATASET.md.")
        return

    print("[train] Loading training split ...")
    X_train, idx_train = stack_feature_matrix(None, config.TRAIN_DIR)
    Y_train = stack_hard_labels(config.TRAIN_DIR, idx_train)

    print("[train] Loading validation split ...")
    X_val, idx_val = stack_feature_matrix(None, config.VALIDATION_DIR)
    Y_val = stack_hard_labels(config.VALIDATION_DIR, idx_val)

    print("[train] Fitting scaler on training split only ...")
    scaler = fit_scaler(X_train)
    X_train_s = apply_scaler(scaler, X_train)
    X_val_s = apply_scaler(scaler, X_val)

    print("[train] Fitting per-class Logistic Regression (C=0.01) + tuning thresholds on validation ...")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = fit_logistic_artifacts(
        X_train_s,
        Y_train,
        X_val_s,
        Y_val,
        class_names=list(config.CLASS_NAMES),
        C=0.01,
        class_weight=None,
        random_state=config.SEED,
        cache_path=config.MODELS_DIR / "lr_C0.01_cwnone.joblib",
    )
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")

    print(f"[train] Done. Validation macro F1 = {artifacts['macro_f1']:.4f} "
          f"(micro F1 = {artifacts['micro_f1']:.4f})")
    print(f"[train] Artifacts written to {config.MODELS_DIR / 'lr_C0.01_cwnone.joblib'}")


if __name__ == "__main__":
    main()
