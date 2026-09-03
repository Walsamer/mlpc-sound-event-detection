#!/usr/bin/env python3
"""
Evaluate a trained Logistic Regression system on a labelled split.

Loads the artifacts produced by scripts/train.py, applies the fitted scaler and the
validation-tuned per-class thresholds, and reports segment-level macro/micro F1 plus
per-class F1. Requires the private challenge dataset; exits gracefully when missing.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from src import config
from src.data_io import stack_feature_matrix
from src.features import apply_scaler
from src.labels import stack_hard_labels
from src.models import predict_proba
from src.predict import apply_thresholds

ARTIFACT_PATH = config.MODELS_DIR / "lr_C0.01_cwnone.joblib"


def main(split: str = "validation") -> None:
    split_dir = getattr(config, f"{split.upper()}_DIR", None)
    if split_dir is None or not split_dir.exists():
        print(f"[evaluate] Split '{split}' not available. Expected layout documented in docs/DATASET.md.")
        return

    if not ARTIFACT_PATH.exists():
        print(f"[evaluate] Trained artifacts not found at {ARTIFACT_PATH}. Run scripts/train.py first.")
        return

    artifacts = joblib.load(ARTIFACT_PATH)
    print(f"[evaluate] Loading {split} split ...")
    X, idx = stack_feature_matrix(None, split_dir)
    Y = stack_hard_labels(split_dir, idx)

    scaler = joblib.load(config.MODELS_DIR / "scaler.joblib") if (config.MODELS_DIR / "scaler.joblib").exists() else None
    model = artifacts["model"]
    thresholds = artifacts["thresholds"]

    X_s = apply_scaler(scaler, X) if scaler is not None else X
    proba = predict_proba(model, X_s)
    pred = apply_thresholds(proba, thresholds, config.CLASS_NAMES)

    macro = f1_score(Y, pred, average="macro", zero_division=0)
    micro = f1_score(Y, pred, average="micro", zero_division=0)
    print(f"\n[evaluate] Segment-level results on {split}:")
    print(f"[evaluate]   Macro F1 = {macro:.4f}")
    print(f"[evaluate]   Micro F1 = {micro:.4f}")
    print("\n[evaluate] Per-class F1:")
    print(classification_report(Y, pred, target_names=config.CLASS_NAMES, zero_division=0, digits=3))


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    main(split)
