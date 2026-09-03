#!/usr/bin/env python3
"""
Evaluation script for the Sound Event Detection project.
Loads a trained model and evaluates it on a dataset split.
"""
import sys
import joblib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_io import load_dataset_splits
from src.features import load_feature_matrix, apply_scaler
from src.labels import load_label_matrix
from src.models import LogisticRegressionModel
from src.evaluate import compute_segment_based_metrics

def main(split="validation"):
    data_dir = Path("data")
    dataset_dir = data_dir / "MLPC2026_challenge"
    model_dir = Path("models")
    
    if not dataset_dir.exists():
        print(f"Dataset not found at {dataset_dir}")
        return
    
    if not (model_dir / "logistic_regression_model.joblib").exists():
        print(f"Trained model not found in {model_dir}/")
        print("Please run train.py first.")
        return
    
    print(f"Loading dataset splits...")
    splits = load_dataset_splits(dataset_dir)
    
    if split not in splits:
        print(f"Split '{split}' not found. Available splits: {list(splits.keys())}")
        return
    
    sample_ids = splits[split]
    print(f"Loading {split} features and labels...")
    # Load features
    X_list = []
    for sample_id in sample_ids:
        feats = load_feature_matrix(dataset_dir, sample_id, split)
        X_list.append(feats)
    X = np.concatenate(X_list, axis=0)
    
    # Load labels
    y_list = []
    for sample_id in sample_ids:
        labels = load_label_matrix(dataset_dir, sample_id, split)
        y_list.append(labels)
    y = np.concatenate(y_list, axis=0)
    
    print(f"{split} data shape: X={X.shape}, y={y.shape}")
    
    # Load model and scaler
    model = joblib.load(model_dir / "logistic_regression_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    
    # Apply scaling
    X_scaled = apply_scaler(X, scaler)
    
    # Predict probabilities
    print("Predicting segment probabilities...")
    y_pred_proba = model.predict_proba(X_scaled)
    
    # Compute metrics
    print("Computing segment-based Macro F1 and other metrics...")
    metrics = compute_segment_based_metrics(y, y_pred_proba, threshold=0.5)
    
    print(f"\nResults for {split} set:")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split to evaluate on (train, validation, test)")
    args = parser.parse_args()
    main(args.split)
