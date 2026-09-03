#!/usr/bin/env python3
"""
Training script for the Sound Event Detection project.
Trains a Logistic Regression model with default hyperparameters.
"""
import sys
import joblib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_io import load_dataset_splits
from src.features import load_feature_matrix, fit_scaler, apply_scaler
from src.labels import load_label_matrix
from src.models import LogisticRegressionModel

def main():
    data_dir = Path("data")
    dataset_dir = data_dir / "MLPC2026_challenge"
    
    if not dataset_dir.exists():
        print(f"Dataset not found at {dataset_dir}")
        print("Please place the MLPC2026_challenge dataset in the data/ directory.")
        return
    
    print("Loading dataset splits...")
    splits = load_dataset_splits(dataset_dir)
    
    print("Loading training features and labels...")
    # Load features for all training samples
    X_list = []
    for sample_id in splits['train']:
        feats = load_feature_matrix(dataset_dir, sample_id, 'train')
        X_list.append(feats)
    X_train = np.concatenate(X_list, axis=0)
    
    # Load labels for all training samples
    y_list = []
    for sample_id in splits['train']:
        labels = load_label_matrix(dataset_dir, sample_id, 'train')
        y_list.append(labels)
    y_train = np.concatenate(y_list, axis=0)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Feature scaling
    print("Fitting scaler on training data...")
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler)
    
    # Initialize and train model
    print("Training Logistic Regression model (C=0.01)...")
    model = LogisticRegressionModel(C=0.01, class_weight=None, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "logistic_regression_model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    print(f"Model and scaler saved to {model_dir}/")

if __name__ == "__main__":
    import numpy as np
    main()
