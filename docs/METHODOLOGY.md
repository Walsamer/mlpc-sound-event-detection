# Methodology

## Problem Definition
- **Task**: Sound Event Detection (SED) for 15 domestic sound-event classes.
- **Input**: Audio recordings up to 35 seconds, represented as log-mel spectrogram features (extracted via librosa).
- **Output**: Presence of each class in each one-second segment (binary prediction per class per segment).
- **Evaluation Metric**: Segment-based Macro F1 at 1-second resolution (official metric).
- **Dataset Structure**: 
  - Split into training, validation, and test sets (test labels hidden for final submission).
  - Each audio file has corresponding `.npz` feature file and `metadata.csv`.
  - Annotations provided per annotator per class per time segment (soft labels averaged and binarized at 0.5 for classical models).

## Data Preparation
1. **Feature Extraction**: Log-mel spectrogram features were precomputed and provided in `.npz` files.
2. **Label Aggregation**: 
   - Annotations from multiple annotators were averaged to produce soft labels per segment per class.
   - Soft labels were binarized at threshold 0.5 to obtain hard labels for training classical models.
   - For CNN training, soft labels were used directly with binary cross-entropy loss.
3. **Dataset Splitting**: 
   - Official train/validation/test split provided by the challenge.
   - No leakage: statistics (scalers, thresholds) computed only on training and validation sets.

## Feature Engineering
- **Base Features**: 40-dimensional log-mel spectrogram coefficients.
- **Derived Features**: 
  - Delta features (first and second order) to capture spectral dynamics.
  - Optional: statistical functionals (mean, std, etc.) over fixed-size windows (not used in final classical models).
- **Feature Processing**:
  - Z-score normalization (fit on training, applied to validation and test).
  - Feature selection (optional) based on variance or mutual information (not used in final model).
  - Final feature dimensionality: typically 120 (40 * 3 for delta-delta) after normalization.

## Model Pipeline
1. **Baseline**: Decision-tree classifier (provided by course) for initial benchmark.
2. **Classical Models**:
   - Logistic Regression (with L2 regularization, hyperparameter-tuned C)
   - Random Forest (hyperparameter-tuned number of trees and depth)
   - Models trained on segment-level features (each segment treated as independent sample).
3. **Temporal Post-Processing**:
   - Median filtering applied to per-class segment probabilities to reduce isolated predictions.
   - Window size tuned on validation set.
4. **Prediction Aggregation**:
   - Segment-wise probabilities (after post-processing) converted to event predictions via thresholding and connected component analysis.
   - Events merged if gap < threshold (temporal smoothing).
   - Final submission: CSV file with columns `filename`, `onset`, `offset`, `event_label`.

## Experimental Design
- **Train/Validation/Test**: Used the official split; validation used for hyperparameter tuning and threshold optimization.
- **Class Imbalance**: 
  - No class weighting in logistic regression (found to decrease performance).
  - Threshold tuning per class on validation set to optimize Macro F1.
- **Random Seeds**: Fixed seeds for reproducibility (numpy, scikit-learn, torch).
- **Hyperparameter Tuning**: 
  - Logistic Regression: swept C values (e.g., 0.001, 0.01, 0.1, 1, 10).
  - Random Forest: swept number of trees (e.g., 10, 50, 100, 200) and max depth.
  - Post-processing: swept median filter window sizes (e.g., 1, 3, 5, 7, 9 seconds).
- **Model Selection**: Based on validation Macro F1; final model evaluated on non-hidden test set.

## Evaluation
- **Primary Metric**: Segment-based Macro F1 (average of per-class F1 scores).
- **Secondary Metrics**: Per-class F1, confusion matrix, precision/recall trade-offs.
- **Statistical Significance**: Not formally computed; improvements judged by absolute Macro F1 gains.
- **Error Analysis**: Qualitative inspection of spectrograms for false positives/negatives, particularly for short-duration classes.

## Reproducibility
- All code resides in `src/` and is imported by notebooks.
- Notebooks are organized to demonstrate specific stages:
  - Data exploration (Task 3)
  - Baseline reproduction (Andreas)
  - Simple classifiers (Samuel)
  - Post-processing (Andreas)
  - Error analysis (Samuel)
- Deterministic behavior ensured by seeding; however, slight variations may occur due to thread scheduling in parallel operations.
