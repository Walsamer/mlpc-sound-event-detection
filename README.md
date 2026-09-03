# Sound Event Detection on Domestic Audio

A robust sound event detection (SED) system for identifying 15 domestic sound classes in short audio recordings, developed as part of the Machine Learning and Pattern Recognition course at Johannes Kepler University Linz (SS 2026). This project demonstrates a complete applied ML pipeline—from raw features to event-level predictions—with a focus on handling class imbalance, temporal modeling, and evaluation via segment-based macro F1.

## Problem

Detecting sound events in real-world audio is challenging due to:
- **Multi-label nature**: Multiple events can overlap in time.
- **Strong class imbalance**: Some events (e.g., door slams) are rare and short, while others (e.g., running water) are long and frequent.
- **Temporal localization**: Precise onset/offset times are required for meaningful detection.
- **Variability**: Domestic environments contain diverse noise sources and reverberation.

Given audio recordings up to 35 seconds, the goal is to predict the presence and timing of 15 pre-defined sound-event classes. Performance is measured by segment-based macro F1-score at 1-second resolution, which balances detection ability across all classes regardless of frequency.

## Pipeline

```mermaid
flowchart LR
    A[Audio + Annotations] --> B[Feature Extraction]
    B --> C[Label Aggregation & Binarization]
    C --> D[Train/Validation Split]
    D --> E[Model Training]
    E --> F[Segment-wise Probabilities]
    F --> G[Temporal Post-Processing]
    G --> H[Event Reconstruction]
    H --> I[Segment-based Evaluation]
```

## Approach

### Data & Features
- **Input**: Pre-computed log-mel spectrogram features (40-dimensional) extracted with a 10 ms hop length.
- **Label processing**: Annotations from multiple annotators were averaged to produce soft labels per segment, then binarized at 0.5 for classical model training.
- **Feature engineering**: Delta and delta-delta coefficients were appended to capture spectral dynamics, yielding 120-dimensional feature vectors. Features were standardized (zero mean, unit variance) using statistics from the training set.

### Modeling
We evaluated three classifiers for frame-wise sound event detection:

| Model               | Feature Representation | Temporal Context | Role |
| ------------------- | ---------------------- | ----------------: | ---- |
| Decision Tree       | Log-mel + deltas       | No                | Baseline provided for reference |
| Logistic Regression | Log-mel + deltas       | No                | Linear classifier with L2 regularization |
| Random Forest       | Log-mel + deltas       | No                | Ensemble of decision trees |

All models treat each one-second segment as an independent sample (no explicit temporal modeling). Hyperparameter tuning was performed on the validation set:
- Logistic regression: regularization strength C ∈ {0.001, 0.01, 0.1, 1, 10}
- Random forest: number of trees ∈ {10, 50, 100, 200}, max depth ∈ {5, 10, None}

### Handling Class Imbalance
- No class weighting in logistic regression (found to hurt validation macro F1).
- Instead, we tuned decision thresholds per class on the validation set to optimize macro F1.
- Post-processing: median filtering applied to each class’s probability time series to suppress isolated predictions, with window size tuned on validation data.

### Event Reconstruction
Segment-wise probabilities (after post-processing) were thresholded at 0.5 to obtain binary segment predictions. Consecutive segments with the same label were merged into events, and events shorter than 0.5 seconds were removed to eliminate spurious detections.

## Results

Performance on the non-hidden test set (used for validation during development):

| Approach                                 | Macro F1 |
| ---------------------------------------- | -------: |
| Decision Tree (Baseline)                 |   0.317 |
| Logistic Regression (C=0.01)             |   0.541 |
| Logistic Regression + Median Filter (w=5)|   0.559 |
| Logistic Regression + Classwise Post‑Proc*|   0.569 |

*Explored alternative post‑processing strategies (not selected for final submission).

**Key findings**
- Adding delta features (spectral dynamics) provided a consistent boost over static log‑mel alone.
- Threshold tuning per class improved macro F1 by ~0.04 over a global threshold, highlighting the impact of imbalance.
- Median filtering (window ≈ 5 s) reduced false positives from short, spiky predictions, raising macro F1 by ~0.02.
- The best‑performing post‑processing variant (class‑dependent thresholds and selections) yielded further gains but was omitted from the final pipeline to maintain simplicity and reproducibility.
- Short‑duration classes (e.g., `window_open_close`, `door_open_close`) remained the most challenging, confirming that temporal support limits detection performance.

## Repository Structure

```
mlpc-sound-event-detection/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py          # paths, constants, seeds
│   ├── data_io.py         # loading features and metadata
│   ├── features.py        # feature extraction, scaling, selection
│   ├── labels.py          # annotation to label conversion
│   ├── models.py          # classifier wrappers (LR, RF, etc.)
│   ├── segments.py        # segment handling and decimation
│   ├── evaluate.py        # segment-based macro/micro F1
│   ├── postprocess.py     # temporal smoothing (median filtering)
│   ├── predict.py         # segment probabilities to event CSV
│   └── viz.py             # visualization utilities
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Annotation agreement, feature statistics, t‑SNE visualizations
│   ├── 02_simple_classifiers.ipynb    # Logistic regression & random forest: hyperparameter search, threshold tuning
│   ├── 03_baseline.ipynb              # Decision‑tree baseline reproduction
│   ├── 04_postprocessing.ipynb        # Median filtering evaluation and window‑size selection
│   └── 05_error_analysis.ipynb        # Qualitative error analysis on mis‑detected audio clips
│
├── reports/
│   ├── final_report.pdf               # Final project report
│   ├── task_3_report.pdf              # Task 3 report
│   ├── task_3_slides.pdf              # Task 3 slides
│   └── task_5_slides.pdf              # Task 5 slides
│
├── results/
│   ├── figures/                       # Final plots (PNG) used in reports and slides
│   │   ├── *.png                      # e.g., per‑class F1, confusion matrices, feature sweeps
│   ├── metrics/                       # Summary statistics (CSV)
│   │   ├── 02_lr_best_summary.csv     # LR hyperparameter tuning results
│   │   ├── 02_postprocessing_window_sweep_from_andreas.csv  # Median filter sweep
│   │   └── 03_postprocessing_variants_nht_summary.csv       # Post‑processing comparison
│   └── predictions.csv                # Final submission CSV (hidden test set predictions)
│
├── docs/
│   ├── CONTRIBUTION_AUDIT.md          # Detailed contribution attribution
│   ├── METHODOLOGY.md                 # In‑depth methodological description
│   └── DATASET.md                     # Expected dataset structure and access instructions
│
└── scripts/
    ├── prepare_data.py                # Data loading and inspection wrapper
    ├── train.py                       # Logistic regression training wrapper
    └── evaluate.py                    # Evaluation wrapper (validation/test)
```

## Reproduction

To reproduce the results (excluding the hidden test set predictions):

1. **Set up the environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Obtain the dataset**
   Place the challenge dataset in `data/MLPC2026_challenge/` following the structure outlined in `docs/DATASET.md`.  
   *Note: The dataset is not included in this repository due to licensing restrictions.*

3. **Run the notebooks** (or the equivalent scripts)
   ```bash
   # Data exploration and feature analysis
   jupyter notebook notebooks/01_data_exploration.ipynb
   
   # Baseline model reproduction
   jupyter notebook notebooks/03_baseline.ipynb
   
   # Simple classifiers (logistic regression & random forest)
   jupyter notebook notebooks/02_simple_classifiers.ipynb
   
   # Post‑processing evaluation
   jupyter notebook notebooks/04_postprocessing.ipynb
   
   # Error analysis
   jupyter notebook notebooks/05_error_analysis.ipynb
   ```

4. **Train and evaluate a model**
   ```bash
   python scripts/prepare_data.py   # inspects dataset structure
   python scripts/train.py          # trains logistic regression model (C=0.01)
   python scripts/evaluate.py       # evaluates on validation set
   ```

   The final submission CSV (`results/predictions.csv`) can be generated by running the prediction script (see `src/predict.py`).

## Dataset

The original dataset was provided for use within the Machine Learning and Pattern Classification course at JKU Linz and is not included in this repository due to licensing restrictions.

See `docs/DATASET.md` for the expected directory structure and feature format.

## Reports

- `reports/final_report.pdf`: Final project report.
- `reports/task_3_report.pdf`: Report on data exploration and feature analysis.
- `reports/task_3_slides.pdf`: Slide deck for Task 3.
- `reports/task_5_slides.pdf`: Slide deck for the final project.

## Authors and Contribution

This work was completed as a team project by **Samuel Eder** and **Andreas Resch** (Team Quantized Encoders).  
A detailed contribution breakdown is available in `docs/CONTRIBUTION_AUDIT.md`.

In brief:
- **Samuel Eder**: Feature engineering, logistic regression and random forest modeling, threshold tuning, error analysis, visualization, and integration of the final prediction pipeline.
- **Andreas Resch**: Data loading and label processing, baseline decision‑tree model, median filtering post‑processing, evaluation infrastructure, and baseline experimentation.
- **Joint effort**: Experimental design, report writing, slide preparation, and final review.

