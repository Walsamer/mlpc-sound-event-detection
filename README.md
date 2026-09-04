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
- **Input**: Per-recording feature/annotation files (`.npz`) covering all splits; the raw waveform is not needed for the final system.
- **Label processing**: Each recording carries per-annotator annotations as a `[T, C, A]` tensor. Soft labels = mean over annotators; hard labels = majority vote (mean >= 0.5), matching the official evaluator's rule.
- **Feature engineering**: Each one-second analysis window (0.5 s hop) is described by statistics (mean/std/min/max) over spectral/temporal descriptors — mel spectrogram, MFCC (+ first/second-order deltas), spectral centroid/bandwidth/contrast/rolloff/flux/flatness, zero-crossing rate, energy and power. The descriptor statistics are stacked into a single 960-dimensional vector per segment.
- **Scaling**: Imputation + z-score standardization, fit on the training split only and applied to validation/test (no leakage).

### Modeling
The final system is a set of independent per-class binary Logistic Regression classifiers (multi-label one-vs-rest). A decision-tree baseline was provided with the challenge and reproduced for comparison; random forests were considered during model selection in an earlier task stage, and logistic regression was selected because it generalized better on the validation split.

| Model               | Feature Representation | Temporal context | Role |
| ------------------- | ---------------------- | ----------------: | ---- |
| Decision Tree       | 960-d descriptor stats  |               No | provided challenge baseline, reproduced |
| Logistic Regression | 960-d descriptor stats  |               No | final system (one binary LR per class) |

All classifiers treat each segment as an independent sample; temporal coherence is handled explicitly by the post-processing stage. Hyperparameters and decision thresholds were selected on the validation split:
- Logistic regression: regularization strength `C` swept over {0.001, 0.01, 0.1, 1, 10} × class weighting {none, balanced}; **C=0.01, no weighting** won.
- Per-class decision thresholds tuned on the validation split over a 0.05–0.95 grid (maximizing per-class F1), never on the hidden test set.

### Handling Class Imbalance
- Class weighting (`balanced`) *hurt* validation macro F1 for every tested `C` — recall gains were outweighed by false positives on the abundant background segments.
- Instead, per-class decision thresholds were tuned on the validation split to optimize per-class F1 (the macro average is dominated by the weakest classes).
- Temporal post-processing: median filtering of each class's probability time series, applied per recording (never across recording boundaries). Window size was selected on the validation split.

### Event Reconstruction
Post-processed probabilities are thresholded per class into binary segment predictions. Consecutive active segments of the same class are merged into single events and emitted as `filename, annotation, onset, offset` rows for submission.

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
- The tuned logistic-regression system roughly **doubled the decision-tree baseline** on the non-hidden test set (0.541 vs 0.317 macro F1) — per-class boundaries learned from the 960-d features matter far more than the provided stump-like baseline.
- Classes are very unequal in difficulty: the baseline's per-class F1 ranges from 0.578 (running water) down to 0.061 (window open/close); short, impulsive events with similar spectra (window/door/wardrobe) are the hard tail.
- Class weighting (`balanced`) consistently reduced macro F1 across all `C` values on validation — more recall on rare classes was not worth the false positives.
- Per-class validation-tuned thresholds beat a global 0.5 threshold by a large margin.
- Median filtering of per-class probabilities (window = 5 s) improved non-hidden-test macro F1 from 0.541 to **0.559** by suppressing spurious isolated activations; larger windows (>= 9 s) start to erase genuinely short events.
- Stronger, class-aware post-processing variants reached 0.569 macro F1 on the non-hidden test split, but the final submission kept the simpler, reproducible median-filter system.

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
│   ├── task_4_report.pdf              # Task 4 report
│   ├── task_4_report.pdf              # Task 4 report
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
- **Samuel Eder**: Feature engineering, logistic regression and random forest modeling, threshold tuning, error analysis, and integration of the final prediction pipeline.
- **Andreas Resch**: Data loading and label processing, feature engineering, baseline decision‑tree model including finetuning of Random Forest, median filtering post‑processing, evaluation infrastructure, and baseline experimentation.
- **Joint effort**: Experimental design, report writing, visualizations, slide preparation, and final review.

