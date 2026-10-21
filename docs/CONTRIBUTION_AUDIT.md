# Contribution Audit

## Project Overview
This repository contains the work completed for the Sound Event Detection (SED) challenge in the Machine Learning and Pattern Recognition course at Johannes Kepler University Linz, SS 2026. The goal was to build a system that detects the presence and temporal boundaries of 15 domestic sound-event classes in short audio recordings.

## Team Members
- Samuel Eder
- Andreas Resch  
*(Team name: Quantized Encoders)*

## Instructor-Provided Material
- The challenge dataset (`MLPC2026_challenge.zip`) and raw audio (`MLPC2026_challenge_raw.zip`) were supplied by the course instructors.
- A decision-tree baseline was provided with the challenge; no evaluation scripts were provided. The rest of the pipeline was developed by the team from scratch.

## Samuel Eder's Contributions
- **Logistic regression (final system)**: Developed the logistic regression classifier wrapper (`src/models.py`) that forms the final system (one binary classifier per class).
- **Hyperparameter search and threshold tuning**: Swept the regularization strength C and class weighting, and tuned per-class decision thresholds on the validation split (notebook `02_simple_classifiers.ipynb`).
- **Prediction pipeline**: Converted segment-wise probabilities to event-level predictions, including thresholding, merging, and short-event removal (`src/predict.py`).
- **Error analysis**: Performed qualitative error analysis on mis-detected audio clips (notebook `05_error_analysis.ipynb`).
- **Reporting**: Co-authored the final project report (Sections 2, 2(d), 4) and contributed to the Task 3 report on data exploration and feature analysis.

## Andreas Resch's Contributions
- **Data & label handling**: Built the data-loading module, label aggregation from multiple annotators, and segmentation utilities (`src/data_io.py`, `src/labels.py`, `src/segments.py`).
- **Baseline model**: Reproduced the decision-tree baseline provided by the course and evaluated its performance (decision-tree wrapper in `src/models.py`, notebook `03_baseline.ipynb`).
- **Random forest**: Designed and implemented the random forest model in full, including the hyperparameter grid search (9 configurations, recording-grouped 5-fold cross-validation, run on small batches), the final training runs, and the evaluation on the held-out grouped test split. Because scikit-learn runs on the CPU only, the total compute time was roughly 72 hours. Results: `notebooks/06_random_forest.ipynb`, `results/metrics/06_rf_grid_search_n50.json`, `results/metrics/06_rf_final_test_results.json`. The random forest was not selected for the final system, since logistic regression scored higher under the same protocol.
- **Post-processing**: Implemented median filtering for temporal smoothing, evaluated window sizes, and integrated the technique into the prediction pipeline (`src/postprocess.py`, notebook `04_postprocessing.ipynb`).
- **Evaluation**: Developed the segment-based macro/micro F1 computation (`src/evaluate.py`).
- **Reporting**: Co-authored the final project report (Sections 1, 3) and authored the baseline model section.

## Shared / Team Contributions
- Joint experimental design, data-splitting strategy, and validation protocol.
- Feature handling (delta features, scaling, optional feature selection; `src/features.py`).
- Visualizations (`src/viz.py` and the figures in `results/figures/`): created by both authors; the individual share was not tracked and is therefore not attributed to one person.
- Co-writing of the final report and slide decks, and final review.
- Integration of the classifier outputs with the median-filter post-processing to produce the final submission CSV.
- Joint work on the initial data-loading foundation (Andreas built the core `src/` data I/O; Samuel used it for classifier experiments).

## Attribution Notes
- The exact boundaries of report writing are blurred due to close collaboration; however, the report clearly attributes sections to each author.
- Visualization work cannot be split reliably between the two authors and is listed under shared contributions.
- No personally identifying information, internal grades, or instructor comments are included in this repository.

## Potential Next Steps
- Extend the classical pipeline with temporal deep-learning approaches such as a CRNN.
- Fine-tune a pretrained sound event detection model and compare it with the logistic-regression system under the same validation protocol.
