# Contribution Audit

## Project Overview
This repository contains the work completed for the Sound Event Detection (SED) challenge in the Machine Learning and Pattern Recognition course at Johannes Kepler University Linz, SS 2026. The goal was to build a system that detects the presence and temporal boundaries of 15 domestic sound-event classes in short audio recordings.

## Team Members
- Samuel Eder
- Andreas Resch  
*(Team name: Quantized Encoders)*

## Instructor‑Provided Material
- The challenge dataset (`MLPC2026_challenge.zip`) and raw audio (`MLPC2026_challenge_raw.zip`) were supplied by the course instructors.
- No baseline code or evaluation scripts were provided; the team developed the entire pipeline from scratch.

## Samuel Eder’s Contributions
- **Feature engineering**: Implemented delta and delta‑delta feature extraction, feature scaling, and optional feature selection (`src/features.py`).
- **Modeling**: Developed the logistic regression and random forest wrappers, hyperparameter tuning, and threshold optimization (`src/models.py`).
- **Prediction pipeline**: Converted segment‑wise probabilities to event‑level predictions, including thresholding, merging, and short‑event removal (`src/predict.py`).
- **Visualization**: Created plotting functions for spectrograms, ground‑truth vs. prediction timelines, and error analysis (`src/viz.py`).
- **Experiments**: Conducted hyperparameter searches for C and class weighting, evaluated multiple threshold strategies, and performed qualitative error analysis on mis‑detected audio clips (notebooks `02_simple_classifiers.ipynb` and `05_error_analysis.ipynb`).
- **Reporting**: Co‑authored the final project report (Sections 2, 2(d), 4) and contributed to the Task 3 report on data exploration and feature analysis.

## Andreas Resch’s Contributions
- **Data & label handling**: Built the data‑loading module, label aggregation from multiple annotators, and segmentation utilities (`src/data_io.py`, `src/labels.py`, `src/segments.py`).
- **Baseline model**: Reproduced the decision‑tree baseline provided by the course and evaluated its performance (`src/models.py` decision‑tree wrapper, notebook `03_baseline.ipynb`).
- **Post‑processing**: Implemented median filtering for temporal smoothing, evaluated window sizes, and integrated the technique into the prediction pipeline (`src/postprocess.py`, notebook `04_postprocessing.ipynb`).
- **Evaluation**: Developed the segment‑based macro/micro F1 computation (`src/evaluate.py`).
- **Reporting**: Co‑authored the final project report (Sections 1, 3) and authored the baseline model section.

## Shared / Team Contributions
- Joint experimental design, data‑splitting strategy, and validation protocol.
- Co‑writing of the final report and slide decks.
- Integration of Samuel’s classifier outputs with Andreas’s post‑processing to produce the final submission CSV.
- Joint work on the initial data‑loading foundation (Andreas built the core `src/` data I/O; Samuel used it for classifier experiments).

## Attribution Notes
- The exact boundaries of report writing are blurred due to close collaboration; however, the report clearly attributes sections to each author.
- No personally identifying information, internal grades, or instructor comments are included in this repository.

## Potential Next Steps
- Extend the classical pipeline with temporal deep-learning approaches such as a CRNN.
- Fine-tune a pretrained sound event detection model and compare it with the logistic-regression system under the same validation protocol.
