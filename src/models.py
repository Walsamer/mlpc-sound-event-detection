"""Classifier wrappers with a common fit/predict interface.

One interface so the decision-tree baseline, logistic regression, and random forest are
swappable in the same pipeline. Each is a per-class binary classifier (one-vs-rest /
independent binary), matching the multi-label baseline design.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

try:
    from .config import SEED
except ImportError:  # pragma: no cover - supports direct notebook execution fallbacks
    from config import SEED


@dataclass
class PerClassBinaryModel:
    """Container for independent per-class binary classifiers.

    Attributes
    ----------
    estimators:
        One fitted binary estimator per class.
    model_type:
        Human-readable model family, e.g. `"logistic_regression"`.
    class_names:
        Optional class labels matching probability columns `[N, C]`.
    """

    estimators: list
    model_type: str
    class_names: list[str] | None = None


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Return a safe sklearn/joblib worker count."""
    if n_jobs is None:
        return -1
    if n_jobs == 0:
        return os.cpu_count() or 1
    return n_jobs


def _fit_one_logistic_class(X_train, y_train, C, class_weight, max_iter, solver, random_state):
    """Fit one binary LR classifier for labels `y_train [N]`."""
    y = np.asarray(y_train).astype(int)
    unique = np.unique(y)
    if unique.size < 2:
        estimator = DummyClassifier(strategy="constant", constant=int(unique[0]))
        return estimator.fit(X_train, y)

    estimator = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )
    return estimator.fit(X_train, y)


def _positive_probability(estimator, X) -> np.ndarray:
    """Return probability for class `1` as `[N]`, including constant classifiers."""
    probabilities = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", []))
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(X.shape[0], dtype=np.float32)


def fit_decision_tree_baseline(X_train, Y_train, **kwargs):
    """Fit one binary decision tree per class (the provided baseline)."""
    raise NotImplementedError("Samuel's Topic 2 path uses Logistic Regression only.")


def fit_logistic_regression(X_train, Y_train, C: float = 0.1, **kwargs):
    """Fit independent per-class Logistic Regression models.

    Parameters
    ----------
    X_train:
        Training features `[N, F]`.
    Y_train:
        Multi-label binary targets `[N, C]`.
    C:
        Inverse regularization strength; larger means weaker regularization.

    Keyword Parameters
    ------------------
    class_weight:
        Passed to sklearn `LogisticRegression`; use `"balanced"` for rare classes.
    max_iter:
        Maximum optimizer iterations, default `1000`.
    solver:
        Sklearn solver, default `"lbfgs"`.
    n_jobs:
        Parallel jobs over classes, default `-1`.
    random_state:
        Seed, default from `src.config.SEED`.
    class_names:
        Optional labels for the `C` probability columns.

    Returns
    -------
    model:
        `PerClassBinaryModel` producing probabilities `[N, C]`.
    """
    X = np.asarray(X_train, dtype=np.float32)
    Y = np.asarray(Y_train).astype(int)
    if Y.ndim != 2:
        raise ValueError(f"Expected Y_train [N, C], got shape {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X/Y row mismatch: {X.shape[0]} vs {Y.shape[0]}")

    n_jobs = _resolve_n_jobs(kwargs.get("n_jobs", -1))
    class_weight = kwargs.get("class_weight", None)
    max_iter = kwargs.get("max_iter", 1000)
    solver = kwargs.get("solver", "lbfgs")
    random_state = kwargs.get("random_state", SEED)
    class_names = kwargs.get("class_names")

    estimators = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_logistic_class)(
            X, Y[:, class_idx], C, class_weight, max_iter, solver, random_state
        )
        for class_idx in range(Y.shape[1])
    )
    return PerClassBinaryModel(
        estimators=estimators,
        model_type="logistic_regression",
        class_names=class_names,
    )


def fit_random_forest(X_train, Y_train, n_estimators: int = 200, **kwargs):
    """Fit a (multi-output) random forest classifier."""
    raise NotImplementedError("Samuel's Topic 2 path uses Logistic Regression only.")


def predict_proba(model, X):
    """Return per-class probabilities `[N, C]` from a fitted model."""
    X_arr = np.asarray(X, dtype=np.float32)
    if isinstance(model, PerClassBinaryModel):
        columns = [_positive_probability(estimator, X_arr) for estimator in model.estimators]
        return np.column_stack(columns).astype(np.float32, copy=False)

    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X_arr)
        if isinstance(raw, list):
            columns = []
            for estimator_probs, estimator in zip(raw, getattr(model, "estimators_", [])):
                classes = list(getattr(estimator, "classes_", []))
                if 1 in classes:
                    columns.append(estimator_probs[:, classes.index(1)])
                else:
                    columns.append(np.zeros(X_arr.shape[0], dtype=np.float32))
            return np.column_stack(columns).astype(np.float32, copy=False)
        if raw.ndim == 2:
            return raw.astype(np.float32, copy=False)

    raise TypeError(f"Unsupported model type for probability prediction: {type(model)!r}")


def fit_logistic_artifacts(
    X_train,
    Y_train,
    X_val,
    Y_val,
    class_names: list[str],
    C: float,
    class_weight=None,
    cache_path: str | Path | None = None,
    force: bool = False,
    **kwargs,
) -> dict:
    """Fit one LR setting and tune thresholds on validation data.

    Parameters
    ----------
    X_train, Y_train:
        Training features `[N_train, F]` and binary labels `[N_train, C]`.
    X_val, Y_val:
        Validation features `[N_val, F]` and binary labels `[N_val, C]` used only for
        threshold tuning and model selection.
    class_names:
        Class names matching label/probability columns `[C]`.
    C:
        Logistic Regression inverse regularization strength.
    class_weight:
        Optional sklearn class-weight setting, e.g. `"balanced"`.
    cache_path:
        Optional `joblib` artifact path. If present and `force=False`, load it instead of
        recomputing.
    force:
        Recompute and overwrite `cache_path` when true.

    Returns
    -------
    artifacts:
        Dict with fitted `model`, tuned `thresholds`, validation `macro_f1`/`micro_f1`,
        and setting metadata.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            return joblib.load(cache_path)

    from .predict import apply_thresholds, tune_thresholds

    model = fit_logistic_regression(
        X_train,
        Y_train,
        C=C,
        class_weight=class_weight,
        class_names=class_names,
        **kwargs,
    )
    prob_val = predict_proba(model, X_val)
    thresholds = tune_thresholds(prob_val, Y_val, class_names)
    pred_val = apply_thresholds(prob_val, thresholds, class_names)
    artifacts = {
        "model": model,
        "thresholds": thresholds,
        "C": C,
        "class_weight": class_weight,
        "macro_f1": f1_score(Y_val, pred_val, average="macro", zero_division=0),
        "micro_f1": f1_score(Y_val, pred_val, average="micro", zero_division=0),
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, cache_path)
    return artifacts


def sweep_logistic_regression(
    X_train,
    Y_train,
    X_val,
    Y_val,
    class_names: list[str],
    C_values: list[float],
    class_weight_values: list | tuple = (None, "balanced"),
    cache_path: str | Path | None = None,
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Run a cache-aware LR hyperparameter sweep on validation Macro F1.

    Parameters
    ----------
    X_train, Y_train:
        Training features `[N_train, F]` and labels `[N_train, C]`.
    X_val, Y_val:
        Validation features `[N_val, F]` and labels `[N_val, C]`.
    class_names:
        Class names matching columns `[C]`.
    C_values:
        Regularization values to evaluate.
    class_weight_values:
        Class-weight settings to evaluate, usually `[None, "balanced"]`.
    cache_path:
        Optional CSV path. If present and `force=False`, load cached sweep results.
    force:
        Recompute and overwrite cached CSV when true.

    Returns
    -------
    sweep_df:
        One row per setting with `C`, `class_weight`, `macro_f1`, and `micro_f1`, sorted by
        decreasing validation Macro F1.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            return pd.read_csv(cache_path)

    rows = []
    for C in C_values:
        for class_weight in class_weight_values:
            artifacts = fit_logistic_artifacts(
                X_train,
                Y_train,
                X_val,
                Y_val,
                class_names,
                C=float(C),
                class_weight=class_weight,
                **kwargs,
            )
            rows.append(
                {
                    "C": float(C),
                    "class_weight": "none" if class_weight is None else str(class_weight),
                    "macro_f1": artifacts["macro_f1"],
                    "micro_f1": artifacts["micro_f1"],
                }
            )

    sweep_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(cache_path, index=False)
    return sweep_df
