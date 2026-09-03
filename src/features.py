"""Feature assembly: select/stack feature keys, standardize, optional selection/PCA.

Anything that learns from data (scaler means/stds, selectors, PCA) must be fit on the
training split only and then applied to validation/test. See ../AGENTS.md (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_KEYS = [
    "zcr_mean", "zcr_std", "zcr_min", "zcr_max",
    "melspect_mean", "melspect_std", "melspect_min", "melspect_max",
    "mfcc_mean", "mfcc_std", "mfcc_min", "mfcc_max",
    "mfcc_d_mean", "mfcc_d_std", "mfcc_d_min", "mfcc_d_max",
    "mfcc_d2_mean", "mfcc_d2_std", "mfcc_d2_min", "mfcc_d2_max",
    "flux_mean", "flux_std", "flux_min", "flux_max",
    "flatness_mean", "flatness_std", "flatness_min", "flatness_max",
    "centroid_mean", "centroid_std", "centroid_min", "centroid_max",
    "bandwidth_mean", "bandwidth_std", "bandwidth_min", "bandwidth_max",
    "contrast_mean", "contrast_std", "contrast_min", "contrast_max",
    "rolloff_low_mean", "rolloff_low_std", "rolloff_low_min", "rolloff_low_max",
    "rolloff_high_mean", "rolloff_high_std", "rolloff_high_min", "rolloff_high_max",
    "energy_mean", "energy_std", "energy_min", "energy_max",
    "power_mean", "power_std", "power_min", "power_max",
]


@dataclass(frozen=True)
class IdentitySelector:
    """No-op transformer used when all features `[N, F]` are retained."""

    def transform(self, X):
        """Return `X [N, F]` unchanged."""
        return X

    def fit_transform(self, X, y=None):
        """Return `X [N, F]` unchanged while matching the sklearn transformer API."""
        return X


def _as_2d_float32(array) -> np.ndarray:
    """Convert one feature array to float32 matrix `[T, d]`."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f"Expected feature array with 1 or 2 dims, got shape {arr.shape}")
    return arr


def _replace_non_finite(X) -> np.ndarray:
    """Convert `X [N, F]` to float32 and mark inf values as NaN for imputation."""
    arr = np.asarray(X, dtype=np.float32)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def default_feature_keys() -> list[str]:
    """Return the list of `.npz` feature keys to stack (the 960-dim default set)."""
    return list(FEATURE_KEYS)


def build_feature_matrix(data: dict, feature_keys: list[str] | None = None) -> np.ndarray:
    """Stack selected `.npz` features for one recording.

    Parameters
    ----------
    data:
        Loaded `.npz` dict containing feature arrays with shared segment axis `T`.
    feature_keys:
        Feature keys to stack. Defaults to the 960-dimensional Task-5 feature set.

    Returns
    -------
    X:
        Feature matrix `[T, F]` as float32, where `T` is segments and `F` is features.
    """
    keys = default_feature_keys() if feature_keys is None else list(feature_keys)
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Missing feature keys in recording: {missing}")

    arrays = [_as_2d_float32(data[key]) for key in keys]
    n_segments = {array.shape[0] for array in arrays}
    if len(n_segments) != 1:
        shapes = {key: array.shape for key, array in zip(keys, arrays)}
        raise ValueError(f"Feature arrays have inconsistent segment counts: {shapes}")
    return np.concatenate(arrays, axis=1)


def fit_scaler(X_train):
    """Fit imputation + standardization on training features `X_train [N, F]`.

    Returns a fitted sklearn `Pipeline` that first mean-imputes missing/non-finite values
    and then standardizes features. Fit this on train only and reuse for validation/test.
    """
    scaler = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("standardizer", StandardScaler()),
        ]
    )
    return scaler.fit(_replace_non_finite(X_train))


def apply_scaler(scaler, X):
    """Apply a fitted scaler to `X [N, F]` -> standardized `[N, F]`."""
    return scaler.transform(_replace_non_finite(X)).astype(np.float32, copy=False)


def select_features(X_train, y_train, method: str = "none", **kwargs):
    """Fit optional feature selection / dimensionality reduction on train only.

    Parameters
    ----------
    X_train:
        Training features `[N, F]`, usually already standardized.
    y_train:
        Training labels `[N, C]`. Multi-label targets are collapsed to a per-segment
        activity indicator only for univariate feature scoring.
    method:
        One of `"none"`, `"pca"`, or `"select_k_best"`.

    Returns
    -------
    selector, X_selected:
        Fitted transformer and transformed training features `[N, F_selected]`.
    """
    method = method.lower()
    if method in {"none", "identity", "all"}:
        selector = IdentitySelector()
        return selector, selector.transform(X_train)

    if method == "pca":
        n_components = kwargs.get("n_components", 0.95)
        random_state = kwargs.get("random_state", 42)
        selector = PCA(n_components=n_components, random_state=random_state)
        return selector, selector.fit_transform(X_train)

    if method == "select_k_best":
        k = kwargs.get("k", min(300, X_train.shape[1]))
        y_any = (np.asarray(y_train).sum(axis=1) > 0).astype(int)
        selector = SelectKBest(score_func=f_classif, k=k)
        return selector, selector.fit_transform(X_train, y_any)

    raise ValueError(f"Unknown feature-selection method: {method}")
