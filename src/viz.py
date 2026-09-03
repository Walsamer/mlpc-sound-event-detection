"""Plotting helpers for the report and slides.

Spectrogram + ground-truth-vs-prediction timelines (error analysis), per-class F1 bar
charts, hyperparameter sweep curves, and threshold plots. Functions save to
`../figures/` with descriptive names so the report can reference them without re-running.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    """Create parent directory for output `path` and return it as `Path`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _spectrogram_2d(spectrogram, n_time: int | None = None) -> np.ndarray:
    """Return spectrogram-like input as matrix `[freq, time]` for plotting."""
    spec = np.asarray(spectrogram, dtype=float)
    if spec.ndim != 2:
        raise ValueError(f"Expected spectrogram [T, F] or [F, T], got {spec.shape}")

    if n_time is not None:
        if spec.shape[0] == n_time:
            return spec.T
        if spec.shape[1] == n_time:
            return spec

    if spec.shape[0] > spec.shape[1]:
        return spec
    spec = spec.T
    return spec


def _active_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous active runs as `(start, length)` pairs."""
    binary = np.asarray(values).astype(bool)
    runs = []
    start = None
    for idx, is_active in enumerate(binary):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            runs.append((start, idx - start))
            start = None
    if start is not None:
        runs.append((start, len(binary) - start))
    return runs


def plot_gt_vs_pred_timeline(spectrogram, gt_segments, pred_segments, class_names, path):
    """Plot spectrogram and GT/prediction timelines for one file.

    Parameters
    ----------
    spectrogram:
        Spectrogram-like matrix `[T, M]` or `[M, T]`, usually `melspect_mean`.
    gt_segments:
        Binary ground-truth labels `[T, C]`.
    pred_segments:
        Binary predictions `[T, C]`.
    class_names:
        Class names matching columns `[C]`.
    path:
        Output figure path.

    Returns
    -------
    path:
        Written figure path.
    """
    out = _ensure_parent(path)
    gt = np.asarray(gt_segments, dtype=float)
    pred = np.asarray(pred_segments, dtype=float)
    spec = _spectrogram_2d(spectrogram, n_time=gt.shape[0])
    if gt.shape != pred.shape:
        raise ValueError(f"GT/prediction shape mismatch: {gt.shape} vs {pred.shape}")
    if gt.ndim != 2:
        raise ValueError(f"Expected labels [T, C], got {gt.shape}")

    active_classes = np.where((gt.sum(axis=0) + pred.sum(axis=0)) > 0)[0]
    if active_classes.size == 0:
        active_classes = np.arange(min(len(class_names), gt.shape[1]))

    timeline = np.ones((active_classes.size * 2, gt.shape[0], 3), dtype=float)
    ytick_labels = []
    gt_color = np.array([0.00, 0.62, 0.45])      # green/teal for ground truth
    pred_color = np.array([0.80, 0.47, 0.65])    # purple/magenta for prediction
    for row, class_idx in enumerate(active_classes):
        gt_active = gt[:, class_idx] > 0
        pred_active = pred[:, class_idx] > 0
        timeline[2 * row, gt_active] = gt_color
        timeline[2 * row + 1, pred_active] = pred_color
        ytick_labels.extend([f"GT {class_names[class_idx]}", f"PR {class_names[class_idx]}"])

    fig_height = max(4.0, 1.8 + 0.35 * len(ytick_labels))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, fig_height),
        gridspec_kw={"height_ratios": [1.2, max(1.0, 0.16 * len(ytick_labels))]},
        constrained_layout=True,
    )
    axes[0].imshow(spec, aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title("Mel-spectrogram features and GT vs prediction timeline")
    axes[0].set_ylabel("mel bin")
    axes[0].set_xlabel("segment")

    axes[1].imshow(timeline, aspect="auto", interpolation="nearest")
    axes[1].set_yticks(np.arange(len(ytick_labels)))
    axes[1].set_yticklabels(ytick_labels, fontsize=8)
    axes[1].set_xlabel("1 s segment")
    axes[1].set_title("Active labels: ground truth (green) vs predicted (purple)")
    axes[1].grid(axis="x", alpha=0.2)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=gt_color, label="ground truth"),
        plt.Rectangle((0, 0), 1, 1, color=pred_color, label="prediction"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_class_f1(results, path):
    """Bar chart of per-class F1, optionally comparing several systems.

    Parameters
    ----------
    results:
        DataFrame with columns `class_name`, `f1`, and optional `system`, or a mapping
        `{system_name: DataFrame}` where each DataFrame has `class_name,f1`.
    path:
        Output figure path.
    """
    out = _ensure_parent(path)
    if isinstance(results, dict):
        frames = []
        for system, frame in results.items():
            current = pd.DataFrame(frame).copy()
            current["system"] = system
            frames.append(current)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(results).copy()
        if "system" not in df.columns:
            df["system"] = "model"

    classes = df["class_name"].drop_duplicates().tolist()
    systems = df["system"].drop_duplicates().tolist()
    x = np.arange(len(classes))
    width = min(0.8 / len(systems), 0.35)

    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, system in enumerate(systems):
        values = (
            df[df["system"] == system]
            .set_index("class_name")
            .reindex(classes)["f1"]
            .to_numpy(dtype=float)
        )
        ax.bar(x + (idx - (len(systems) - 1) / 2) * width, values, width=width, label=system)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=55, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1")
    ax.grid(axis="y", alpha=0.3)
    if len(systems) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_hyperparam_sweep(values, scores, xlabel, path):
    """Line plot of a hyperparameter sweep vs. Segment-based Macro F1.

    `values` and `scores` may be arrays, or `values` may be a DataFrame containing `C`,
    `class_weight`, and `macro_f1` columns.
    """
    out = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if isinstance(values, pd.DataFrame):
        df = values.copy()
        for class_weight, group in df.groupby("class_weight", sort=False):
            group = group.sort_values("C")
            ax.plot(group["C"], group["macro_f1"], marker="o", label=f"class_weight={class_weight}")
        ax.set_xscale("log")
        ax.legend()
    else:
        ax.plot(values, scores, marker="o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Logistic Regression hyperparameter sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_thresholds(thresholds, path):
    """Bar chart of per-class decision thresholds.

    Parameters
    ----------
    thresholds:
        DataFrame with `class_name,threshold` or mapping `{class_name: threshold}`.
    path:
        Output figure path.
    """
    out = _ensure_parent(path)
    if isinstance(thresholds, dict):
        df = pd.DataFrame({"class_name": list(thresholds), "threshold": list(thresholds.values())})
    else:
        df = pd.DataFrame(thresholds).copy()

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(df["class_name"], df["threshold"])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="default 0.5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("threshold")
    ax.set_title("Validation-tuned per-class thresholds")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["class_name"], rotation=55, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lr_result_figures(results_dir: str | Path, figures_dir: str | Path) -> dict[str, Path]:
    """Create report-ready plots from Samuel's LR result CSV files.

    Parameters
    ----------
    results_dir:
        Directory containing `02_lr_hyperparameter_sweep.csv`,
        `02_lr_best_per_class_f1.csv`, and `02_lr_best_thresholds.csv`.
    figures_dir:
        Output directory for generated figures.

    Returns
    -------
    paths:
        Mapping from descriptive figure name to written path.
    """
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    sweep = pd.read_csv(results_dir / "02_lr_hyperparameter_sweep.csv")
    per_class = pd.read_csv(results_dir / "02_lr_best_per_class_f1.csv")
    thresholds = pd.read_csv(results_dir / "02_lr_best_thresholds.csv")

    paths = {
        "hyperparameter_sweep": plot_hyperparam_sweep(
            sweep,
            None,
            "C (log scale)",
            figures_dir / "02_lr_hyperparameter_sweep.png",
        ),
        "per_class_f1": plot_per_class_f1(
            per_class,
            figures_dir / "02_lr_per_class_f1.png",
        ),
        "thresholds": plot_thresholds(
            thresholds,
            figures_dir / "02_lr_per_class_thresholds.png",
        ),
    }
    return paths


def plot_system_macro_comparison(summary, path):
    """Bar chart comparing non-hidden-test Macro F1 across systems.

    Parameters
    ----------
    summary:
        DataFrame or records with columns `system` and `macro_f1`.
    path:
        Output figure path.
    """
    out = _ensure_parent(path)
    df = pd.DataFrame(summary).copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(df["system"], df["macro_f1"], color=["#9ca3af", "#3b82f6", "#10b981"][: len(df)])
    ax.set_ylim(0, max(0.65, float(df["macro_f1"].max()) + 0.08))
    ax.set_ylabel("non-hidden test Macro F1")
    ax.set_title("System comparison on non-hidden test split")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xticklabels(df["system"], rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_postprocessing_window_sweep(window_sweep, path):
    """Line plot for median-filter window size vs Macro/Micro F1.

    Parameters
    ----------
    window_sweep:
        DataFrame with columns `window`, `macro_f1`, and `micro_f1`.
    path:
        Output figure path.
    """
    out = _ensure_parent(path)
    df = pd.DataFrame(window_sweep).copy().sort_values("window")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["window"], df["macro_f1"], marker="o", label="Macro F1")
    ax.plot(df["window"], df["micro_f1"], marker="o", label="Micro F1")
    ax.set_xlabel("median-filter window size")
    ax.set_ylabel("F1")
    ax.set_title("Post-processing sweep on LR predictions")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_postprocessing_class_delta(class_comparison, path):
    """Bar chart of per-class F1 delta from post-processing.

    Parameters
    ----------
    class_comparison:
        DataFrame with columns `class_name` and `delta`.
    path:
        Output figure path.
    """
    out = _ensure_parent(path)
    df = pd.DataFrame(class_comparison).copy().sort_values("delta", ascending=False)
    colors = np.where(df["delta"].to_numpy(dtype=float) >= 0, "#10b981", "#ef4444")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(df["class_name"], df["delta"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("F1 delta (median filter - raw LR)")
    ax.set_title("Per-class effect of median filtering")
    ax.set_xticklabels(df["class_name"], rotation=55, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out
