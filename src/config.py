"""Paths, constants, and seeds for Task 5.

Single source of truth for dataset locations, the class label space, the segment
geometry (1 s window, 0.5 s hop), and the random seed. Import from here instead of
hard-coding values in modules or notebooks.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------------
# Resolved relative to this file so notebooks work regardless of CWD.
TASK_DIR = Path(__file__).resolve().parents[1]
EXTERNAL_DATA_ROOT = Path("/Volumes/2TB_drive/jku_coding/SS26/mlpc/data_task5")
DATA_DIR = EXTERNAL_DATA_ROOT / "MLPC2026_challenge"
RAW_DATA_DIR = EXTERNAL_DATA_ROOT / "MLPC2026_challenge_raw"
BASELINE_DIR = EXTERNAL_DATA_ROOT / "challenge_baseline"
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"          # hidden test: features only, no labels
RESULTS_DIR = TASK_DIR / "results"
FIGURES_DIR = TASK_DIR / "figures"
MODELS_DIR = TASK_DIR / "models"

# --- Label space -----------------------------------------------------------------
# Alphabetical; must match the `class_names` axis of the annotation arrays.
CLASS_NAMES = [
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
]
N_CLASSES = len(CLASS_NAMES)

# --- Segment geometry ------------------------------------------------------------
WINDOW_SECONDS = 1.0   # feature aggregation window
HOP_SECONDS = 0.5      # hop between overlapping segments (50% overlap)
EVAL_RESOLUTION_SECONDS = 1.0  # metric resolution (non-overlapping 1 s segments)

# --- Reproducibility -------------------------------------------------------------
SEED = 42


# --- Hardware acceleration --------------------------------------------------------
def get_device() -> str:
    """Return best available torch device: CUDA, then MPS, then CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
