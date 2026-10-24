"""Shared constants, paths and helpers.

Every script is run from the repository root as a module, e.g.
    python -m src.download_data --n-per-occupation 800
"""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
IMAGES = RAW / "images"
OCR_DIR = DATA / "ocr"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
BASELINES_DIR = MODELS / "baselines"
BERT_DIR = MODELS / "bert"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
METRICS = OUTPUT / "metrics"
PREDICTIONS = OUTPUT / "predictions"

METADATA_CSV = RAW / "metadata.csv"          # id, occupation, exp_year, sex, race (fairness only)
REFERENCE_CSV = RAW / "reference_text.csv"   # id, text  (OCR quality + Q2 only, never a feature)
SAMPLED_IDS = RAW / "sampled_ids.txt"
PROCESSED_CSV = PROCESSED / "cv_text.csv"

# ---------------------------------------------------------------- task constants
SEED = 42
OCCUPATIONS = [
    "registered_nurses",
    "elementary_middle_school_teachers",
    "software_developers",
    "accountants_auditors",
    "construction_laborers",
]
CAP = 20                                    # change only if >15% of the sample sits at the cap
BANDS = [(0, 2), (3, 5), (6, 10), (11, CAP)]
BAND_LABELS = ["0-2", "3-5", "6-10", f"11-{CAP}"]

# Text variants produced by preprocessing.py. "ocr_clean" is the main model input.
TEXT_COLUMNS = {
    "ocr_raw": "text_ocr_raw",
    "ocr_clean": "text_ocr_clean",
    "ocr_conf": "text_ocr_conf",            # cleaned + low-confidence lines removed
    "reference": "text_reference",          # Q2 only (evaluation on clean text)
}

# Columns that must never reach a model as a feature.
FORBIDDEN_FEATURES = ["age", "sex", "race", "name", "state", "region"]


def ensure_dirs() -> None:
    for p in [IMAGES, OCR_DIR, PROCESSED, SPLITS, BASELINES_DIR, BERT_DIR,
              FIGURES, TABLES, METRICS, PREDICTIONS]:
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- target helpers
def clip_years(y, cap: int = CAP):
    return np.clip(np.asarray(y, dtype=float), 0, cap)


def to_band(y) -> np.ndarray:
    """Map (rounded) years to band index 0..3."""
    y = np.rint(clip_years(y)).astype(int)
    out = np.zeros_like(y)
    for i, (lo, hi) in enumerate(BANDS):
        out[(y >= lo) & (y <= hi)] = i
    return out


# ---------------------------------------------------------------- IO helpers
def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_split(name: str) -> pd.DataFrame:
    """Return the split joined with the processed text (id, occupation, y, text columns...)."""
    split = pd.read_csv(SPLITS / f"{name}.csv", dtype={"id": str})
    text = pd.read_csv(PROCESSED_CSV, dtype={"id": str})
    return split.merge(text, on="id", how="left", validate="1:1")


def save_predictions(model: str, text_source: str, df: pd.DataFrame, y_pred) -> Path:
    """Store test predictions in a common format consumed by evaluation.py."""
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "id": df["id"].values,
        "occupation": df["occupation"].values,
        "y_true": df["y"].values,
        "y_pred": clip_years(y_pred),
    })
    path = PREDICTIONS / f"{model}__{text_source}.csv"
    out.to_csv(path, index=False)
    return path


def model_tag(model: str, train_text: str) -> str:
    """'tfidf' for the main setup, 'tfidf@ocr_raw' for a Q3 sensitivity variant."""
    return model if train_text == "ocr_clean" else f"{model}@{train_text}"


def update_run_info(model: str, **info) -> None:
    """Merge training time, inference time and hyper-parameters into output/metrics/runs.json."""
    path = METRICS / "runs.json"
    runs = load_json(path) if path.exists() else {}
    runs.setdefault(model, {}).update(info)
    save_json(runs, path)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - clip_years(p))))


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


def hardware_info() -> str:
    import platform
    info = f"{platform.processor() or platform.machine()} | {os.cpu_count()} CPU threads"
    try:
        import torch
        if torch.cuda.is_available():
            info += f" | GPU: {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return info
