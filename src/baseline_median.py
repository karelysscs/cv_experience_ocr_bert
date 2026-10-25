"""Baseline 0 — always predict the median years of the training set.

Every other model must beat it; otherwise it learned nothing.

    python -m src.baseline_median
"""
from __future__ import annotations

import time

import numpy as np

from src.utils import (BASELINES_DIR, TEXT_COLUMNS, ensure_dirs, load_json, load_split,
                       mae, save_json, save_predictions, update_run_info)

PATH = BASELINES_DIR / "median.json"


def predict(texts, median: float | None = None) -> np.ndarray:
    median = load_json(PATH)["median"] if median is None else median
    return np.full(len(texts), median, dtype=float)


def main():
    ensure_dirs()
    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    t0 = time.perf_counter()
    median = float(np.median(train["y"]))
    train_s = time.perf_counter() - t0
    save_json({"median": median}, PATH)

    print(f"median={median}  val MAE={mae(val['y'], predict(val, median)):.3f}")
    t0 = time.perf_counter()
    pred = predict(test, median)
    infer = (time.perf_counter() - t0) / len(test)
    # the median ignores the text, so its predictions are identical for every text source
    for src in TEXT_COLUMNS:
        save_predictions("median", src, test, pred)
    update_run_info("median", train_s=train_s, infer_s_per_cv=infer, params={"median": median})


if __name__ == "__main__":
    main()
