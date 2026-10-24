"""Baseline 1 — TF-IDF + Ridge regression.

Word 1-2 grams, optionally plus character 3-5 grams. Years and numbers are kept as tokens
by default (they are the evidence); `--no-digits` removes them for the Q3 experiment.
Regularization strength and the char-gram switch are tuned on the validation set.

    python -m src.baseline_tfidf                       # main setup
    python -m src.baseline_tfidf --no-digits           # Q3: numbers removed
    python -m src.baseline_tfidf --train-text ocr_raw  # Q3: raw OCR
"""
from __future__ import annotations

import argparse
import re
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline

from src.utils import (BASELINES_DIR, TEXT_COLUMNS, ensure_dirs, load_split, mae,
                       model_tag, save_predictions, set_seed, update_run_info)

ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def strip_digits(s: str) -> str:
    return re.sub(r"\d", " ", s.lower())


def build(alpha: float, use_char: bool, keep_digits: bool) -> Pipeline:
    pre = None if keep_digits else strip_digits
    feats = [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                                      sublinear_tf=True, preprocessor=pre,
                                      token_pattern=r"(?u)\b\w+\b"))]
    if use_char:
        feats.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                              max_features=100_000, sublinear_tf=True,
                                              preprocessor=pre)))
    return Pipeline([("tfidf", FeatureUnion(feats)), ("ridge", Ridge(alpha=alpha))])


def path_for(tag: str):
    return BASELINES_DIR / f"{tag.replace('@', '__')}.joblib"


def predict(texts, tag: str = "tfidf", model=None) -> np.ndarray:
    model = model or joblib.load(path_for(tag))
    return np.clip(model.predict(list(texts)), 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-text", choices=list(TEXT_COLUMNS)[:3], default="ocr_clean")
    ap.add_argument("--no-digits", action="store_true")
    args = ap.parse_args()
    ensure_dirs(); set_seed()

    tag = model_tag("tfidf_nodigits" if args.no_digits else "tfidf", args.train_text)
    col = TEXT_COLUMNS[args.train_text]
    train, val, test = (load_split(s) for s in ("train", "val", "test"))

    best = None
    for use_char in (False, True):
        for alpha in ALPHAS:
            m = build(alpha, use_char, not args.no_digits).fit(train[col], train["y"])
            score = mae(val["y"], m.predict(val[col]))
            print(f"char={use_char} alpha={alpha:<5} val MAE={score:.3f}")
            if best is None or score < best[0]:
                best = (score, alpha, use_char)
    _, alpha, use_char = best

    t0 = time.perf_counter()
    model = build(alpha, use_char, not args.no_digits).fit(train[col], train["y"])
    train_s = time.perf_counter() - t0
    joblib.dump(model, path_for(tag))

    t0 = time.perf_counter()
    save_predictions(tag, args.train_text, test, predict(test[col], model=model))
    infer = (time.perf_counter() - t0) / len(test)
    if args.train_text == "ocr_clean":  # Q2: same model, clean reference text
        save_predictions(tag, "reference", test, predict(test[TEXT_COLUMNS["reference"]], model=model))

    update_run_info(tag, train_s=train_s, infer_s_per_cv=infer, val_mae=best[0],
                    params={"alpha": alpha, "char_ngrams": use_char,
                            "keep_digits": not args.no_digits, "train_text": args.train_text})
    print(f"[{tag}] best alpha={alpha} char={use_char} val MAE={best[0]:.3f}")


if __name__ == "__main__":
    main()
