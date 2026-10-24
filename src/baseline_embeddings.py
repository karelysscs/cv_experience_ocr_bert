"""Baseline 2 — frozen sentence embeddings + Ridge regression.

Each CV is split into chunks that fit the encoder's limit (default 512 tokens for
bge-small-en-v1.5), every chunk is encoded, and the chunk vectors are averaged into one
CV vector. The encoder is NOT fine-tuned; only Ridge is fit (alpha tuned on validation).
Embeddings are cached in models/baselines/emb_cache/ so re-runs are instant.

    python -m src.baseline_embeddings --encoder BAAI/bge-small-en-v1.5
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from src.utils import (BASELINES_DIR, TEXT_COLUMNS, ensure_dirs, load_split, mae,
                       model_tag, save_predictions, set_seed, update_run_info)

ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
CACHE = BASELINES_DIR / "emb_cache"
_ENCODERS = {}


def get_encoder(name: str):
    if name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer
        _ENCODERS[name] = SentenceTransformer(name)
    return _ENCODERS[name]


def chunk_text(text: str, tokenizer, max_tokens: int, stride: int) -> list[str]:
    ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return [str(text)]
    step = max_tokens - stride
    return [tokenizer.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), step)
            if ids[i:i + max_tokens]]


def embed(texts, encoder_name: str, batch_size: int = 32) -> np.ndarray:
    """One vector per CV = mean of its normalized chunk embeddings."""
    enc = get_encoder(encoder_name)
    max_tokens = enc.max_seq_length - 2          # room for [CLS] / [SEP]
    chunks, owner = [], []
    for i, t in enumerate(texts):
        cs = chunk_text(t, enc.tokenizer, max_tokens, stride=64)
        chunks += cs
        owner += [i] * len(cs)
    vecs = enc.encode(chunks, batch_size=batch_size, normalize_embeddings=True,
                      show_progress_bar=True, convert_to_numpy=True)
    owner = np.asarray(owner)
    out = np.vstack([vecs[owner == i].mean(0) for i in range(len(texts))])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def cached_embed(df, col: str, encoder_name: str, split: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{encoder_name.replace('/', '_')}__{col}__{split}.npy"
    if path.exists():
        return np.load(path)
    X = embed(df[col].fillna("").tolist(), encoder_name)
    np.save(path, X)
    return X


def path_for(tag: str):
    return BASELINES_DIR / f"{tag.replace('@', '__')}.joblib"


def predict(texts, tag: str = "embeddings") -> np.ndarray:
    bundle = joblib.load(path_for(tag))
    X = embed(list(texts), bundle["encoder"])
    return np.clip(bundle["ridge"].predict(X), 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--train-text", choices=list(TEXT_COLUMNS)[:3], default="ocr_clean")
    args = ap.parse_args()
    ensure_dirs(); set_seed()

    tag = model_tag("embeddings", args.train_text)
    col = TEXT_COLUMNS[args.train_text]
    train, val, test = (load_split(s) for s in ("train", "val", "test"))

    t0 = time.perf_counter()
    Xtr = cached_embed(train, col, args.encoder, "train")
    Xva = cached_embed(val, col, args.encoder, "val")
    best = min(((mae(val["y"], Ridge(alpha=a).fit(Xtr, train["y"]).predict(Xva)), a) for a in ALPHAS))
    ridge = Ridge(alpha=best[1]).fit(Xtr, train["y"])
    train_s = time.perf_counter() - t0
    joblib.dump({"encoder": args.encoder, "ridge": ridge}, path_for(tag))

    t0 = time.perf_counter()
    Xte = embed(test[col].fillna("").tolist(), args.encoder)  # timed: includes encoding
    infer = (time.perf_counter() - t0) / len(test)
    save_predictions(tag, args.train_text, test, np.clip(ridge.predict(Xte), 0, None))
    if args.train_text == "ocr_clean":
        Xref = cached_embed(test, TEXT_COLUMNS["reference"], args.encoder, "test")
        save_predictions(tag, "reference", test, np.clip(ridge.predict(Xref), 0, None))

    update_run_info(tag, train_s=train_s, infer_s_per_cv=infer, val_mae=best[0],
                    params={"encoder": args.encoder, "alpha": best[1],
                            "long_text": "chunks of max_seq_length tokens, stride 64, mean-pooled",
                            "train_text": args.train_text})
    print(f"[{tag}] alpha={best[1]} val MAE={best[0]:.3f}")


if __name__ == "__main__":
    main()
