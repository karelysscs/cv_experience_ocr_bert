"""Stage 6 — evaluation of every model on the same test set.

Reads output/predictions/<model>__<text_source>.csv and writes:

    output/tables/model_comparison.csv     main table (Stage 6)
    output/tables/bert_gain_ci.csv         BERT MAE gain vs each baseline, 95% paired bootstrap CI
    output/tables/mae_by_band.csv          MAE by true experience band
    output/tables/mae_by_occupation.csv
    output/tables/ocr_vs_reference.csv     Q2: OCR text vs clean reference text
    output/tables/sensitivity.csv          Q3: every alternative setup vs the main one
    output/tables/fairness.csv             Q4: MAE and mean error by sex and race
    output/tables/bert_top10_errors.csv    to label by cause (column `cause`)
    output/metrics/confusion_matrices.json

    python -m src.evaluation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, r2_score

from src.utils import (BAND_LABELS, METADATA_CSV, METRICS, PREDICTIONS, PROCESSED_CSV,
                       SEED, TABLES, ensure_dirs, load_json, save_json, to_band)

MAIN_MODELS = {"median": "Median", "tfidf": "TF-IDF + Ridge",
               "embeddings": "Embeddings + Ridge", "rules": "Rule-based dates (bonus)",
               "bert": "Fine-tuned BERT"}
MAIN_TEXT = "ocr_clean"


# ======================================================================== metrics

def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(p - y)
    yb, pb = to_band(y), to_band(p)
    slope = np.polyfit(y, p, 1)[0] if np.std(y) > 0 and np.std(p) > 0 else 0.0
    return {
        "MAE (years)": err.mean(),
        "RMSE (years)": np.sqrt(((p - y) ** 2).mean()),
        "Acc ±1": (err <= 1).mean(), "Acc ±2": (err <= 2).mean(), "Acc ±3": (err <= 3).mean(),
        "Band accuracy": (yb == pb).mean(),
        "Band macro-F1": f1_score(yb, pb, average="macro", labels=range(len(BAND_LABELS)), zero_division=0),
        "R²": r2_score(y, p),
        "Mean error (pred-true)": (p - y).mean(),
        "Slope pred~true": slope,       # < 1 means predictions are pulled toward the middle
    }


def paired_bootstrap(y, p_base, p_bert, n: int = 2000, seed: int = SEED) -> dict:
    """MAE(baseline) - MAE(BERT) with a 95% percentile CI from paired resampling of CVs."""
    rng = np.random.default_rng(seed)
    y, a, b = (np.asarray(v, float) for v in (y, p_base, p_bert))
    ea, eb = np.abs(a - y), np.abs(b - y)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diffs = ea[idx].mean(1) - eb[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"gain_mae": ea.mean() - eb.mean(), "ci_low": lo, "ci_high": hi,
            "p_gain_le_0": float((diffs <= 0).mean()),
            "significant": bool(lo > 0 or hi < 0), "n_resamples": n}


# ======================================================================== IO

def load_preds() -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for f in PREDICTIONS.glob("*__*.csv"):
        model, src = f.stem.split("__", 1)
        out[(model, src)] = pd.read_csv(f, dtype={"id": str})
    return out


def main():
    ensure_dirs()
    preds = load_preds()
    runs = load_json(METRICS / "runs.json") if (METRICS / "runs.json").exists() else {}
    meta = pd.read_csv(METADATA_CSV, dtype={"id": str})

    # ---------------------------------------------------------------- main table
    rows = []
    for m, label in MAIN_MODELS.items():
        df = preds.get((m, MAIN_TEXT))
        if df is None:
            continue
        r = {"Model": label, "key": m, **metrics(df["y_true"], df["y_pred"])}
        r["Training time (s)"] = runs.get(m, {}).get("train_s")
        r["Inference time per CV (ms)"] = (runs.get(m, {}).get("infer_s_per_cv") or np.nan) * 1000
        rows.append(r)
    comp = pd.DataFrame(rows)
    comp.round(4).to_csv(TABLES / "model_comparison.csv", index=False)
    print(comp.drop(columns="key").round(3).to_string(index=False))

    # ---------------------------------------------------------------- BERT gain with CI
    bert = preds.get(("bert", MAIN_TEXT))
    if bert is not None:
        gains = []
        for m in MAIN_MODELS:
            if m == "bert" or (m, MAIN_TEXT) not in preds:
                continue
            base = preds[(m, MAIN_TEXT)].set_index("id").loc[bert["id"]]
            gains.append({"baseline": MAIN_MODELS[m], "key": m,
                          **paired_bootstrap(bert["y_true"], base["y_pred"], bert["y_pred"])})
        pd.DataFrame(gains).round(4).to_csv(TABLES / "bert_gain_ci.csv", index=False)

    # ---------------------------------------------------------------- by band / occupation
    by_band, by_occ, cms = [], [], {}
    for m, label in MAIN_MODELS.items():
        df = preds.get((m, MAIN_TEXT))
        if df is None:
            continue
        df = df.assign(err=(df["y_pred"] - df["y_true"]).abs(), signed=df["y_pred"] - df["y_true"],
                       band=[BAND_LABELS[b] for b in to_band(df["y_true"])])
        for b, g in df.groupby("band"):
            by_band.append({"Model": label, "band": b, "n": len(g), "MAE": g["err"].mean(),
                            "Mean error": g["signed"].mean()})
        for o, g in df.groupby("occupation"):
            by_occ.append({"Model": label, "occupation": o, "n": len(g), "MAE": g["err"].mean()})
        cms[m] = confusion_matrix(to_band(df["y_true"]), to_band(df["y_pred"]),
                                  labels=range(len(BAND_LABELS))).tolist()
    pd.DataFrame(by_band).round(4).to_csv(TABLES / "mae_by_band.csv", index=False)
    pd.DataFrame(by_occ).round(4).to_csv(TABLES / "mae_by_occupation.csv", index=False)
    save_json({"labels": BAND_LABELS, "matrices": cms}, METRICS / "confusion_matrices.json")

    # ---------------------------------------------------------------- Q2 OCR vs reference
    q2 = []
    for m, label in MAIN_MODELS.items():
        a, b = preds.get((m, MAIN_TEXT)), preds.get((m, "reference"))
        if a is None or b is None:
            continue
        mo, mr = metrics(a["y_true"], a["y_pred"]), metrics(b["y_true"], b["y_pred"])
        q2.append({"Model": label, "MAE OCR": mo["MAE (years)"], "MAE reference": mr["MAE (years)"],
                   "Gap (OCR - ref)": mo["MAE (years)"] - mr["MAE (years)"],
                   "Acc±2 OCR": mo["Acc ±2"], "Acc±2 reference": mr["Acc ±2"]})
    pd.DataFrame(q2).round(4).to_csv(TABLES / "ocr_vs_reference.csv", index=False)

    # ---------------------------------------------------------------- Q3 sensitivity
    q3 = []
    for (m, src), df in preds.items():
        if src == "reference":
            continue
        base_key = m.split("@")[0].split("_")[0]          # bert_huber@ocr_raw -> bert
        main = preds.get((base_key, MAIN_TEXT))
        if main is None or base_key == "median":
            continue
        mae_v = metrics(df["y_true"], df["y_pred"])["MAE (years)"]
        mae_m = metrics(main["y_true"], main["y_pred"])["MAE (years)"]
        q3.append({"model": MAIN_MODELS.get(base_key, base_key), "setup": f"{m} | text={src}",
                   "MAE": mae_v, "MAE main setup": mae_m, "Δ vs main": mae_v - mae_m})
    pd.DataFrame(q3).sort_values(["model", "setup"]).round(4).to_csv(TABLES / "sensitivity.csv", index=False)

    # ---------------------------------------------------------------- Q4 fairness
    fair = []
    for m, label in MAIN_MODELS.items():
        df = preds.get((m, MAIN_TEXT))
        if df is None:
            continue
        df = df.merge(meta[["id", "sex", "race"]], on="id", how="left")
        df["signed"] = df["y_pred"] - df["y_true"]
        for attr in ["sex", "race"]:
            for g, gdf in df.groupby(attr):
                fair.append({"Model": label, "attribute": attr, "group": g, "n": len(gdf),
                             "MAE": gdf["signed"].abs().mean(), "Mean error (pred-true)": gdf["signed"].mean(),
                             "Mean true years": gdf["y_true"].mean()})
    pd.DataFrame(fair).round(4).to_csv(TABLES / "fairness.csv", index=False)

    # ---------------------------------------------------------------- top-10 BERT errors
    if bert is not None:
        text = pd.read_csv(PROCESSED_CSV, dtype={"id": str})[["id", "text_ocr_clean", "mean_conf"]]
        top = bert.assign(abs_err=(bert["y_pred"] - bert["y_true"]).abs()) \
                  .nlargest(10, "abs_err").merge(text, on="id", how="left")
        q = TABLES / "ocr_quality_per_cv.csv"
        if q.exists():
            top = top.merge(pd.read_csv(q, dtype={"id": str})[["id", "cer", "years_ref", "years_hit"]],
                            on="id", how="left")
        top["n_tokens_approx"] = top["text_ocr_clean"].str.split().str.len()
        top["cause"] = ""   # fill by hand: ocr_date | reading_order | career_gap | overlap | missing_dates | truncation | other
        top.to_csv(TABLES / "bert_top10_errors.csv", index=False)


if __name__ == "__main__":
    main()
