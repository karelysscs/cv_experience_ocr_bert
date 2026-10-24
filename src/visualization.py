"""Figures for the README, the video and the Streamlit app.

Each `fig_*` function returns a matplotlib Figure (the app reuses them);
`main()` saves all of them to output/figures/.

    python -m src.visualization
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.utils import (BAND_LABELS, CAP, FIGURES, IMAGES, METADATA_CSV, METRICS, OCR_DIR,
                       PREDICTIONS, TABLES, clip_years, ensure_dirs, load_json)

sns.set_theme(style="whitegrid", context="notebook")
MODEL_LABELS = {"median": "Median", "tfidf": "TF-IDF + Ridge", "embeddings": "Embeddings + Ridge",
                "rules": "Rules (bonus)", "bert": "Fine-tuned BERT"}


def fig_experience_distribution(meta: pd.DataFrame):
    y = clip_years(meta["exp_year"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    sns.histplot(y, bins=np.arange(-0.5, CAP + 1.5, 1), ax=axes[0], color="#4C72B0")
    axes[0].set(title=f"Years of experience (clipped at {CAP})", xlabel="years", ylabel="CVs")
    at_cap = (meta["exp_year"] >= CAP).mean()
    axes[0].annotate(f"{at_cap:.1%} at cap", xy=(CAP, 0), xytext=(CAP - 6, axes[0].get_ylim()[1] * 0.85))
    sns.boxplot(x=y, y=meta["occupation"].str.replace("_", " "), ax=axes[1], color="#DD8452")
    axes[1].set(title="By occupation", xlabel="years", ylabel="")
    fig.tight_layout()
    return fig


def fig_ocr_quality(per_cv: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(per_cv["cer"].clip(upper=0.5), bins=40, ax=axes[0])
    axes[0].set(title="Character error rate per CV", xlabel="CER (clipped at 0.5)")
    sns.histplot(per_cv["wer"].clip(upper=0.8), bins=40, ax=axes[1], color="#55A868")
    axes[1].set(title="Word error rate per CV", xlabel="WER (clipped at 0.8)")
    g = per_cv.groupby("occupation")[["years_hit", "years_ref"]].sum()
    share = (g["years_hit"] / g["years_ref"]).sort_values()
    axes[2].barh(share.index.str.replace("_", " "), share.values, color="#C44E52")
    axes[2].set(title="Share of reference years read correctly", xlim=(min(0.8, share.min() - 0.02), 1))
    fig.tight_layout()
    return fig


def fig_ocr_boxes(cv_id: str, min_conf: float = 0.80):
    rec = load_json(OCR_DIR / f"{cv_id}.json")
    img = Image.open(IMAGES / f"{cv_id}.png").convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 8 * img.height / img.width))
    ax.imshow(img)
    for l in rec["lines"]:
        x1, y1, x2, y2 = l["box"]
        color = "#2ca02c" if l["score"] >= min_conf else "#d62728"
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=0.8, ec=color))
    ax.set_axis_off()
    ax.set_title(f"PaddleOCR boxes (red = score < {min_conf})")
    fig.tight_layout()
    return fig


def load_main_preds(text_source: str = "ocr_clean") -> dict[str, pd.DataFrame]:
    out = {}
    for key in MODEL_LABELS:
        f = PREDICTIONS / f"{key}__{text_source}.csv"
        if f.exists():
            out[key] = pd.read_csv(f, dtype={"id": str})
    return out


def fig_pred_vs_true(preds: dict[str, pd.DataFrame]):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for ax, (key, df) in zip(axes[0], preds.items()):
        jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(df))
        ax.scatter(df["y_true"] + jitter, df["y_pred"], s=8, alpha=0.35)
        ax.plot([0, CAP], [0, CAP], "k--", lw=1, label="45° line")
        slope, icpt = np.polyfit(df["y_true"], df["y_pred"], 1) if df["y_pred"].std() > 0 else (0, df["y_pred"].mean())
        xs = np.array([0, CAP])
        ax.plot(xs, icpt + slope * xs, color="#C44E52", lw=1.5, label=f"fit (slope {slope:.2f})")
        mae = (df["y_pred"] - df["y_true"]).abs().mean()
        ax.set(title=f"{MODEL_LABELS.get(key, key)}\nMAE {mae:.2f}", xlabel="true years",
               ylabel="predicted years", xlim=(-1, CAP + 1), ylim=(-1, CAP + 1))
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def fig_mae_by(table: pd.DataFrame, by: str, order=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    t = table.copy()
    t[by] = t[by].astype(str).str.replace("_", " ")
    sns.barplot(data=t, x=by, y="MAE", hue="Model", order=order, ax=ax)
    ax.set(title=f"MAE by {by}", xlabel="", ylabel="MAE (years)")
    ax.legend(fontsize=8, ncols=2)
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    return fig


def fig_confusion(cm, title: str):
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=BAND_LABELS, yticklabels=BAND_LABELS, ax=ax)
    ax.set(title=title, xlabel="predicted band", ylabel="true band")
    fig.tight_layout()
    return fig


def fig_bert_gain(gain: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(gain))
    ax.errorbar(gain["gain_mae"], y, xerr=[gain["gain_mae"] - gain["ci_low"], gain["ci_high"] - gain["gain_mae"]],
                fmt="o", capsize=4)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_yticks(y, gain["baseline"])
    ax.set(title="BERT MAE reduction vs each baseline (95% paired bootstrap CI)", xlabel="years saved per CV")
    fig.tight_layout()
    return fig


def main():
    ensure_dirs()
    def save(fig, name):
        fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", name)

    if METADATA_CSV.exists():
        save(fig_experience_distribution(pd.read_csv(METADATA_CSV)), "experience_distribution.png")
    if (TABLES / "ocr_quality_per_cv.csv").exists():
        per_cv = pd.read_csv(TABLES / "ocr_quality_per_cv.csv", dtype={"id": str})
        save(fig_ocr_quality(per_cv), "ocr_quality.png")
        example = per_cv.sort_values("cer").iloc[len(per_cv) // 2]["id"]   # a typical CV
        if (IMAGES / f"{example}.png").exists():
            save(fig_ocr_boxes(example), "ocr_boxes_example.png")
    preds = load_main_preds()
    if preds:
        save(fig_pred_vs_true(preds), "pred_vs_true.png")
    if (TABLES / "mae_by_band.csv").exists():
        save(fig_mae_by(pd.read_csv(TABLES / "mae_by_band.csv"), "band", BAND_LABELS), "mae_by_band.png")
        save(fig_mae_by(pd.read_csv(TABLES / "mae_by_occupation.csv"), "occupation"), "mae_by_occupation.png")
    if (METRICS / "confusion_matrices.json").exists():
        for key, cm in load_json(METRICS / "confusion_matrices.json")["matrices"].items():
            save(fig_confusion(cm, MODEL_LABELS.get(key, key)), f"confusion_{key}.png")
    if (TABLES / "bert_gain_ci.csv").exists():
        save(fig_bert_gain(pd.read_csv(TABLES / "bert_gain_ci.csv")), "bert_gain_ci.png")


if __name__ == "__main__":
    main()
