"""Stage 2a — OCR quality: CER, WER and how well years/dates are read.

Both texts (OCR in rebuilt reading order, and the reference `text`) are normalized the
same way before scoring: lower-case, unified dashes, punctuation spacing, single spaces.

Outputs
    output/tables/ocr_quality_per_cv.csv
    output/tables/ocr_quality_by_occupation.csv
    output/metrics/ocr_quality.json

    python -m src.ocr_quality
"""
from __future__ import annotations

import re
from collections import Counter

import jiwer
import pandas as pd

from src.preprocessing import normalize_text, rebuild_reading_order
from src.utils import (METADATA_CSV, METRICS, OCR_DIR, REFERENCE_CSV, TABLES,
                       ensure_dirs, load_json, save_json)

YEAR = re.compile(r"\b(?:19[5-9]\d|20[0-4]\d)\b")
RANGE = re.compile(r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2}|present|current|now)\b", re.I)


def norm(s: str) -> str:
    s = normalize_text(str(s)).lower()
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"([,;:()|•·])", r" \1 ", s)
    return re.sub(r"\s+", " ", s).strip()


def multiset_recall(ref_items: list, hyp_items: list) -> tuple[int, int]:
    ref, hyp = Counter(ref_items), Counter(hyp_items)
    hit = sum(min(c, hyp[k]) for k, c in ref.items())
    return hit, sum(ref.values())


def score_cv(ocr_text: str, ref_text: str) -> dict:
    o, r = norm(ocr_text), norm(ref_text)
    y_hit, y_tot = multiset_recall(YEAR.findall(r), YEAR.findall(o))
    r_hit, r_tot = multiset_recall([tuple(m) for m in RANGE.findall(r)],
                                   [tuple(m) for m in RANGE.findall(o)])
    return {
        "cer": jiwer.cer(r, o),
        "wer": jiwer.wer(r, o),
        "years_ref": y_tot, "years_hit": y_hit,
        "ranges_ref": r_tot, "ranges_hit": r_hit,
    }


def main():
    ensure_dirs()
    meta = pd.read_csv(METADATA_CSV, dtype={"id": str}).set_index("id")
    ref = pd.read_csv(REFERENCE_CSV, dtype={"id": str}).set_index("id")["text"]

    rows = []
    for cv_id in meta.index:
        p = OCR_DIR / f"{cv_id}.json"
        if not p.exists() or cv_id not in ref.index:
            continue
        rec = load_json(p)
        ocr_text = "\n".join(rebuild_reading_order(rec["lines"], rec.get("image_size", [None])[0]))
        rows.append({"id": cv_id, "occupation": meta.at[cv_id, "occupation"],
                     "mean_conf": rec.get("mean_conf"), **score_cv(ocr_text, ref[cv_id])})
    df = pd.DataFrame(rows)
    df["year_recall"] = df["years_hit"] / df["years_ref"].where(df["years_ref"] > 0)
    df.to_csv(TABLES / "ocr_quality_per_cv.csv", index=False)

    by_occ = df.groupby("occupation").agg(
        n=("id", "size"), cer_mean=("cer", "mean"), cer_median=("cer", "median"),
        wer_mean=("wer", "mean"), wer_median=("wer", "median"),
        years_ref=("years_ref", "sum"), years_hit=("years_hit", "sum"),
        ranges_ref=("ranges_ref", "sum"), ranges_hit=("ranges_hit", "sum"),
        mean_conf=("mean_conf", "mean"),
    )
    by_occ["share_years_correct"] = by_occ["years_hit"] / by_occ["years_ref"]
    by_occ["share_ranges_correct"] = by_occ["ranges_hit"] / by_occ["ranges_ref"]
    by_occ.round(4).to_csv(TABLES / "ocr_quality_by_occupation.csv")

    summary = {
        "n": len(df),
        "cer_mean": df["cer"].mean(), "cer_median": df["cer"].median(),
        "wer_mean": df["wer"].mean(), "wer_median": df["wer"].median(),
        "share_years_correct": df["years_hit"].sum() / max(df["years_ref"].sum(), 1),
        "share_date_ranges_correct": df["ranges_hit"].sum() / max(df["ranges_ref"].sum(), 1),
        "share_cvs_all_years_correct": float((df["years_hit"] == df["years_ref"]).mean()),
        "worst_10_ids_by_cer": df.nlargest(10, "cer")["id"].tolist(),
    }
    save_json({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()},
              METRICS / "ocr_quality.json")
    print(by_occ.round(3))
    print(summary)


if __name__ == "__main__":
    main()
