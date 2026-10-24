"""Optional bonus — rule-based date arithmetic.

Extracts job date ranges with regular expressions ("Mar 2014 - Jun 2019",
"2014 - Present", "05/2016 - 08/2020"), keeps only those in work-experience sections
when the CV has such headings, merges overlapping intervals and adds up their length.
"Present" is resolved to a reference year tuned on the training set. CVs with no
readable range fall back to the training median.

It is an honest test of whether BERT does better than simple arithmetic.

    python -m src.baseline_rules
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np

from src.utils import (BASELINES_DIR, TEXT_COLUMNS, ensure_dirs, load_json, load_split,
                       mae, model_tag, save_json, save_predictions, update_run_info)

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"


def _point(p: str) -> str:
    """A date: optional month name or MM/ before a 4-digit year; groups <p>m, <p>n, <p>y."""
    return rf"(?:(?P<{p}m>{MON})\s+|(?P<{p}n>\d{{1,2}})\s*/\s*)?(?P<{p}y>(?:19|20)\d{{2}})"


END = rf"(?:{_point('e')}|(?P<present>present|current|now|today))"
RANGE = re.compile(rf"{_point('s')}\s*(?:-|to|until)\s*{END}", re.I)

WORK_HEAD = re.compile(r"^\W*(?:professional\s+|work\s+|employment\s+|relevant\s+)?"
                       r"(?:experience|employment|work history|career history)\W*$", re.I)
OTHER_HEAD = re.compile(r"^\W*(?:education|skills|certifications?|licenses?|summary|profile|"
                        r"projects|awards|languages|references|volunteer\w*|interests)\W*$", re.I)
PATH = BASELINES_DIR / "rules.json"


def _month(m, n, default):
    if m:
        return MONTHS.get(m[:3].lower(), default)
    if n and 1 <= int(n) <= 12:
        return int(n)
    return default


def work_section(text: str) -> str:
    """Text inside work-experience sections, or the whole CV if no heading is found."""
    keep, inside, found = [], False, False
    for line in str(text).splitlines():
        if WORK_HEAD.match(line):
            inside, found = True, True
            continue
        if OTHER_HEAD.match(line):
            inside = False
        if inside:
            keep.append(line)
    return "\n".join(keep) if found else str(text)


def intervals(text: str, present_year: int) -> list[tuple[float, float]]:
    out = []
    for m in RANGE.finditer(work_section(text)):
        s = int(m["sy"]) + (_month(m["sm"], m["sn"], 1) - 1) / 12
        if m["present"]:
            e = present_year + 0.5
        else:
            e = int(m["ey"]) + _month(m["em"], m["en"], 12) / 12
        if 0 < e - s < 45:
            out.append((s, e))
    return out


def total_years(text: str, present_year: int) -> float | None:
    iv = sorted(intervals(text, present_year))
    if not iv:
        return None
    total, (cs, ce) = 0.0, iv[0]
    for s, e in iv[1:]:
        if s <= ce:                     # overlapping jobs count once
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + ce - cs


def predict(texts, present_year: int | None = None, fallback: float | None = None) -> np.ndarray:
    if present_year is None:
        p = load_json(PATH)
        present_year, fallback = p["present_year"], p["fallback"]
    preds = [total_years(t, present_year) for t in texts]
    return np.array([fallback if v is None else v for v in preds], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-text", choices=list(TEXT_COLUMNS)[:3], default="ocr_clean")
    args = ap.parse_args()
    ensure_dirs()
    tag = model_tag("rules", args.train_text)
    col = TEXT_COLUMNS[args.train_text]
    train, val, test = (load_split(s) for s in ("train", "val", "test"))

    t0 = time.perf_counter()
    fallback = float(np.median(train["y"]))
    present = min(range(2015, 2031), key=lambda yr: mae(train["y"], predict(train[col], yr, fallback)))
    train_s = time.perf_counter() - t0
    save_json({"present_year": present, "fallback": fallback}, PATH)
    coverage = float(np.mean([total_years(t, present) is not None for t in test[col]]))

    t0 = time.perf_counter()
    save_predictions(tag, args.train_text, test, predict(test[col], present, fallback))
    infer = (time.perf_counter() - t0) / len(test)
    if args.train_text == "ocr_clean":
        save_predictions(tag, "reference", test, predict(test[TEXT_COLUMNS["reference"]], present, fallback))

    val_mae = mae(val["y"], predict(val[col], present, fallback))
    update_run_info(tag, train_s=train_s, infer_s_per_cv=infer, val_mae=val_mae,
                    params={"present_year": present, "fallback_median": fallback,
                            "test_coverage": coverage})
    print(f"[{tag}] present_year={present} coverage={coverage:.1%} val MAE={val_mae:.3f}")


if __name__ == "__main__":
    main()
