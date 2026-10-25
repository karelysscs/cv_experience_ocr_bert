"""Question 1 probe — do the models read dates or just vocabulary?

Takes 20 test CVs and applies two edits:
    shift_+5   every year in the CV moves +5 (durations stay the same, except jobs that end
               in "Present", which become 5 years shorter)
    drop_job   the first job date range (and its line) is deleted

The "recruiter" expectation is the change computed by date arithmetic (baseline_rules).
For each model we report how its prediction moves and whether it moves the right way.

    python -m src.probe            (run after every model is trained; BERT is fast enough on CPU)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import baseline_embeddings, baseline_median, baseline_rules, baseline_tfidf, train_bert
from src.utils import BERT_DIR, SEED, TABLES, TEXT_COLUMNS, ensure_dirs, load_json, load_split

YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def shift_years(text: str, k: int = 5) -> str:
    return YEAR.sub(lambda m: str(int(m.group(1)) + k), text)


def drop_first_job(text: str) -> str:
    lines = str(text).splitlines()
    for i, line in enumerate(lines):
        if baseline_rules.RANGE.search(line):
            return "\n".join(lines[:i] + lines[i + 1:])
    return text


def predictors() -> dict:
    out = {"median": baseline_median.predict, "tfidf": baseline_tfidf.predict,
           "embeddings": baseline_embeddings.predict, "rules": baseline_rules.predict}
    if (BERT_DIR / "bert" / "run_config.json").exists():
        out["bert"] = lambda t: train_bert.predict(t, BERT_DIR / "bert")
    return out


def main(n: int = 20):
    ensure_dirs()
    test = load_split("test")
    col = TEXT_COLUMNS["ocr_clean"]
    # only CVs with at least one readable date range, so both edits do something
    has_range = test[col].fillna("").map(lambda t: bool(baseline_rules.RANGE.search(t)))
    sample = test[has_range].sample(n=min(n, has_range.sum()), random_state=SEED)
    orig = sample[col].fillna("").tolist()
    edits = {"shift_+5": [shift_years(t) for t in orig], "drop_job": [drop_first_job(t) for t in orig]}

    rules_cfg = load_json(baseline_rules.PATH)
    def recruiter(texts):
        return baseline_rules.predict(texts, rules_cfg["present_year"], rules_cfg["fallback"])

    rows, summary = [], []
    base_expect = recruiter(orig)
    for name, fn in predictors().items():
        p0 = fn(orig)
        for edit, texts in edits.items():
            p1 = fn(texts)
            expected = recruiter(texts) - base_expect
            moved = p1 - p0
            right_dir = np.where(np.abs(expected) < 0.5, np.abs(moved) < 0.5,
                                 np.sign(moved) == np.sign(expected))
            for i, cv_id in enumerate(sample["id"]):
                rows.append({"model": name, "edit": edit, "id": cv_id, "pred_before": p0[i],
                             "pred_after": p1[i], "change": moved[i], "expected_change": expected[i]})
            summary.append({
                "model": name, "edit": edit,
                "mean_change": moved.mean(), "mean_expected_change": expected.mean(),
                "mean_abs_change": np.abs(moved).mean(),
                "share_right_direction": right_dir.mean(),
                "corr_with_expected": np.corrcoef(moved, expected)[0, 1]
                if np.std(moved) > 0 and np.std(expected) > 0 else np.nan,
            })
    pd.DataFrame(rows).round(3).to_csv(TABLES / "probe_details.csv", index=False)
    s = pd.DataFrame(summary).round(3)
    s.to_csv(TABLES / "probe_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
