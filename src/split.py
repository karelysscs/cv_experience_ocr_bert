"""Stage 3 — one 70/15/15 split by CV id, stratified by occupation x experience band.

Writes data/splits/{train,val,test}.csv with: id, occupation, y (clipped years), band.
Every model reads these files, so all differences come from the model, not the data.

    python -m src.split
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (BAND_LABELS, CAP, METADATA_CSV, METRICS, PROCESSED_CSV, SEED,
                       SPLITS, clip_years, ensure_dirs, save_json, to_band)


def main():
    ensure_dirs()
    meta = pd.read_csv(METADATA_CSV, dtype={"id": str})
    have_text = set(pd.read_csv(PROCESSED_CSV, dtype={"id": str}, usecols=["id"])["id"])
    df = meta[meta["id"].isin(have_text)].copy()
    df["y"] = clip_years(df["exp_year"])
    df["band"] = [BAND_LABELS[b] for b in to_band(df["y"])]
    strata = df["occupation"] + "|" + df["band"]
    # strata too small to appear in all three parts fall back to occupation only
    small = strata.map(strata.value_counts()) < 10
    strata = strata.where(~small, df["occupation"] + "|other")

    train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50,
                                 stratify=strata.loc[rest.index], random_state=SEED)

    cols = ["id", "occupation", "y", "band"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[cols].sort_values("id").to_csv(SPLITS / f"{name}.csv", index=False)

    report = {
        "seed": SEED, "cap": CAP,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "n_at_cap": int((df["exp_year"] >= CAP).sum()),
        "share_at_cap": round(float((df["exp_year"] >= CAP).mean()), 4),
        "band_share": {n: p["band"].value_counts(normalize=True).round(3).to_dict()
                       for n, p in [("train", train), ("val", val), ("test", test)]},
    }
    save_json(report, METRICS / "split.json")
    print(report)


if __name__ == "__main__":
    main()
