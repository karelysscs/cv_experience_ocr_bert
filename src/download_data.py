"""Stage 0 — stratified sampling of no-photo CV images from PopResume.

Streams one occupation at a time (never downloads the ~20 GB of images), keeps only
`has_photo == False`, shuffles with a fixed seed and saves:

    data/raw/images/<id>.png        one image per sampled CV   (not committed)
    data/raw/metadata.csv           id, occupation, exp_year, sex, race
    data/raw/sampled_ids.txt        reproducibility
    data/raw/reference_text.csv     id, text  -> only for OCR quality (CER/WER) and Q2

Requires `huggingface-cli login` (or HF_TOKEN) after accepting the dataset conditions.

    python -m src.download_data --n-per-occupation 800
"""
from __future__ import annotations

import argparse
import io

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.utils import (CAP, IMAGES, METADATA_CSV, OCCUPATIONS, REFERENCE_CSV,
                       SAMPLED_IDS, SEED, ensure_dirs)

REPO = "sumin-yu/PopResume"
# keyword used to find each occupation's shard files (data/images/<occupation>-NNNN.parquet)
OCC_KEYWORDS = {
    "registered_nurses": "nurse",
    "elementary_middle_school_teachers": "teacher",
    "software_developers": "software",
    "accountants_auditors": "account",
    "construction_laborers": "construction",
}
# metadata kept for evaluation / fairness only — never used as model features
KEEP_META = ["sex", "race"]


def list_shards() -> dict[str, list[str]]:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{REPO}/data/images/*.parquet"))
    shards = {}
    for occ, kw in OCC_KEYWORDS.items():
        shards[occ] = [f"hf://{f}" for f in files if kw in f.rsplit("/", 1)[-1].lower()]
        if not shards[occ]:
            raise RuntimeError(f"No image shards found for {occ!r} (keyword {kw!r}). Files: {files[:10]}")
    return shards


def cv_key(r: dict) -> str:
    """PopResume ids restart in every occupation (00018 exists for several jobs),
    so a CV is identified by job + id, e.g. 'registered_nurses_00020'."""
    return f"{r['job']}_{r['id']}"


def to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict) and img.get("bytes") is not None:
        return Image.open(io.BytesIO(img["bytes"]))
    raise TypeError(f"Unexpected image type: {type(img)}")


def sample_occupation(occ: str, files: list[str], n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    ds = ds.filter(lambda r: not r["has_photo"]).shuffle(seed=seed, buffer_size=5000)
    rows = []
    for r in tqdm(ds.take(n), total=n, desc=occ):
        cv_id = cv_key(r)
        path = IMAGES / f"{cv_id}.png"
        if not path.exists():
            to_pil(r["image"]).convert("RGB").save(path)
        rows.append({"id": cv_id, "occupation": occ, "exp_year": float(r["exp_year"]),
                     **{k: r.get(k) for k in KEEP_META}})
    return rows


def fetch_reference_text(ids: set[str]) -> pd.DataFrame:
    """Reference text of the sampled CVs (streamed; only id + text are kept)."""
    from datasets import load_dataset
    ds = load_dataset(REPO, "text", split="train", streaming=True)
    out = []
    for r in tqdm(ds, desc="reference text"):
        key = cv_key(r)
        if key in ids:
            out.append({"id": key, "text": r["text"]})
            if len(out) == len(ids):
                break
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-occupation", type=int, default=800)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    shards = list_shards()
    rows = []
    for occ in OCCUPATIONS:
        rows += sample_occupation(occ, shards[occ], args.n_per_occupation, args.seed)
    meta = pd.DataFrame(rows)
    meta.to_csv(METADATA_CSV, index=False)
    SAMPLED_IDS.write_text("\n".join(meta["id"]), encoding="utf-8")
    print(f"Saved {len(meta)} CVs -> {METADATA_CSV}")

    at_cap = (meta["exp_year"] >= CAP).mean()
    print(f"Share of CVs with exp_year >= {CAP}: {at_cap:.1%}"
          + ("  -> above 15%: consider raising CAP in src/utils.py and justify it" if at_cap > 0.15 else ""))

    if not args.skip_text:
        ref = fetch_reference_text(set(meta["id"]))
        ref.to_csv(REFERENCE_CSV, index=False)
        print(f"Reference text for {len(ref)}/{len(meta)} CVs -> {REFERENCE_CSV}")


if __name__ == "__main__":
    main()
