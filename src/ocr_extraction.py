"""Stage 1 — PaddleOCR on every sampled image, with per-CV caching.

Writes one JSON per CV to data/ocr/<id>.json:
    {"id", "model", "device", "image_size", "time_s", "mean_conf",
     "lines": [{"text", "score", "box": [x1, y1, x2, y2]}, ...]}

Already-processed CVs are skipped, so the script can be re-run after a Colab/Kaggle
disconnect and it resumes where it stopped. Run it on a GPU:

    python -m src.ocr_extraction --model server --device gpu
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.utils import (IMAGES, METADATA_CSV, METRICS, OCR_DIR, ensure_dirs,
                       hardware_info, load_json, save_json)

MODELS = {
    # server: higher accuracy on small fonts and digits (dates matter most here); needs a GPU.
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    # mobile: ~3-5x faster, good enough for the CPU demo in the Streamlit app.
    "mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
}


def build_ocr(model: str = "server", device: str | None = None):
    from paddleocr import PaddleOCR
    det, rec = MODELS[model]
    kwargs = dict(
        lang="en",
        text_detection_model_name=det,
        text_recognition_model_name=rec,
        # CV images are upright, flat, digital renders: no orientation / unwarping needed
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if device:
        kwargs["device"] = device
    return PaddleOCR(**kwargs)


def ocr_image(ocr, image_path: str | Path) -> list[dict]:
    """Run OCR on one image and return [{text, score, box}] (box = x1, y1, x2, y2)."""
    res = ocr.predict(str(image_path))[0]
    texts, scores, boxes = res["rec_texts"], res["rec_scores"], res["rec_boxes"]
    return [
        {"text": t, "score": round(float(s), 4), "box": [int(v) for v in b]}
        for t, s, b in zip(texts, scores, boxes)
        if str(t).strip()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="server")
    ap.add_argument("--device", default=None, help="gpu, gpu:0 or cpu (default: Paddle decides)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N (testing)")
    args = ap.parse_args()
    ensure_dirs()

    ids = pd.read_csv(METADATA_CSV, dtype={"id": str})["id"].tolist()[: args.limit]
    ocr = build_ocr(args.model, args.device)
    failed = []

    for cv_id in tqdm(ids, desc="PaddleOCR"):
        out = OCR_DIR / f"{cv_id}.json"
        if out.exists():
            continue
        img = IMAGES / f"{cv_id}.png"
        try:
            t0 = time.perf_counter()
            lines = ocr_image(ocr, img)
            dt = time.perf_counter() - t0
            save_json({
                "id": cv_id,
                "model": MODELS[args.model][1],
                "device": args.device or "auto",
                "image_size": Image.open(img).size,
                "time_s": round(dt, 3),
                "mean_conf": round(statistics.fmean(l["score"] for l in lines), 4) if lines else None,
                "lines": lines,
            }, out)
        except Exception as e:  # keep going; failures are reported
            failed.append({"id": cv_id, "error": repr(e)})

    summarize(failed, args.model)


def summarize(failed: list[dict] | None = None, model: str = "server"):
    """Aggregate time / confidence over all cached JSONs -> output/metrics/ocr_summary.json."""
    recs = [load_json(p) for p in OCR_DIR.glob("*.json")]
    times = [r["time_s"] for r in recs]
    confs = [r["mean_conf"] for r in recs if r["mean_conf"] is not None]
    empty = [r["id"] for r in recs if not r["lines"]]
    summary = {
        "n_processed": len(recs),
        "model": MODELS[model][1],
        "hardware": hardware_info(),
        "time_per_cv_s_mean": round(statistics.fmean(times), 3) if times else None,
        "time_per_cv_s_median": round(statistics.median(times), 3) if times else None,
        "mean_confidence": round(statistics.fmean(confs), 4) if confs else None,
        "n_empty_output": len(empty),
        "failed": (failed or []) + [{"id": i, "error": "no text detected"} for i in empty],
    }
    save_json(summary, METRICS / "ocr_summary.json")
    print({k: v for k, v in summary.items() if k != "failed"}, f"failed={len(summary['failed'])}")
    return summary


if __name__ == "__main__":
    main()
