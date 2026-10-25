"""Turn a Trainer checkpoint into the final BERT model when training is stopped by hand.

Used when the CPU run was stopped after epoch 6 (validation MAE still improving, but each
epoch took ~30 min on a laptop CPU). Copies the best checkpoint to models/bert/<tag>/,
writes run_config.json, predicts the test set (OCR and reference text) and records the
run in output/metrics/runs.json, exactly as train_bert.main() would have done.

    python -m src.finalize_bert_checkpoint --start 2026-10-25T08:58:47
"""
from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime

from src.train_bert import CVDataset, date_survival, predict
from src.utils import (BERT_DIR, TEXT_COLUMNS, hardware_info, load_json, load_split,
                       save_json, save_predictions, update_run_info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bert")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--start", required=True, help="training start time, ISO format")
    args = ap.parse_args()

    out_dir = BERT_DIR / args.tag
    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    state = load_json(ckpts[-1] / "trainer_state.json")
    best = state.get("best_model_checkpoint") or str(ckpts[-1])
    best_dir = next(p for p in ckpts if p.name == best.replace("\\", "/").rsplit("/", 1)[-1])
    print(f"best checkpoint: {best_dir.name}  val MAE={state['best_metric']:.4f}  epoch={state['epoch']}")

    for f in best_dir.iterdir():
        if f.name.startswith(("config", "model", "tokenizer", "special_tokens", "vocab")):
            shutil.copy2(f, out_dir / f.name)
    train_s = best_dir.stat().st_mtime - datetime.fromisoformat(args.start).timestamp()
    run_cfg = {"model": args.model, "strategy": "head", "loss": "mse", "train_text": "ocr_clean",
               "max_length": args.max_length, "lr": 5e-5, "batch_size": 16, "epochs": 10,
               "patience": 2, "seed": 42, "tag": args.tag,
               "stopped_by_hand_after_epoch": state["epoch"]}
    save_json(run_cfg, out_dir / "run_config.json")

    test = load_split("test")
    col = TEXT_COLUMNS["ocr_clean"]
    t0 = time.perf_counter()
    pred = predict(test[col].fillna(""), out_dir)
    infer = (time.perf_counter() - t0) / len(test)
    save_predictions(args.tag, "ocr_clean", test, pred)
    save_predictions(args.tag, "reference", test, predict(test[TEXT_COLUMNS["reference"]].fillna(""), out_dir))

    from transformers import AutoTokenizer
    ds = CVDataset(test[col].fillna(""), None, AutoTokenizer.from_pretrained(out_dir), "head", args.max_length)
    update_run_info(args.tag, train_s=train_s, infer_s_per_cv=infer, val_mae=state["best_metric"],
                    hardware=hardware_info(), epochs_run=state["epoch"], best_checkpoint=best_dir.name,
                    date_survival=date_survival(test[col].fillna("").tolist(), ds, args.max_length),
                    params={k: v for k, v in run_cfg.items() if k != "tag"})
    print(f"[{args.tag}] val MAE={state['best_metric']:.3f}  train {train_s/60:.0f} min  infer {infer*1000:.0f} ms/CV")


if __name__ == "__main__":
    main()
