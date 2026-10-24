"""Stage 5 — fine-tune a lightweight BERT-family model as a regressor (run on a GPU).

* One output: AutoModelForSequenceClassification(num_labels=1, problem_type="regression").
* Target scaled to [0, 1] (y / CAP) during training; predictions are mapped back to
  years and clipped to [0, CAP].
* Loss: MSE (default) or Huber (`--loss huber`, Q3).
* Long CVs (> max_length tokens), `--strategy`:
    head        first max_length tokens (default; the experience section comes first in
                most CVs — check `date_survival` in output/metrics/runs.json)
    head_tail   first 128 + last (max_length - 130) tokens
    chunk       sliding windows (stride 128); every window is trained with the CV label and
                the CV prediction is the mean over its windows
* Early stopping on validation MAE (in years).

    python -m src.train_bert --model distilbert-base-uncased
    python -m src.train_bert --strategy head_tail          # Q3
    python -m src.train_bert --loss huber                  # Q3
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import (BERT_DIR, CAP, SEED, TEXT_COLUMNS, clip_years, ensure_dirs,
                       hardware_info, load_json, load_split, mae, model_tag, save_json,
                       save_predictions, set_seed, update_run_info)

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


# ======================================================================== encoding

def segments(ids: list[int], strategy: str, max_len: int, stride: int = 128) -> list[list[int]]:
    room = max_len - 2  # [CLS] ... [SEP]
    if len(ids) <= room:
        return [ids]
    if strategy == "head":
        return [ids[:room]]
    if strategy == "head_tail":
        head = min(128, room // 4)
        return [ids[:head] + ids[-(room - head):]]
    if strategy == "chunk":
        stride = min(stride, room // 4)                # overlap between windows
        step = room - stride
        return [ids[i:i + room] for i in range(0, max(len(ids) - stride, 1), step)]
    raise ValueError(strategy)


class CVDataset(Dataset):
    """Tokenized CV segments; `owner[i]` is the CV index of segment i."""

    def __init__(self, texts, labels, tokenizer, strategy, max_len):
        cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.items, owner, self.n_tokens, self.kept_text = [], [], [], []
        for i, text in enumerate(texts):
            ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
            self.n_tokens.append(len(ids))
            segs = segments(ids, strategy, max_len)
            self.kept_text.append(tokenizer.decode([t for s in segs for t in s]))
            for seg in segs:
                item = {"input_ids": [cls_id] + seg + [sep_id]}
                if labels is not None:
                    item["labels"] = float(labels[i]) / CAP
                self.items.append(item)
                owner.append(i)
        self.owner = np.asarray(owner)
        self.n_cvs = len(texts)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def per_cv(self, seg_preds: np.ndarray) -> np.ndarray:
        """Average segment predictions per CV and convert back to years."""
        seg_preds = np.asarray(seg_preds).reshape(-1)
        out = np.array([seg_preds[self.owner == i].mean() for i in range(self.n_cvs)])
        return clip_years(out * CAP)


def date_survival(texts, ds: CVDataset, max_len: int) -> dict:
    """How many of the CV's years survive truncation (are the dates still visible?)."""
    tot = kept = full_cv = 0
    for text, kept_text in zip(texts, ds.kept_text):
        ys = YEAR.findall(str(text))
        ks = YEAR.findall(kept_text)
        tot += len(ys)
        kept += min(len(ks), len(ys))
        full_cv += len(ks) >= len(ys)
    n_tok = np.asarray(ds.n_tokens)
    return {"share_years_kept": round(kept / max(tot, 1), 4),
            "share_cvs_all_years_kept": round(full_cv / len(texts), 4),
            "tokens_median": int(np.median(n_tok)), "tokens_p90": int(np.percentile(n_tok, 90)),
            "share_cvs_over_limit": round(float(np.mean(n_tok > max_len - 2)), 4)}


# ======================================================================== train / predict

def make_trainer_cls(loss: str):
    from transformers import Trainer

    class RegressionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            pred = outputs.logits.squeeze(-1)
            if loss == "huber":
                l = torch.nn.functional.huber_loss(pred, labels.to(pred.dtype), delta=0.1)  # 0.1 = 2 years
            else:
                l = torch.nn.functional.mse_loss(pred, labels.to(pred.dtype))
            return (l, outputs) if return_outputs else l

    return RegressionTrainer


def predict(texts, model_dir=None, batch_size: int = 32) -> np.ndarray:
    """Predict years for raw texts with a saved model (used by the app and the probe)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    model_dir = model_dir or BERT_DIR / "bert"
    cfg = load_json(model_dir / "run_config.json")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ds = CVDataset(list(texts), None, tok, cfg["strategy"], cfg["max_length"])
    collate = DataCollatorWithPadding(tok)
    preds = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = collate([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.append(model(**batch).logits.squeeze(-1).float().cpu().numpy())
    return ds.per_cv(np.concatenate(preds) if preds else np.array([]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased",
                    help="distilbert-base-uncased | answerdotai/ModernBERT-base | "
                         "microsoft/deberta-v3-small | nreimers/MiniLM-L6-H384-uncased")
    ap.add_argument("--strategy", choices=["head", "head_tail", "chunk"], default="head")
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--train-text", choices=list(TEXT_COLUMNS)[:3], default="ocr_clean")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    ensure_dirs(); set_seed(args.seed)

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments)

    name = "bert" + ("" if args.strategy == "head" else f"_{args.strategy}") \
                  + ("" if args.loss == "mse" else "_huber")
    tag = model_tag(name, args.train_text)
    out_dir = BERT_DIR / tag.replace("@", "__")
    col = TEXT_COLUMNS[args.train_text]
    train, val, test = (load_split(s) for s in ("train", "val", "test"))

    tok = AutoTokenizer.from_pretrained(args.model)
    ds_tr = CVDataset(train[col].fillna(""), train["y"].values, tok, args.strategy, args.max_length)
    ds_va = CVDataset(val[col].fillna(""), val["y"].values, tok, args.strategy, args.max_length)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return {"mae": mae(val["y"].values, ds_va.per_cv(preds))}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression",
        ignore_mismatched_sizes=True)  # replaces any existing classification head
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2, num_train_epochs=args.epochs,
        weight_decay=0.01, warmup_steps=0.1, lr_scheduler_type="linear",  # float warmup = ratio of steps
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="mae", greater_is_better=False,
        fp16=torch.cuda.is_available(), seed=args.seed, report_to="none", logging_steps=50,
    )
    trainer = make_trainer_cls(args.loss)(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    run_cfg = {**vars(args), "tag": tag}
    save_json(run_cfg, out_dir / "run_config.json")

    def predict_df(texts):
        ds = CVDataset(texts.fillna(""), None, tok, args.strategy, args.max_length)
        return ds, ds.per_cv(trainer.predict(ds).predictions)

    t0 = time.perf_counter()
    ds_te, pred = predict_df(test[col])
    infer = (time.perf_counter() - t0) / len(test)
    save_predictions(tag, args.train_text, test, pred)
    if args.train_text == "ocr_clean":
        save_predictions(tag, "reference", test, predict_df(test[TEXT_COLUMNS["reference"]])[1])

    val_mae = trainer.evaluate()["eval_mae"]
    update_run_info(tag, train_s=train_s, infer_s_per_cv=infer, val_mae=val_mae,
                    hardware=hardware_info(),
                    epochs_run=round(trainer.state.epoch, 2),
                    best_checkpoint=trainer.state.best_model_checkpoint,
                    date_survival=date_survival(test[col].fillna("").tolist(), ds_te, args.max_length),
                    params={k: v for k, v in run_cfg.items() if k != "tag"})
    print(f"[{tag}] val MAE={val_mae:.3f}  train {train_s/60:.1f} min  infer {infer*1000:.1f} ms/CV")


if __name__ == "__main__":
    main()
