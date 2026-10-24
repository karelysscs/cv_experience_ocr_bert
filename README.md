# CV Experience Estimator — PaddleOCR + BERT

> **Status:** pipeline implemented; numbers marked `TODO` are filled in after running the
> notebooks (they are produced in `output/tables` and `output/metrics`).

## 1. What does the project do, and for whom?

The system reads a **CV image**, extracts its text with **PaddleOCR**, and predicts the candidate's
**years of work experience (0–20)** with a fine-tuned **BERT regressor**. It is compared against
three cheaper models (median, TF-IDF + Ridge, sentence embeddings + Ridge) and a bonus
rule-based date counter, all on the same split.

Potential users: recruiting agencies, job boards, public employment services and HR teams that
receive thousands of CVs and want to **pre-fill a candidate profile** or **sort CVs for reading**.
It must never decide who is interviewed or hired (see §11).

## 2. Repository layout

```
app.py                     Streamlit app (4 tabs)
notebooks/00_colab_full_pipeline.ipynb  Colab: the whole pipeline in one notebook
notebooks/01_ocr_gpu.ipynb     Kaggle/Colab: sampling + PaddleOCR (GPU) + cleaning + split
notebooks/02_models_gpu.ipynb  Kaggle/Colab: baselines + BERT (GPU) + evaluation
src/download_data.py       stratified sampling from PopResume (streaming)
src/ocr_extraction.py      PaddleOCR, one cached JSON per CV, resumable
src/ocr_quality.py         CER / WER / years read correctly
src/preprocessing.py       reading order, cleaning, PII removal, masking
src/split.py               70/15/15 stratified split
src/baseline_median.py     Baseline 0
src/baseline_tfidf.py      Baseline 1
src/baseline_embeddings.py Baseline 2
src/baseline_rules.py      bonus: date arithmetic
src/train_bert.py          BERT regression fine-tuning
src/probe.py               Q1 probe (shift dates / delete a job)
src/evaluation.py          all metrics, bootstrap CIs, Q2–Q4 tables
src/visualization.py       figures
src/utils.py               paths, seeds, constants
```

`data/raw/images/` and `models/` are git-ignored (see §12 to regenerate them).

## 3. Sampling and target

* Dataset: [PopResume](https://huggingface.co/datasets/sumin-yu/PopResume) (Yu, Park & Moon, EMNLP 2026, CC BY 4.0).
* **800 CVs per occupation × 5 = 4,000**, only `has_photo == False`, streamed per occupation and
  shuffled with **seed 42** (`buffer_size=5000`). IDs: `data/raw/sampled_ids.txt`.
* Target `exp_year` clipped to **0–20**. CVs at the cap: **TODO %** (`output/metrics/split.json`).
  *(If > 15%, justify raising the cap in `src/utils.py`.)*
* Distribution: `output/figures/experience_distribution.png` — TODO one sentence.

## 4. OCR

* PaddleOCR **3.7.0** on PaddlePaddle **3.3.1** (GPU build), model **`PP-OCRv5_server_det` +
  `PP-OCRv5_server_rec`**: the server recognizer is more accurate on small digits than the mobile one,
  and dates are the evidence the models need. Orientation classification, unwarping and text-line
  orientation are disabled because the images are upright digital renders.
* For every line we store text, confidence and box (`data/ocr/<id>.json`).
* **Reading order** (`preprocessing.rebuild_reading_order`): boxes are grouped into rows by vertical
  overlap and each row is read left → right. A **two-column layout** is detected when an x position
  between 20% and 80% of the page width is crossed by ≤ 5% of the boxes while both sides hold ≥ 15%
  of them. Then the header (everything above where both columns start) is read first, then the left
  column, then the right column — so a job title is never separated from its dates.
* Hardware: TODO · time per CV: TODO s · mean confidence: TODO · failed images: TODO
  (`output/metrics/ocr_summary.json`).

## 5. OCR quality

Both texts are normalized the same way (lower-case, unified dashes, spaced punctuation, single spaces).

| Occupation | CER | WER | Years read correctly | Date ranges read correctly |
|---|---|---|---|---|
| TODO (from `output/tables/ocr_quality_by_occupation.csv`) | | | | |

Where it fails: TODO (see `worst_10_ids_by_cer` in `output/metrics/ocr_quality.json`).

## 6. Cleaning and masking

* Broken lines and hyphenation are joined; dashes and spaces normalized.
* PII removed: name (first title-cased header line), email, phone, URLs, street address, "City, ST ZIP".
* Low-confidence lines: kept in the main setup; the variant `ocr_conf` drops lines with score
  < **0.80** (TODO: justify from the score histogram) and is compared in Q3.
* **Masking**: after reading 50 random OCR texts (`output/tables/inspection_sample.txt`), phrases that
  state total experience ("12 years of experience", "Experience: 7 yrs", "a decade of experience"),
  age or date of birth are replaced by `[MASK]`. **TODO CVs (TODO %) were affected**
  (`output/metrics/masking.json`). Job dates and graduation years are kept.

## 7. Split

70 / 15 / 15 by CV id, stratified by occupation × experience band (0–2, 3–5, 6–10, 11–20), seed 42.
Files in `data/splits/`; every model uses them. All tuning on validation; the test set is used once.

## 8. Baselines

| Model | Setup |
|---|---|
| Median | median `y` of train |
| TF-IDF + Ridge | word 1–2 grams (+ char 3–5 grams if better on val), sublinear TF, **years kept as tokens** (they are evidence; Q3 tests removing them), α tuned on val |
| Embeddings + Ridge | frozen `BAAI/bge-small-en-v1.5`; CVs longer than 510 tokens are split into overlapping chunks (stride 64), chunk vectors averaged; α tuned on val |
| Rules (bonus) | regex date ranges in experience sections, overlaps merged, "Present" = year tuned on train |

## 9. BERT

* `distilbert-base-uncased`, `AutoModelForSequenceClassification(num_labels=1, problem_type="regression")`.
* Target scaled to [0, 1] (÷ 20); predictions mapped back and clipped to 0–20. Loss MSE (Huber in Q3).
* Long CVs: **head truncation at 512 tokens** in the main setup. TODO: % of CVs over the limit and
  % of years that survive truncation (`date_survival` in `output/metrics/runs.json`); head+tail and
  chunking are compared in Q3.
* lr 5e-5, batch 16, ≤ 10 epochs, warm-up 10%, weight decay 0.01, early stopping (patience 2) on
  validation MAE, fp16, seed 42. Training time / hardware: TODO.

## 10. Results (test set)

`output/tables/model_comparison.csv`

| Model | MAE | RMSE | Acc ±1 | Acc ±2 | Acc ±3 | Band acc | Band macro-F1 | R² | Train time | Inference / CV |
|---|---|---|---|---|---|---|---|---|---|---|
| Median | | | | | | | | | | |
| TF-IDF + Ridge | | | | | | | | | | |
| Embeddings + Ridge | | | | | | | | | | |
| Fine-tuned BERT | | | | | | | | | | |

**BERT's MAE gain** (95% paired bootstrap, 2,000 resamples, `output/tables/bert_gain_ci.csv`): TODO.

Figures: `pred_vs_true.png`, `mae_by_band.png`, `mae_by_occupation.png`, `confusion_*.png`, `bert_gain_ci.png`.
Top-10 BERT errors labelled by cause: `output/tables/bert_top10_errors.csv`.

## 11. Analytical questions

**Q1 — What does BERT learn that the baselines cannot?** TODO (gain + CI; errors by band; probe in
`output/tables/probe_summary.csv`: shifting all years +5 should only change CVs whose last job ends in
"Present"; deleting a job should lower the prediction by about its duration).

**Q2 — How much does OCR noise cost?** TODO (`output/tables/ocr_vs_reference.csv`: each model on OCR
text vs reference text).

**Q3 — Methodological sensitivity.** TODO (`output/tables/sensitivity.csv`: OCR cleaning raw / clean /
low-confidence removed for every model; BERT head vs head+tail vs chunk; MSE vs Huber; TF-IDF digits).

**Q4 — Business value, fairness and ethics.** TODO (`output/tables/fairness.csv`: MAE and mean error by
sex and race). Limitations: synthetic resumes, 5 occupations, U.S. statistics, English only, one
template family. The estimate can pre-fill a profile or help prioritize reading; it must never decide
who is hired.

## 12. How to reproduce

### Option A (recommended) — everything in one Google Colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karelysscs/cv_experience_ocr_bert/blob/Practica_04/notebooks/00_colab_full_pipeline.ipynb)

`notebooks/00_colab_full_pipeline.ipynb` runs the whole pipeline on a T4 GPU with *Run all*:
sampling, PaddleOCR, OCR quality, cleaning, split, all models, Q1–Q4 analyses, figures, the
Streamlit app with a public link, and pushes the results to GitHub. Needs the Colab secrets
`HF_TOKEN` (Hugging Face) and `GH_TOKEN` (fine-grained GitHub token, contents read/write on this
repo only). Progress is saved to Google Drive, so a disconnected run resumes where it stopped.
PaddlePaddle-GPU runs in an isolated virtual environment because it clashes with PyTorch's CUDA
libraries.

### Option B — Kaggle notebooks + local machine

The GPU stages run on **Kaggle or Google Colab**; everything else runs locally.

1. **Access:** accept the dataset conditions on Hugging Face, create a read token, and store it as the
   secret `HF_TOKEN` in Kaggle (*Add-ons → Secrets*) or Colab (*🔑 Secrets*).
2. **Stage 1 (GPU):** open `notebooks/01_ocr_gpu.ipynb` in Kaggle/Colab, GPU on, run all.
   It samples the CVs, runs PaddleOCR, measures OCR quality, cleans/masks the text and creates the
   split. Download `stage1_ocr_results.zip`, unzip it in the repo, commit and push.
3. **Stage 2 (GPU):** open `notebooks/02_models_gpu.ipynb`, run all. It trains every model and the Q3
   variants, runs the probe, evaluation and figures. Download `stage2_outputs.zip` (commit it) and
   `stage2_models.zip` (unzip locally, git-ignored).

Equivalent commands (from the repo root):

```bash
python -m src.download_data --n-per-occupation 800
python -m src.ocr_extraction --model server --device gpu
python -m src.ocr_quality
python -m src.preprocessing --inspect 50   # read the sample first
python -m src.preprocessing --min-conf 0.80
python -m src.split
python -m src.baseline_median
python -m src.baseline_tfidf
python -m src.baseline_embeddings
python -m src.baseline_rules
python -m src.train_bert
python -m src.probe
python -m src.evaluation
python -m src.visualization
```

## 13. Run the Streamlit app

Use Python **3.10–3.13** (PaddlePaddle has no 3.14 wheels):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run app.py
```

Tabs 1–3 only need the committed `output/` and `data/` files. Tab 4 (live estimator) runs PaddleOCR
mobile on CPU and needs the models from `stage2_models.zip` in `models/`.

## 14. Git workflow

Work happens on feature branches (`feature/data-sampling`, `feature/paddleocr-extraction`,
`feature/baselines`, `feature/bert-regression`, `feature/evaluation`, `feature/streamlit-dashboard`, …)
merged by Pull Request into **`Practica_04`**, which is merged into `main` at the end.

## Video

Link in `video/link.txt`.
