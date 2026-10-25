"""Streamlit app — CV experience estimator (PaddleOCR + BERT).

    streamlit run app.py

Everything shown is read from output/ and data/ (produced by the pipeline); the
estimator tab runs PaddleOCR (mobile model, CPU) and the trained models live.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

# On Windows, PyTorch must load its DLLs before PaddlePaddle does (PaddleOCR imports torch
# indirectly through modelscope); otherwise torch fails with "WinError 127 ... shm.dll".
import torch  # noqa: F401
import numpy as np
import pandas as pd
import streamlit as st

from src.utils import (BAND_LABELS, BERT_DIR, CAP, FIGURES, IMAGES, METADATA_CSV, METRICS,
                       OCR_DIR, PREDICTIONS, PROCESSED_CSV, TABLES, load_json)
from src import visualization as viz

st.set_page_config(page_title="CV Experience Estimator", page_icon="📄", layout="wide")


# ======================================================================== helpers

@st.cache_data
def read_csv(path: Path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


@st.cache_data
def read_json(path: Path):
    return load_json(path) if path.exists() else None


def show_fig_or_png(fig_fn, png: str, *args):
    """Prefer a live figure; fall back to the saved PNG; otherwise explain what is missing."""
    try:
        st.pyplot(fig_fn(*args), clear_figure=True)
    except Exception:
        if (FIGURES / png).exists():
            st.image(str(FIGURES / png))
        else:
            st.info(f"Not available yet — run the pipeline (missing `{png}`).")


def caption(text: str):
    st.caption(f"**Why this chart:** {text}")


def show_table(df):
    if df is None:
        st.info("Pending — run the pipeline.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)


def fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}%}" if pct else f"{x:.{nd}f}"


meta = read_csv(METADATA_CSV, dtype={"id": str})
ocr_q = read_json(METRICS / "ocr_quality.json") or {}
ocr_s = read_json(METRICS / "ocr_summary.json") or {}
mask = read_json(METRICS / "masking.json") or {}
split = read_json(METRICS / "split.json") or {}
comp = read_csv(TABLES / "model_comparison.csv")

st.title("📄 Predicting years of experience from CV images")
st.markdown("PaddleOCR reads the CV image · four models estimate the years of work experience (0–"
            f"{CAP}) · all compared on the same test set.")

tab1, tab2, tab3, tab4 = st.tabs(["1 · Problem & Methodology", "2 · OCR & Exploratory Analysis",
                                  "3 · Model Results", "4 · Interactive Estimator"])

# ======================================================================== TAB 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CVs sampled", f"{len(meta):,}" if meta is not None else "–")
    c2.metric("Mean OCR CER", fmt(ocr_q.get("cer_mean"), 3))
    c3.metric("Years read correctly", fmt(ocr_q.get("share_years_correct"), 1, pct=True))
    c4.metric("CVs masked", f"{mask.get('n_cvs_masked', '–')}")

    st.header("Objective")
    st.markdown(f"""
Estimate a candidate's **years of work experience (0–{CAP})** from the **image** of their CV.
The number is rarely written on the CV: it has to be inferred by connecting the dates of each
job, how many positions there are, how titles progress and when education ended — what a
recruiter does in their head.
""")
    st.header("Data")
    st.markdown(f"""
* **PopResume** (Yu, Park & Moon, EMNLP 2026, CC BY 4.0): 60,884 synthetic resumes, 5 occupations,
  generated from U.S. population statistics — no real person's data.
* **Sample:** {len(meta) // 5 if meta is not None else '–'} CVs per occupation, **no-photo images only**,
  streamed and shuffled with seed 42 (IDs in `data/raw/sampled_ids.txt`).
* **Target:** `exp_year` clipped to 0–{CAP}; {fmt(split.get('share_at_cap'), 1, pct=True)} of the CVs sit at the cap.
* **Masking rule:** phrases that state total experience, age or date of birth are replaced by `[MASK]`
  ({mask.get('n_cvs_masked', '–')} CVs affected). Job dates and graduation years are kept — they are the evidence.
* **Never used as features:** age, sex, race, names, state, region. Sex and race are only used for the fairness check.
""")
    st.header("OCR and reading order")
    st.markdown(f"""
* **PaddleOCR 3.x**, recognition model `{ocr_s.get('model', 'PP-OCRv5_server_rec')}` (more accurate on small
  digits than the mobile model — dates are what matter). Orientation / unwarping disabled: the
  images are flat digital renders. Mean confidence {fmt(ocr_s.get('mean_confidence'), 3)},
  {fmt(ocr_s.get('time_per_cv_s_mean'), 2)} s per CV on {ocr_s.get('hardware', '–')}.
* **Reading order:** boxes are grouped into rows (top → bottom, left → right). A two-column layout is
  detected by a vertical gutter crossed by almost no box; the columns are then read one after the
  other, so a job title stays next to its dates.
""")
    st.header("Models and split")
    st.markdown(f"""
| | Model | What it can see |
|---|---|---|
| 0 | Median of the training set | nothing — the floor every model must beat |
| 1 | TF-IDF (word 1–2 grams) + Ridge | which words and years appear, not how they relate |
| 2 | Frozen sentence embeddings (`bge-small-en-v1.5`, chunked + averaged) + Ridge | the overall meaning |
| 3 | Fine-tuned BERT regressor | token interactions: title ↔ dates ↔ position in the career |
| bonus | Rule-based date arithmetic | sums the job date ranges explicitly |

Split **70 / 15 / 15** by CV id, stratified by occupation × experience band, seed 42
({split.get('sizes', {})}). All tuning on validation; the test set is used once.
""")
    st.header("Limitations")
    st.markdown("""
Synthetic resumes · only 5 occupations · U.S. statistics · English only · one CV template family.
An experience estimate can help **pre-fill a profile or prioritize reading**; it must **never decide
who is hired or interviewed**.
""")

# ======================================================================== TAB 2
with tab2:
    st.header("Years of experience")
    if meta is not None:
        st.pyplot(viz.fig_experience_distribution(meta), clear_figure=True)
    else:
        st.info("Run `python -m src.download_data` first.")
    caption("a histogram shows the shape and the pile-up at the cap; the box plots show that "
            "occupations differ, which is why the split is stratified by occupation and band.")

    st.header("OCR quality")
    per_cv = read_csv(TABLES / "ocr_quality_per_cv.csv", dtype={"id": str})
    if per_cv is not None:
        st.pyplot(viz.fig_ocr_quality(per_cv), clear_figure=True)
        by_occ = read_csv(TABLES / "ocr_quality_by_occupation.csv")
        st.dataframe(by_occ, width="stretch", hide_index=True)
    else:
        st.info("Run `python -m src.ocr_quality`.")
    caption("CER/WER distributions show whether errors are rare and concentrated in a few CVs; "
            "the share of years read correctly matters most, because a misread digit changes the answer.")

    st.header("Example: image → PaddleOCR boxes → text")
    proc = read_csv(PROCESSED_CSV, dtype={"id": str})
    if proc is not None:
        with_img = [i for i in proc["id"] if (IMAGES / f"{i}.png").exists()]
        options = with_img or proc["id"].tolist()
        cv_id = st.selectbox("CV", options[:200], key="eda_cv")
        c1, c2 = st.columns(2)
        with c1:
            if (IMAGES / f"{cv_id}.png").exists() and (OCR_DIR / f"{cv_id}.json").exists():
                st.pyplot(viz.fig_ocr_boxes(cv_id), clear_figure=True)
            elif (FIGURES / "ocr_boxes_example.png").exists():
                st.image(str(FIGURES / "ocr_boxes_example.png"),
                         caption="Images are not committed; showing the saved example.")
        with c2:
            st.text_area("Cleaned + masked OCR text (model input)",
                         proc.set_index("id").at[cv_id, "text_ocr_clean"], height=600)
    caption("seeing boxes next to the extracted text makes reading-order and digit errors visible.")

# ======================================================================== TAB 3
with tab3:
    st.header("Model comparison (test set)")
    if comp is not None:
        st.dataframe(comp.drop(columns="key").style.format(precision=3), width="stretch", hide_index=True)
    else:
        st.info("Run `python -m src.evaluation`.")

    gain = read_csv(TABLES / "bert_gain_ci.csv")
    if gain is not None and len(gain):
        st.subheader("BERT's MAE gain over each baseline")
        st.pyplot(viz.fig_bert_gain(gain), clear_figure=True)
        st.dataframe(gain.drop(columns="key"), width="stretch", hide_index=True)
        caption("a confidence interval that does not cross 0 means the gain is not explained by which "
                "CVs happened to land in the test set (paired bootstrap, 2,000 resamples).")

    preds = viz.load_main_preds()
    if preds:
        st.subheader("Predicted vs true years")
        chosen = st.multiselect("Models", list(preds), default=list(preds),
                                format_func=lambda k: viz.MODEL_LABELS.get(k, k))
        if chosen:
            st.pyplot(viz.fig_pred_vs_true({k: preds[k] for k in chosen}), clear_figure=True)
        caption("points on the dashed line are perfect; a red fit line flatter than 45° means the model "
                "pulls predictions to the middle (overestimates juniors, underestimates seniors).")

    band = read_csv(TABLES / "mae_by_band.csv")
    occ = read_csv(TABLES / "mae_by_occupation.csv")
    if band is not None:
        st.subheader("Where are the errors?")
        st.pyplot(viz.fig_mae_by(band, "band", BAND_LABELS), clear_figure=True)
        st.pyplot(viz.fig_mae_by(occ, "occupation"), clear_figure=True)
        caption("grouped bars compare models within each band/occupation, showing where BERT's gain comes from.")

    cms = read_json(METRICS / "confusion_matrices.json")
    if cms:
        st.subheader("Band confusion matrices")
        cols = st.columns(min(len(cms["matrices"]), 3))
        for i, (k, cm) in enumerate(cms["matrices"].items()):
            with cols[i % len(cols)]:
                st.pyplot(viz.fig_confusion(cm, viz.MODEL_LABELS.get(k, k)), clear_figure=True)
        caption("bands are what a recruiter acts on; off-diagonal cells next to the diagonal are near misses.")

    st.subheader("Q2 · Cost of OCR noise (same models on OCR text vs reference text)")
    q2 = read_csv(TABLES / "ocr_vs_reference.csv")
    show_table(q2)

    st.subheader("Q3 · Methodological sensitivity")
    q3 = read_csv(TABLES / "sensitivity.csv")
    show_table(q3)

    st.subheader("Q1 · Probe: shift every date +5 years / delete one job")
    probe = read_csv(TABLES / "probe_summary.csv")
    show_table(probe)

    st.subheader("Q4 · Fairness check")
    fair = read_csv(TABLES / "fairness.csv")
    if fair is not None:
        model = st.selectbox("Model", fair["Model"].unique(), index=len(fair["Model"].unique()) - 1)
        st.dataframe(fair[fair["Model"] == model], width="stretch", hide_index=True)
        caption("a mean error far from 0 for one group means the model systematically over- or "
                "underestimates that group, even though sex and race are not inputs.")

# ======================================================================== TAB 4
with tab4:
    st.header("Estimate experience from a CV image")

    @st.cache_resource
    def get_ocr():
        from src.ocr_extraction import build_ocr
        return build_ocr("mobile", "cpu")

    def all_predictions(text: str) -> dict:
        from src import baseline_median, baseline_rules, baseline_tfidf, baseline_embeddings, train_bert
        out = {}
        for key, fn in [("median", baseline_median.predict), ("tfidf", baseline_tfidf.predict),
                        ("embeddings", baseline_embeddings.predict), ("rules", baseline_rules.predict),
                        ("bert", lambda t: train_bert.predict(t, BERT_DIR / "bert"))]:
            try:
                out[viz.MODEL_LABELS[key]] = float(np.clip(fn([text])[0], 0, CAP))
            except Exception as e:  # model not trained / weights not downloaded
                out[viz.MODEL_LABELS[key]] = f"unavailable ({type(e).__name__})"
        return out

    up = st.file_uploader("CV image or PDF", type=["png", "jpg", "jpeg", "pdf"])
    if up is not None:
        from PIL import Image
        from src.ocr_extraction import ocr_image
        from src.preprocessing import process_ocr_record
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "cv.png"
            if up.name.lower().endswith(".pdf"):
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(up.read())
                pdf[0].render(scale=2).to_pil().save(img_path)     # first page
            else:
                Image.open(up).convert("RGB").save(img_path)
            img = Image.open(img_path)
            with st.spinner("Running PaddleOCR (CPU, mobile model)…"):
                lines = ocr_image(get_ocr(), img_path)
            rec = {"lines": lines, "image_size": img.size}
            text = process_ocr_record(rec)["text_ocr_clean"]
            c1, c2 = st.columns(2)
            c1.image(img, width="stretch")
            c2.text_area("Extracted text (cleaned, masked)", text, height=500)
        with st.spinner("Predicting…"):
            res = all_predictions(text)
        st.subheader("Predicted years of experience")
        cols = st.columns(len(res))
        for col, (k, v) in zip(cols, res.items()):
            col.metric(k, f"{v:.1f}" if isinstance(v, float) else "–", help=None if isinstance(v, float) else v)

    st.divider()
    st.header("Browse test CVs")
    preds = viz.load_main_preds()
    proc = read_csv(PROCESSED_CSV, dtype={"id": str})
    if preds and "bert" in preds and proc is not None:
        wide = preds["bert"][["id", "occupation", "y_true"]].copy()
        for k, df in preds.items():
            wide[viz.MODEL_LABELS[k]] = df.set_index("id").loc[wide["id"], "y_pred"].values
        wide["BERT error"] = wide["Fine-tuned BERT"] - wide["y_true"]
        c1, c2, c3 = st.columns(3)
        occs = c1.multiselect("Occupation", sorted(wide["occupation"].unique()))
        yr = c2.slider("True years", 0, CAP, (0, CAP))
        err = c3.slider("|BERT error| at least", 0.0, float(CAP), 0.0, 0.5)
        f = wide[wide["y_true"].between(*yr) & (wide["BERT error"].abs() >= err)]
        if occs:
            f = f[f["occupation"].isin(occs)]
        st.dataframe(f.round(2), width="stretch", hide_index=True)
        pick = st.selectbox("Read a CV", f["id"].tolist())
        if pick:
            st.text_area("OCR text", proc.set_index("id").at[pick, "text_ocr_clean"], height=350)

        st.subheader("Good and bad predictions")
        top = read_csv(TABLES / "bert_top10_errors.csv", dtype={"id": str})
        best = wide.assign(a=wide["BERT error"].abs()).nsmallest(3, "a")
        st.markdown("**Good** (smallest BERT errors)")
        st.dataframe(best.drop(columns="a").round(2), hide_index=True, width="stretch")
        if top is not None:
            st.markdown("**Bad** (largest BERT errors, with the labelled cause)")
            st.dataframe(top[["id", "occupation", "y_true", "y_pred", "cause"]].round(2),
                         hide_index=True, width="stretch")
    else:
        st.info("Test predictions not found — train the models (notebooks/02_models_gpu.ipynb).")
