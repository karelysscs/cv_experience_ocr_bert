"""Stage 2b — reading order, cleaning, PII removal and masking of direct answers.

Produces data/processed/cv_text.csv with one row per CV and four text variants
(see utils.TEXT_COLUMNS):

    text_ocr_raw    PaddleOCR lines in detector order, masked only       (Q3 "raw OCR")
    text_ocr_clean  reading order rebuilt + cleaning + PII + masking     (main model input)
    text_ocr_conf   as clean, but lines with score < --min-conf dropped  (Q3)
    text_reference  dataset `text`, cleaned the same way                 (Q2 only, never trained on)

Every variant that can reach a model is masked: phrases stating total experience,
age or date of birth are replaced with [MASK]. Job dates and graduation years stay.

    python -m src.preprocessing --inspect 50     # dump 50 random OCR texts to read first
    python -m src.preprocessing --min-conf 0.80
"""
from __future__ import annotations

import argparse
import re
import statistics

import numpy as np
import pandas as pd

from src.utils import (METADATA_CSV, METRICS, OCR_DIR, PROCESSED_CSV, REFERENCE_CSV,
                       SEED, TABLES, ensure_dirs, load_json, save_json)

# ======================================================================== reading order

def _rows(lines: list[dict]) -> list[str]:
    """Group boxes of one column into visual rows (top->bottom), each row left->right."""
    if not lines:
        return []
    heights = [l["box"][3] - l["box"][1] for l in lines]
    tol = 0.5 * statistics.median(heights)
    lines = sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    rows, current, cy = [], [lines[0]], (lines[0]["box"][1] + lines[0]["box"][3]) / 2
    for l in lines[1:]:
        y = (l["box"][1] + l["box"][3]) / 2
        if abs(y - cy) <= tol:
            current.append(l)
        else:
            rows.append(current)
            current, cy = [l], y
    rows.append(current)
    return [" ".join(l["text"] for l in sorted(r, key=lambda l: l["box"][0])) for r in rows]


def find_gutter(lines: list[dict], width: int) -> int | None:
    """Return the x of a vertical gutter separating two columns, or None for one column.

    A gutter is an x position (between 20% and 80% of the page) crossed by almost no box
    while both sides hold a substantial share of the text lines.
    """
    if len(lines) < 10:
        return None
    best_x, best_cross = None, len(lines)
    for x in range(int(0.2 * width), int(0.8 * width), max(1, width // 200)):
        cross = sum(1 for l in lines if l["box"][0] < x < l["box"][2])
        if cross < best_cross:
            best_x, best_cross = x, cross
    left = sum(1 for l in lines if l["box"][2] <= best_x)
    right = sum(1 for l in lines if l["box"][0] >= best_x)
    if best_cross <= 0.05 * len(lines) and min(left, right) >= 0.15 * len(lines):
        return best_x
    return None


def rebuild_reading_order(lines: list[dict], width: int | None = None) -> list[str]:
    """Top-to-bottom, left-to-right; two-column layouts are read column by column.

    Everything above the point where both columns have started (name, title, a header
    that may sit in either half) is read first. Lines that span the gutter below that
    are read last if they sit under both columns, otherwise merged into the left column
    by y. This keeps a job title next to its dates: both live in the same column.
    """
    if not lines:
        return []
    width = width or max(l["box"][2] for l in lines)
    gx = find_gutter(lines, width)
    if gx is None:
        return _rows(lines)
    left = [l for l in lines if l["box"][2] <= gx]
    right = [l for l in lines if l["box"][0] >= gx]
    span = [l for l in lines if l["box"][0] < gx < l["box"][2]]
    top = max(min(l["box"][1] for l in left), min(l["box"][1] for l in right))
    head = [l for l in lines if l["box"][3] <= top]
    left = [l for l in left if l not in head]
    right = [l for l in right if l not in head]
    bottom = max(l["box"][3] for l in left + right)
    tail = [l for l in span if l not in head and l["box"][1] >= bottom]
    middle = [l for l in span if l not in head and l not in tail]
    return _rows(head) + _rows(left + middle) + _rows(right) + _rows(tail)


# ======================================================================== cleaning

DASHES = re.compile(r"[‐-―−]")

def normalize_text(s: str) -> str:
    s = DASHES.sub("-", s)
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def fix_broken_lines(lines: list[str]) -> list[str]:
    """Join hyphenated words and sentences broken across lines."""
    out: list[str] = []
    for line in (normalize_text(l) for l in lines):
        if not line:
            continue
        if out and out[-1].endswith("-") and line[:1].islower():
            out[-1] = out[-1][:-1] + line                      # hyphenation: "manage-\nment"
        elif (out and line[:1].islower() and len(out[-1]) >= 35 and not out[-1].isupper()
              and not re.search(r"[.:;!?\]]$", out[-1])):
            out[-1] = out[-1] + " " + line                     # wrapped sentence (long line, not a heading)
        else:
            out.append(line)
    return out


# ---------------------------------------------------------------- PII
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S*", re.I)
PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
STREET = re.compile(r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|"
                    r"Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pl|Place|Pkwy|Hwy)\b\.?[^\n]*")
CITY_STATE_ZIP = re.compile(r"\b[A-Z][a-zA-Z .]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b")
STATE_COUNTRY = re.compile(r"\b[A-Z]{2},\s*USA\b")  # PopResume header: "AL, USA"
# letters only, 1-4 words; OCR sometimes drops the space ("TYLERVANCE"), so one word counts
NAME_LINE = re.compile(r"^[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){0,3}$")
SECTION_HEADING = re.compile(r"^(?:work history|(?:professional |work )?experience|employment(?: history)?|"
                             r"education|skills|summary|profile|certifications?|projects|languages)$", re.I)


def remove_pii(lines: list[str]) -> list[str]:
    out = []
    for i, line in enumerate(lines):
        if (i < 3 and NAME_LINE.match(line.strip()) and not SECTION_HEADING.match(line.strip())
                and "[NAME]" not in out):
            # the candidate's name is the first letters-only line of the header (never a section title)
            out.append("[NAME]")
            continue
        for pat, tok in [(EMAIL, "[EMAIL]"), (URL, "[URL]"), (PHONE, "[PHONE]"),
                         (STREET, "[ADDRESS]"), (CITY_STATE_ZIP, "[ADDRESS]"),
                         (STATE_COUNTRY, "[LOCATION]")]:
            line = pat.sub(tok, line)
        out.append(line)
    return out


# ---------------------------------------------------------------- masking of direct answers
NUM_WORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
             r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty[- ]\w+|thirty)")
NUM = rf"(?:\d{{1,2}}(?:\.\d)?\+?|{NUM_WORDS})"
MASK_PATTERNS = {
    # "12 years of experience", "over 5+ yrs of professional experience", "ten years' experience"
    "total_experience": re.compile(
        rf"\b(?:over|more than|nearly|almost|about|approximately|around)?\s*{NUM}\s*\+?\s*"
        rf"(?:years?|yrs?)['’]?\.?\s*(?:of\s+)?(?:\w+\s+){{0,3}}?experience\b", re.I),
    # "Experience: 12 years", "Total experience - 7 yrs"
    "experience_colon": re.compile(rf"\b(?:total\s+)?experience\s*[:\-]\s*{NUM}\s*(?:years?|yrs?)\b", re.I),
    # "a decade of experience"
    "decade": re.compile(r"\b(?:a|over a|nearly a|more than a)\s+decade\s+of\s+(?:\w+\s+){0,2}?experience\b", re.I),
    "age": re.compile(r"\bage[d]?\s*[:\-]?\s*\d{2}\b|\b\d{2}\s*(?:years?|yrs?)\s*old\b", re.I),
    "birth": re.compile(r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|born(?:\s+on|\s+in)?)\b\s*[:\-]?[^\n]{0,30}", re.I),
}


def mask_direct_answers(text: str) -> tuple[str, dict[str, int]]:
    counts = {}
    for name, pat in MASK_PATTERNS.items():
        text, n = pat.subn("[MASK]", text)
        counts[name] = n
    return text, counts


# ======================================================================== pipeline

def ocr_variants(rec: dict, min_conf: float) -> dict[str, str]:
    lines = rec["lines"]
    width = rec.get("image_size", [None])[0]
    raw = "\n".join(normalize_text(l["text"]) for l in lines)
    clean = "\n".join(fix_broken_lines(remove_pii(rebuild_reading_order(lines, width))))
    kept = [l for l in lines if l["score"] >= min_conf]
    conf = "\n".join(fix_broken_lines(remove_pii(rebuild_reading_order(kept, width))))
    return {"text_ocr_raw": raw, "text_ocr_clean": clean, "text_ocr_conf": conf}


def clean_reference(text: str) -> str:
    return "\n".join(fix_broken_lines(remove_pii(str(text).splitlines())))


def process_ocr_record(rec: dict, min_conf: float = 0.80) -> dict:
    """OCR JSON -> masked text variants (also used by the Streamlit estimator)."""
    row = {}
    for col, txt in ocr_variants(rec, min_conf).items():
        row[col], counts = mask_direct_answers(txt)
        if col == "text_ocr_clean":
            row["n_masked"] = sum(counts.values())
            row.update({f"mask_{k}": v for k, v in counts.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.80,
                    help="drop OCR lines below this score in the 'ocr_conf' variant")
    ap.add_argument("--inspect", type=int, default=0,
                    help="write N random OCR texts (before masking) to read manually and exit")
    args = ap.parse_args()
    ensure_dirs()

    meta = pd.read_csv(METADATA_CSV, dtype={"id": str})
    recs = {}
    for cv_id in meta["id"]:
        p = OCR_DIR / f"{cv_id}.json"
        if p.exists():
            recs[cv_id] = load_json(p)
    print(f"OCR JSON found for {len(recs)}/{len(meta)} CVs")

    if args.inspect:
        rng = np.random.default_rng(SEED)
        ids = rng.choice(sorted(recs), size=min(args.inspect, len(recs)), replace=False)
        with open(TABLES / "inspection_sample.txt", "w", encoding="utf-8") as f:
            for i in ids:
                v = ocr_variants(recs[i], args.min_conf)
                f.write(f"===== {i} =====\n{v['text_ocr_clean']}\n\n")
        print(f"Wrote {len(ids)} texts to {TABLES / 'inspection_sample.txt'}")
        return

    ref = pd.read_csv(REFERENCE_CSV, dtype={"id": str}).set_index("id")["text"] \
        if REFERENCE_CSV.exists() else pd.Series(dtype=str)

    rows = []
    for cv_id, rec in recs.items():
        row = {"id": cv_id, **process_ocr_record(rec, args.min_conf),
               "n_lines": len(rec["lines"]), "mean_conf": rec.get("mean_conf")}
        if cv_id in ref.index:
            row["text_reference"], c = mask_direct_answers(clean_reference(ref[cv_id]))
            row["n_masked_reference"] = sum(c.values())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_CSV, index=False)

    mask_cols = [c for c in df.columns if c.startswith("mask_")]
    report = {
        "min_conf": args.min_conf,
        "n_cvs": len(df),
        "n_cvs_masked": int((df["n_masked"] > 0).sum()),
        "share_cvs_masked": round(float((df["n_masked"] > 0).mean()), 4),
        "cvs_masked_by_pattern": {c[5:]: int((df[c] > 0).sum()) for c in mask_cols},
        "n_cvs_masked_reference": int((df.get("n_masked_reference", pd.Series(dtype=int)) > 0).sum()),
        "lines_dropped_low_conf": int(sum(sum(l["score"] < args.min_conf for l in r["lines"])
                                          for r in recs.values())),
        "lines_total": int(sum(len(r["lines"]) for r in recs.values())),
    }
    save_json(report, METRICS / "masking.json")
    print(report)


if __name__ == "__main__":
    main()
