"""
preprocessing.py — Dataset loading and normalisation for CatchFish.

Responsibilities:
  - Discover real-world CSV files in dataset/ (skip synthetic datasetN.csv)
  - Parse two schemas: Schema A (sender/body) and Schema B (email_text/sender_domain)
  - Normalise all columns to a consistent DataFrame format
  - Merge multiple CSVs into one clean dataset
"""

import os

import pandas as pd

from config import (
    DATASET_DIR,
    HTML_TAG_RE,
    LABEL_MAP,
    MULTI_WS_RE,
    SENDER_RE,
    SYNTHETIC_CSV,
    URGENT_KEYWORDS,
    URL_RE,
)


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text) -> str:
    """Strip HTML tags and collapse whitespace. Preserves URLs as tokens."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return MULTI_WS_RE.sub(" ", text).strip()


def extract_sender_domain(sender) -> str:
    """Pull the bare domain from a sender string like 'Name <user@domain.com>'."""
    if not isinstance(sender, str) or not sender.strip():
        return ""
    m = SENDER_RE.search(sender)
    if m:
        return m.group(1).lower().strip(".")
    if "@" in sender:
        return sender.split("@", 1)[1].strip().lower().rstrip(">")
    return ""


def count_links(text) -> int:
    return len(URL_RE.findall(str(text)))


def count_urgent_keywords(subject, body) -> int:
    combined = (str(subject) + " " + str(body)).lower()
    return sum(1 for kw in URGENT_KEYWORDS if kw in combined)


def to_int_col(series: pd.Series) -> pd.Series:
    """Strip surrounding quotes and coerce a column to integer (default 0)."""
    return (
        pd.to_numeric(
            series.astype(str).str.strip().str.strip('"').str.strip("'"),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )


# ── CSV loader ─────────────────────────────────────────────────────────────────

def load_csv(fpath: str) -> pd.DataFrame:
    """
    Load a single CSV and normalise it to the shared schema:

        label (int 0/1), email_text, subject, sender_domain,
        has_attachment, links_count, urgent_keywords

    Supports two input schemas automatically:
      Schema A  columns: sender, body, subject, label
      Schema B  columns: email_text, sender_domain, has_attachment,
                         links_count, urgent_keywords, label
    """
    raw = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
    raw.columns = raw.columns.str.strip().str.strip('"').str.strip("'").str.lower()

    out = pd.DataFrame()

    # label
    lbl = raw["label"].astype(str).str.strip().str.strip('"').str.strip("'").str.lower()
    out["label"] = lbl.map(LABEL_MAP).fillna(0).astype(int)

    # email_text
    if "email_text" in raw.columns:
        out["email_text"] = raw["email_text"].fillna("").apply(clean_text)
    elif "body" in raw.columns:
        out["email_text"] = raw["body"].fillna("").apply(clean_text)
    else:
        out["email_text"] = ""

    # subject
    out["subject"] = (
        raw["subject"].fillna("").apply(clean_text)
        if "subject" in raw.columns else ""
    )

    # sender_domain
    if "sender_domain" in raw.columns:
        out["sender_domain"] = (
            raw["sender_domain"].astype(str)
            .str.strip().str.strip('"').str.strip("'").fillna("")
        )
    elif "sender" in raw.columns:
        out["sender_domain"] = raw["sender"].fillna("").apply(extract_sender_domain)
    else:
        out["sender_domain"] = ""

    # numeric columns — use precomputed values when available, otherwise derive them
    out["has_attachment"] = (
        to_int_col(raw["has_attachment"]) if "has_attachment" in raw.columns else 0
    )
    out["links_count"] = (
        to_int_col(raw["links_count"]) if "links_count" in raw.columns
        else out["email_text"].apply(count_links)
    )
    out["urgent_keywords"] = (
        to_int_col(raw["urgent_keywords"]) if "urgent_keywords" in raw.columns
        else out.apply(
            lambda r: count_urgent_keywords(r["subject"], r["email_text"]), axis=1
        )
    )

    return out


# ── Dataset discovery & merge ──────────────────────────────────────────────────

def discover_csv_files() -> list[str]:
    """Return sorted real-world CSV filenames from dataset/, skipping datasetN.csv."""
    return sorted(
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(".csv") and not SYNTHETIC_CSV.match(f)
    )


def load_dataset() -> tuple[pd.DataFrame, dict]:
    """
    Load, normalise, and merge all CSVs from dataset/.

    Returns:
        df           — merged DataFrame, duplicates removed, empty rows dropped
        per_dataset  — {filename: {rows, phish, legit}} stats per file
    """
    csv_files = discover_csv_files()
    print(f"[Data] Found {len(csv_files)} CSV file(s)\n")

    parts = []
    per_dataset = {}

    for fname in csv_files:
        fpath = os.path.join(DATASET_DIR, fname)
        try:
            part = load_csv(fpath)
            ph = int((part["label"] == 1).sum())
            lg = int((part["label"] == 0).sum())
            per_dataset[fname] = {"rows": len(part), "phish": ph, "legit": lg}
            print(f"  {fname:30s}  {len(part):7,} rows  phishing={ph:,}  legit={lg:,}")
            parts.append(part)
        except Exception as exc:
            print(f"  {fname:30s}  SKIPPED — {exc}")

    if not parts:
        raise RuntimeError("No valid CSV files found in dataset/")

    df = pd.concat(parts, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["email_text", "subject"])
    print(f"\n  Duplicates removed : {before - len(df):,}")

    # drop rows where both email_text and subject are blank
    df = df[~(
        (df["email_text"].str.strip() == "") &
        (df["subject"].str.strip() == "")
    )]

    # enforce types
    df["has_attachment"]  = df["has_attachment"].astype(int)
    df["links_count"]     = df["links_count"].astype(int)
    df["urgent_keywords"] = df["urgent_keywords"].astype(int)
    df["sender_domain"]   = df["sender_domain"].fillna("")
    df["email_text"]      = df["email_text"].fillna("")
    df["subject"]         = df["subject"].fillna("")

    n_ph = int((df["label"] == 1).sum())
    n_lg = int((df["label"] == 0).sum())
    print(f"\n[Data] Total : {len(df):,} rows  phishing={n_ph:,}  legit={n_lg:,}")

    return df, per_dataset
