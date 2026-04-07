"""
SmartJasus — ML Training Pipeline
===================================
Trains multiple classifiers on ALL CSV datasets found in dataset/,
selects the best by F1-score at the production threshold (0.60),
and saves the winner to models/phishing_detection.pkl.

Supported dataset schemas (auto-detected per file):
  Schema A — original datasets (CEAS_08, Enron, Ling, Nazario, …):
    columns: sender, body, subject, label (int 0/1)
  Schema B — pre-processed format (dataset1–dataset5, …):
    columns: email_text, subject, has_attachment, links_count,
             sender_domain, urgent_keywords, label (text phishing/legitimate)

Any CSV added to dataset/ is automatically included — no code changes needed.

Reference data (dataset/):
  - legitimate_domains.json
  - suspicious_tlds.json
  - url_shorteners.json

Algorithms compared:
  1. Logistic Regression
  2. Random Forest
  3. Extra Trees
  4. Gradient Boosting
  5. XGBoost (if installed)
"""

import json
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE    = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.join(_BASE, "dataset")


def _load_json_set(filename: str, key: str) -> set:
    path = os.path.join(_DATASET, filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return set(json.load(f).get(key, []))
    return set()


LEGITIMATE_DOMAINS: set = _load_json_set("legitimate_domains.json", "domains")
SUSPICIOUS_TLDS: set    = _load_json_set("suspicious_tlds.json", "tlds")
URL_SHORTENERS: set     = _load_json_set("url_shorteners.json", "shorteners")

URGENT_KEYWORDS = [
    "urgent", "immediate", "action required", "verify", "suspended",
    "click now", "confirm", "account", "password", "login", "update",
    "alert", "security", "limited time", "expire", "expires",
]

# ── Regex helpers ──────────────────────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"[ \t]+")
_URL_RE      = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_SENDER_RE   = re.compile(r"[\w.\-+]+@([\w.\-]+\.\w{2,})")

# Label normalisation map — covers both int and text variants
_LABEL_MAP = {
    "1": 1, "phishing": 1, "spam": 1, "malicious": 1,
    "0": 0, "legitimate": 0, "ham": 0, "safe": 0, "benign": 0,
}


def _clean_text(text) -> str:
    """Strip HTML tags and normalise whitespace. Preserve URLs."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _MULTI_WS_RE.sub(" ", text).strip()


def _extract_domain(sender) -> str:
    """Extract bare domain from a sender field (e.g. 'Name <user@domain.com>')."""
    if not isinstance(sender, str) or not sender.strip():
        return ""
    m = _SENDER_RE.search(sender)
    if m:
        return m.group(1).lower().strip(".")
    if "@" in sender:
        return sender.split("@", 1)[1].strip().lower().rstrip(">")
    return ""


def _count_links(text) -> int:
    return len(_URL_RE.findall(str(text)))


def _count_urgent(subject, body) -> int:
    combined = (str(subject) + " " + str(body)).lower()
    return sum(1 for kw in URGENT_KEYWORDS if kw in combined)


def _to_numeric_col(series: pd.Series) -> pd.Series:
    """Strip quotes and coerce to integer, defaulting to 0."""
    return (
        pd.to_numeric(
            series.astype(str).str.strip().str.strip('"').str.strip("'"),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )


# ── Universal CSV loader ───────────────────────────────────────────────────────

def _load_any_csv(fpath: str) -> pd.DataFrame:
    """
    Load ANY dataset CSV and normalise it to the pipeline's required format:
      label (int 0/1), email_text, subject, sender_domain,
      has_attachment, links_count, urgent_keywords

    Auto-detects schema:
      Schema A: has 'body' column  → original dataset format
      Schema B: has 'email_text'   → pre-processed dataset format
    """
    raw = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")

    # Normalise column names (strip surrounding quotes/spaces from some CSVs)
    raw.columns = raw.columns.str.strip().str.strip('"').str.strip("'").str.lower()

    out = pd.DataFrame()

    # ── Label → int 0/1 ──────────────────────────────────────────────────────
    lbl = (
        raw["label"]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .str.lower()
    )
    out["label"] = lbl.map(_LABEL_MAP).fillna(0).astype(int)

    # ── Email text ────────────────────────────────────────────────────────────
    if "email_text" in raw.columns:
        # Schema B: already has email_text
        out["email_text"] = raw["email_text"].fillna("").apply(_clean_text)
    elif "body" in raw.columns:
        # Schema A: body column
        out["email_text"] = raw["body"].fillna("").apply(_clean_text)
    else:
        out["email_text"] = ""

    # ── Subject ───────────────────────────────────────────────────────────────
    out["subject"] = (
        raw["subject"].fillna("").apply(_clean_text)
        if "subject" in raw.columns else ""
    )

    # ── Sender domain ─────────────────────────────────────────────────────────
    if "sender_domain" in raw.columns:
        # Schema B: pre-computed domain
        out["sender_domain"] = (
            raw["sender_domain"]
            .astype(str)
            .str.strip()
            .str.strip('"')
            .str.strip("'")
            .fillna("")
        )
    elif "sender" in raw.columns:
        # Schema A: derive from sender address
        out["sender_domain"] = raw["sender"].fillna("").apply(_extract_domain)
    else:
        out["sender_domain"] = ""

    # ── has_attachment ────────────────────────────────────────────────────────
    if "has_attachment" in raw.columns:
        out["has_attachment"] = _to_numeric_col(raw["has_attachment"])
    else:
        out["has_attachment"] = 0

    # ── links_count ───────────────────────────────────────────────────────────
    if "links_count" in raw.columns:
        out["links_count"] = _to_numeric_col(raw["links_count"])
    else:
        out["links_count"] = out["email_text"].apply(_count_links)

    # ── urgent_keywords ───────────────────────────────────────────────────────
    if "urgent_keywords" in raw.columns:
        out["urgent_keywords"] = _to_numeric_col(raw["urgent_keywords"])
    else:
        out["urgent_keywords"] = out.apply(
            lambda r: _count_urgent(r["subject"], r["email_text"]), axis=1
        )

    return out


def _discover_csv_files() -> list:
    """Return all CSV filenames in dataset/, sorted alphabetically."""
    return sorted(f for f in os.listdir(_DATASET) if f.lower().endswith(".csv"))


def load_real_world_dataset() -> pd.DataFrame:
    """Merge ALL CSVs found in dataset/ into one clean DataFrame."""
    csv_files = _discover_csv_files()
    print(f"[Train] Found {len(csv_files)} CSV file(s) in dataset/")
    print(f"[Train] Loading datasets …\n")

    parts = []
    for fname in csv_files:
        fpath = os.path.join(_DATASET, fname)
        try:
            part = _load_any_csv(fpath)
            ph = int((part["label"] == 1).sum())
            lg = int((part["label"] == 0).sum())
            print(f"  {fname:30s} → {len(part):7,} rows  phishing={ph:,}  legitimate={lg:,}")
            parts.append(part)
        except Exception as e:
            print(f"  {fname:30s} → SKIPPED ({e})")

    if not parts:
        raise RuntimeError("No valid CSV files loaded from dataset/")

    df = pd.concat(parts, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["email_text", "subject"])
    print(f"\n  Duplicates removed : {before - len(df):,}")

    df = df[~((df["email_text"].str.strip() == "") &
               (df["subject"].str.strip()   == ""))]

    df["has_attachment"]  = df["has_attachment"].astype(int)
    df["links_count"]     = df["links_count"].astype(int)
    df["urgent_keywords"] = df["urgent_keywords"].astype(int)
    df["sender_domain"]   = df["sender_domain"].fillna("")
    df["email_text"]      = df["email_text"].fillna("")
    df["subject"]         = df["subject"].fillna("")

    n_ph = int((df["label"] == 1).sum())
    n_lg = int((df["label"] == 0).sum())
    print(f"\n[Train] Merged total : {len(df):,} rows  phishing={n_ph:,}  legitimate={n_lg:,}")
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def extract_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered features.
    Must stay in-sync with app.py extract_features_from_email().
    """
    df = df.copy()

    df["email_length"]   = df["email_text"].apply(lambda x: len(str(x)))
    df["subject_length"] = df["subject"].apply(lambda x: len(str(x)))
    df["link_density"]   = df["links_count"] / (df["email_length"] + 1)

    df["legitimate_domain"] = df["sender_domain"].apply(
        lambda x: 1 if str(x).lower().strip() in LEGITIMATE_DOMAINS else 0)
    df["suspicious_tld"] = df["sender_domain"].apply(
        lambda x: 1 if any(str(x).lower().strip().endswith(t)
                           for t in SUSPICIOUS_TLDS) else 0)
    df["domain_length"]     = df["sender_domain"].apply(lambda x: len(str(x)))
    df["domain_has_digits"] = df["sender_domain"].apply(
        lambda x: int(any(c.isdigit() for c in str(x))))
    df["domain_has_hyphen"] = df["sender_domain"].apply(
        lambda x: int("-" in str(x)))
    df["domain_age"] = df["sender_domain"].apply(
        lambda x: 30 if str(x).lower().strip() in LEGITIMATE_DOMAINS
        else max(1, abs(hash(str(x))) % 8 + 1))

    def count_ip_urls(text: str) -> int:
        return len(re.findall(
            r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", str(text).lower()))

    def count_shorteners(text: str) -> int:
        t = str(text).lower()
        return sum(1 for s in URL_SHORTENERS if s in t)

    def count_https(text: str) -> int:
        return len(re.findall(r"https://", str(text).lower()))

    def count_http(text: str) -> int:
        return len(re.findall(r"http://", str(text).lower()))

    df["ip_url_count"]        = df["email_text"].apply(count_ip_urls)
    df["shortener_url_count"] = df["email_text"].apply(count_shorteners)
    df["https_url_count"]     = df["email_text"].apply(count_https)
    df["http_url_count"]      = df["email_text"].apply(count_http)
    df["http_ratio"]          = df["http_url_count"] / (df["links_count"] + 1)

    df["special_chars"] = df["email_text"].apply(
        lambda x: len(re.findall(r"[!$%^&*()_+|~=`{}\[\]:\"\'<>?,./]", str(x))))
    df["html_tags"] = df["email_text"].apply(
        lambda x: len(re.findall(r"<[^>]+>", str(x).lower())))

    return df


# ── Preprocessing ──────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "has_attachment", "links_count", "urgent_keywords",
    "email_length", "subject_length", "link_density",
    "domain_age", "special_chars", "html_tags",
    "legitimate_domain", "suspicious_tld",
    "ip_url_count", "shortener_url_count",
    "https_url_count", "http_url_count", "http_ratio",
    "domain_length", "domain_has_digits", "domain_has_hyphen",
]


def build_preprocessor() -> ColumnTransformer:
    text_vec = Pipeline([
        ("hash", HashingVectorizer(
            n_features=2 ** 16, alternate_sign=False,
            stop_words="english", ngram_range=(1, 2),
        ))
    ])
    domain_vec = Pipeline([
        ("hash", HashingVectorizer(n_features=512, alternate_sign=False))
    ])
    return ColumnTransformer(transformers=[
        ("email_text",    text_vec,          "email_text"),
        ("subject",       text_vec,          "subject"),
        ("sender_domain", domain_vec,        "sender_domain"),
        ("num",           StandardScaler(),  NUMERIC_FEATURES),
    ])


# ── Model catalogue ────────────────────────────────────────────────────────────

def get_model_catalogue() -> dict:
    catalogue = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=5,
            subsample=0.8, max_features="sqrt", min_samples_leaf=4,
            random_state=42),
    }
    try:
        from xgboost import XGBClassifier  # noqa: F401
        catalogue["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        pass
    return catalogue


# ── Training entry-point ───────────────────────────────────────────────────────

def train_and_save_model(threshold: float = 0.60) -> dict:
    """
    Load all dataset CSVs, train all models, evaluate at `threshold`,
    save the best one to models/phishing_detection.pkl.
    Returns the evaluation results dict.
    """
    df = load_real_world_dataset()
    df = extract_additional_features(df)

    X = df.drop("label", axis=1)
    y = df["label"]

    n_phish = int(y.sum())
    n_legit = int((y == 0).sum())
    print(f"[Train] Rows: {len(df):,}  |  Phishing: {n_phish:,}  |  Legitimate: {n_legit:,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    w_phish = 1.0
    w_legit = (n_phish / n_legit) * 1.4
    sw_train = np.where(y_train == 1, w_phish, w_legit)

    catalogue = get_model_catalogue()
    results   = {}
    best_name     = None
    best_f1       = -1.0
    best_pipeline = None

    print(f"\n{'Model':<25} {'AUC':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1@0.5':>8} {'F1@{:.2f}'.format(threshold):>8} "
          f"{'FP@{:.2f}'.format(threshold):>8} {'FN@{:.2f}'.format(threshold):>8}")
    print("─" * 85)

    for name, clf in catalogue.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier",   clf),
        ])

        fit_params = {}
        if name in ("Gradient Boosting", "XGBoost"):
            fit_params["classifier__sample_weight"] = sw_train

        pipe.fit(X_train, y_train, **fit_params)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            y_proba   = pipe.predict_proba(X_test)[:, 1]
            y_pred_50 = pipe.predict(X_test)
        y_pred_th = (y_proba >= threshold).astype(int)

        auc   = roc_auc_score(y_test, y_proba)
        acc   = accuracy_score(y_test, y_pred_th)
        prec  = precision_score(y_test, y_pred_th, zero_division=0)
        rec   = recall_score(y_test, y_pred_th, zero_division=0)
        f1_50 = f1_score(y_test, y_pred_50, zero_division=0)
        f1_th = f1_score(y_test, y_pred_th, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_th).ravel()

        results[name] = {
            "auc":       round(float(auc),  4),
            "accuracy":  round(float(acc),  4),
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
            "f1_at_50":  round(float(f1_50), 4),
            f"f1_at_{threshold}":  round(float(f1_th), 4),
            f"fp_at_{threshold}":  int(fp),
            f"fn_at_{threshold}":  int(fn),
            "tn": int(tn), "tp": int(tp),
        }

        print(f"{name:<25} {auc:>6.4f} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} "
              f"{f1_50:>8.4f} {f1_th:>8.4f} {fp:>8d} {fn:>8d}")

        if f1_th > best_f1:
            best_f1       = f1_th
            best_name     = name
            best_pipeline = pipe

    print(f"\n[Train] ★ Best model: {best_name}  (F1@{threshold}={best_f1:.4f})")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_best  = (y_proba_best >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    print(f"\n[Train] === {best_name} — Detailed Report (threshold={threshold}) ===")
    print(classification_report(y_test, y_pred_best,
                                 target_names=["legitimate", "phishing"]))
    print(f"False Positives (legit→phishing): {fp}")
    print(f"False Negatives (phishing→legit): {fn}")

    os.makedirs(os.path.join(_BASE, "models"), exist_ok=True)
    save_path = os.path.join(_BASE, "models", "phishing_detection.pkl")

    csv_files = _discover_csv_files()
    meta = {
        "model_name":    best_name,
        "threshold":     threshold,
        "f1":            best_f1,
        "auc":           results[best_name]["auc"],
        "datasets_used": csv_files,
        "total_rows":    len(df),
        "results_all":   results,
    }
    joblib.dump({"pipeline": best_pipeline, "meta": meta}, save_path, compress=3)
    print(f"[Train] Model saved → {save_path}")
    print(f"[Train] Datasets used: {csv_files}")
    print(f"[Train] Metadata: {json.dumps({k: v for k, v in meta.items() if k != 'results_all'}, indent=2)}")

    return meta


if __name__ == "__main__":
    train_and_save_model()
