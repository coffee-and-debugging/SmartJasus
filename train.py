"""
CatchFish — ML Training Pipeline
===================================
Real-world production-grade phishing detection.
Trains multiple classifiers on CSV datasets in dataset/,
selects the best by F1-score at the production threshold,
and saves the winner + full metadata to models/phishing_detection.pkl.

Identical logic is reproduced in CatchFish.ipynb —
both files produce the exact same model given the same data and threshold.

Synthetic files named datasetN.csv are automatically excluded.

Supported dataset schemas (auto-detected per file):
  Schema A — columns: sender, body, subject, label
  Schema B — columns: email_text, sender_domain, has_attachment,
                      links_count, urgent_keywords, label

Reference data loaded from dataset/*.json:
  - legitimate_domains.json
  - suspicious_tlds.json
  - url_shorteners.json

Algorithms evaluated:
  1. Logistic Regression
  2. Random Forest
  3. Extra Trees
  4. Gradient Boosting
  5. XGBoost  (if installed)
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
    average_precision_score,
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

# 24-entry urgency/deception lexicon — must match app.py exactly
URGENT_KEYWORDS = [
    "urgent", "immediate", "action required", "verify now",
    "security alert", "account suspended", "password expired",
    "click here", "limited time", "offer expires", "verify account",
    "confirm identity", "unusual activity", "unauthorized access",
    "your account", "win a prize", "congratulations you", "claim now",
    "update your", "log in now", "sign in now", "confirm your",
    "verify", "suspended",
]

# ── Regex helpers ──────────────────────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"[ \t]+")
_URL_RE      = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_SENDER_RE   = re.compile(r"[\w.\-+]+@([\w.\-]+\.\w{2,})")

# Label normalisation — covers int and text variants
_LABEL_MAP = {
    "1": 1, "phishing": 1, "spam": 1, "malicious": 1,
    "0": 0, "legitimate": 0, "ham": 0, "safe": 0, "benign": 0,
}


# ── Text / data helpers ────────────────────────────────────────────────────────

def _clean_text(text) -> str:
    """Strip HTML tags and normalise whitespace. Preserve URLs as tokens."""
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
    raw.columns = raw.columns.str.strip().str.strip('"').str.strip("'").str.lower()

    out = pd.DataFrame()

    # ── Label → int 0/1 ──────────────────────────────────────────────────────
    lbl = (
        raw["label"]
        .astype(str).str.strip().str.strip('"').str.strip("'").str.lower()
    )
    out["label"] = lbl.map(_LABEL_MAP).fillna(0).astype(int)

    # ── Email text ────────────────────────────────────────────────────────────
    if "email_text" in raw.columns:
        out["email_text"] = raw["email_text"].fillna("").apply(_clean_text)
    elif "body" in raw.columns:
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
        out["sender_domain"] = (
            raw["sender_domain"].astype(str)
            .str.strip().str.strip('"').str.strip("'").fillna("")
        )
    elif "sender" in raw.columns:
        out["sender_domain"] = raw["sender"].fillna("").apply(_extract_domain)
    else:
        out["sender_domain"] = ""

    # ── has_attachment ────────────────────────────────────────────────────────
    out["has_attachment"] = (
        _to_numeric_col(raw["has_attachment"]) if "has_attachment" in raw.columns else 0
    )

    # ── links_count ───────────────────────────────────────────────────────────
    out["links_count"] = (
        _to_numeric_col(raw["links_count"]) if "links_count" in raw.columns
        else out["email_text"].apply(_count_links)
    )

    # ── urgent_keywords ───────────────────────────────────────────────────────
    out["urgent_keywords"] = (
        _to_numeric_col(raw["urgent_keywords"]) if "urgent_keywords" in raw.columns
        else out.apply(lambda r: _count_urgent(r["subject"], r["email_text"]), axis=1)
    )

    return out


_SYNTHETIC_CSV_RE = re.compile(r"^dataset\d+\.csv$", re.IGNORECASE)


def _discover_csv_files() -> list:
    """Return real-world CSV filenames in dataset/, skipping datasetN.csv files."""
    return sorted(
        f for f in os.listdir(_DATASET)
        if f.lower().endswith(".csv") and not _SYNTHETIC_CSV_RE.match(f)
    )


def load_real_world_dataset() -> pd.DataFrame:
    """Merge ALL CSVs found in dataset/ into one clean DataFrame."""
    csv_files = _discover_csv_files()
    print(f"[Train] Found {len(csv_files)} CSV file(s) in dataset/")
    print("[Train] Loading datasets …\n")

    parts = []
    per_dataset = {}
    for fname in csv_files:
        fpath = os.path.join(_DATASET, fname)
        try:
            part = _load_any_csv(fpath)
            ph = int((part["label"] == 1).sum())
            lg = int((part["label"] == 0).sum())
            per_dataset[fname] = {"rows": len(part), "phish": ph, "legit": lg}
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
    Compute all 19 engineered numeric features.
    MUST stay byte-for-byte identical with app.py extract_features_from_email()
    and the Colab notebook — any divergence causes train/inference skew.
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
    """
    Three-branch ColumnTransformer:
      - email_text  → HashingVectorizer (2^16 buckets, 1-2grams)
      - subject     → HashingVectorizer (2^16 buckets, 1-2grams)
      - sender_domain → HashingVectorizer (512 buckets — encodes TLD/depth patterns)
      - 19 numeric features → StandardScaler
    """
    text_vec = Pipeline([
        ("hash", HashingVectorizer(
            n_features=2 ** 16,
            alternate_sign=False,
            stop_words="english",
            ngram_range=(1, 2),
        ))
    ])
    domain_vec = Pipeline([
        ("hash", HashingVectorizer(n_features=512, alternate_sign=False))
    ])
    return ColumnTransformer(transformers=[
        ("email_text",    text_vec,         "email_text"),
        ("subject",       text_vec,         "subject"),
        ("sender_domain", domain_vec,       "sender_domain"),
        ("num",           StandardScaler(), NUMERIC_FEATURES),
    ])


# ── Model catalogue ────────────────────────────────────────────────────────────

def get_model_catalogue() -> dict:
    """
    Five production-grade classifiers.
    All hyperparameters are tuned for phishing detection on ~50k-row tabular+text data.
    class_weight='balanced' compensates for dataset imbalance without oversampling.
    """
    catalogue = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0,
            class_weight="balanced",
            solver="lbfgs", random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20,
            min_samples_leaf=2, min_samples_split=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=20,
            min_samples_leaf=2, min_samples_split=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            max_features="sqrt", min_samples_leaf=2,
            random_state=42,
        ),
    }
    try:
        from xgboost import XGBClassifier
        catalogue["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        pass
    return catalogue


# ── Training entry-point ───────────────────────────────────────────────────────

def train_and_save_model(threshold: float = 0.60) -> dict:
    """
    Full pipeline:
      1. Load & merge all CSVs from dataset/
      2. Engineer 19 numeric features
      3. 80/20 stratified split (random_state=42 — reproducible)
      4. Compute inverse-frequency sample weights for boosting models
      5. Train all classifiers, evaluate at both 0.50 and `threshold`
      6. Select winner by F1@threshold
      7. Save {pipeline, meta} dict to models/phishing_detection.pkl

    Returns the metadata dict (same structure as saved in pkl).
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

    # Inverse-frequency sample weights for boosting models
    # Upweight the minority class by a factor of 1.4
    w_phish = 1.0
    w_legit = (n_phish / n_legit) * 1.4
    sw_train = np.where(y_train == 1, w_phish, w_legit)

    print(f"[Train] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"[Train] Sample weights — phishing: {w_phish:.3f}  legitimate: {w_legit:.4f}\n")

    catalogue = get_model_catalogue()
    results   = {}
    best_name     = None
    best_f1       = -1.0
    best_pipeline = None

    hdr = (f"\n{'Model':<25} {'AUC':>6} {'AP':>6} {'Acc':>6} "
           f"{'Prec':>6} {'Rec':>6} {'F1@0.5':>8} "
           f"{'F1@{:.2f}'.format(threshold):>9} "
           f"{'FP':>6} {'FN':>6}")
    print(hdr)
    print("─" * len(hdr))

    for name, clf in catalogue.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier",   clf),
        ])

        fit_params = {}
        if name in ("Gradient Boosting", "XGBoost"):
            fit_params["classifier__sample_weight"] = sw_train

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            pipe.fit(X_train, y_train, **fit_params)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y_proba   = pipe.predict_proba(X_test)[:, 1]
            y_pred_50 = pipe.predict(X_test)

        y_pred_th = (y_proba >= threshold).astype(int)

        auc   = roc_auc_score(y_test, y_proba)
        ap    = average_precision_score(y_test, y_proba)
        acc   = accuracy_score(y_test, y_pred_th)
        prec  = precision_score(y_test, y_pred_th, zero_division=0)
        rec   = recall_score(y_test, y_pred_th, zero_division=0)
        f1_50 = f1_score(y_test, y_pred_50, zero_division=0)
        f1_th = f1_score(y_test, y_pred_th, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_th).ravel()

        results[name] = {
            "auc":                        round(float(auc),  4),
            "average_precision":          round(float(ap),   4),
            "accuracy":                   round(float(acc),  4),
            "precision":                  round(float(prec), 4),
            "recall":                     round(float(rec),  4),
            "f1_at_50":                   round(float(f1_50), 4),
            f"f1_at_{threshold}":         round(float(f1_th), 4),
            f"fp_at_{threshold}":         int(fp),
            f"fn_at_{threshold}":         int(fn),
            "tn":                         int(tn),
            "tp":                         int(tp),
        }

        print(f"{name:<25} {auc:>6.4f} {ap:>6.4f} {acc:>6.4f} "
              f"{prec:>6.4f} {rec:>6.4f} {f1_50:>8.4f} "
              f"{f1_th:>9.4f} {fp:>6d} {fn:>6d}")

        if f1_th > best_f1:
            best_f1       = f1_th
            best_name     = name
            best_pipeline = pipe

    print(f"\n[Train] ★ Best model: {best_name}  (F1@{threshold}={best_f1:.4f})")

    # Detailed report for best model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_best  = (y_proba_best >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    print(f"\n[Train] === {best_name} — Detailed Report (threshold={threshold}) ===")
    print(classification_report(y_test, y_pred_best,
                                 target_names=["legitimate", "phishing"]))
    print(f"  False Positives (legit→phishing) : {fp}")
    print(f"  False Negatives (phishing→legit) : {fn}")
    print(f"  Average Precision (PR-AUC)        : {results[best_name]['average_precision']:.4f}")

    # Save
    os.makedirs(os.path.join(_BASE, "models"), exist_ok=True)
    save_path = os.path.join(_BASE, "models", "phishing_detection.pkl")

    csv_files = _discover_csv_files()
    meta = {
        "model_name":         best_name,
        "threshold":          threshold,
        "f1":                 best_f1,
        "auc":                results[best_name]["auc"],
        "average_precision":  results[best_name]["average_precision"],
        "datasets_used":      csv_files,
        "total_rows":         len(df),
        "phishing_rows":      int(y.sum()),
        "legitimate_rows":    int((y == 0).sum()),
        "per_dataset":        per_dataset,
        "results_all":        results,
    }
    joblib.dump({"pipeline": best_pipeline, "meta": meta}, save_path, compress=3)
    print(f"\n[Train] Model saved → {save_path}")
    print(f"[Train] Datasets : {csv_files}")
    print(f"[Train] Meta     : {json.dumps({k: v for k, v in meta.items() if k != 'results_all'}, indent=2)}")

    return meta


if __name__ == "__main__":
    import sys
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
    train_and_save_model(threshold=threshold)
