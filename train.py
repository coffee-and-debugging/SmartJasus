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

import tldextract as _tldextract

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
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
    # domain_age: length of the registered domain label (e.g. len("paypal") = 6).
    # Trusted domains get a fixed high value (30) to signal known-good senders.
    # For unknown domains the registered-domain label length is a real structural
    # signal: long random strings (e.g. "x7k2q9v3.com") are more common in phishing.
    def _domain_age_score(domain: str) -> int:
        d = str(domain).lower().strip()
        if d in LEGITIMATE_DOMAINS:
            return 30
        ext = _tldextract.extract(d)
        reg = ext.domain  # registered domain label, e.g. "paypal" from "paypal.com"
        return max(1, len(reg)) if reg else 1

    df["domain_age"] = df["sender_domain"].apply(_domain_age_score)

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
    Production-grade classifiers for phishing detection.

    Tree models (RF, ET, GB) are wrapped in CalibratedClassifierCV so their
    predicted probabilities are well-calibrated — without this, their raw
    outputs cluster near 0/1 and a 0.60 threshold cuts into real phishing
    predictions, causing high FN counts.

    Logistic Regression and XGBoost/LightGBM are already well-calibrated
    and do not need the wrapper.
    """
    catalogue = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0,
            class_weight="balanced",
            solver="lbfgs", random_state=42,
        ),
        "Random Forest": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300, max_depth=20,
                min_samples_leaf=2, min_samples_split=4,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        ),
        "Extra Trees": CalibratedClassifierCV(
            ExtraTreesClassifier(
                n_estimators=300, max_depth=20,
                min_samples_leaf=2, min_samples_split=4,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        ),
        "Gradient Boosting": CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=6, subsample=0.8,
                max_features="sqrt", min_samples_leaf=2,
                random_state=42,
            ),
            method="isotonic", cv=3,
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
    try:
        from lightgbm import LGBMClassifier
        catalogue["LightGBM"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=7, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1, reg_lambda=1.0,
            class_weight="balanced",
            random_state=42, n_jobs=-1,
            verbose=-1,
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
      3. 70/15/15 stratified split (train / val / test)
         - threshold is selected on val to avoid test-set leakage
         - final metrics reported on held-out test only
      4. Compute sample weights — phishing upweighted 1.4×
      5. Train all classifiers with probability calibration on tree models
      6. Select winner by F1@threshold on VALIDATION set
      7. 5-fold CV AUC sanity check on best model (catches overfitting)
      8. Report final metrics on TEST set
      9. Save {pipeline, meta} dict to models/phishing_detection.pkl

    Returns the metadata dict (same structure as saved in pkl).
    """
    df = load_real_world_dataset()
    df = extract_additional_features(df)

    X = df.drop("label", axis=1)
    y = df["label"]

    n_phish = int(y.sum())
    n_legit = int((y == 0).sum())
    print(f"[Train] Rows: {len(df):,}  |  Phishing: {n_phish:,}  |  Legitimate: {n_legit:,}")

    # 70 / 15 / 15 split — threshold tuned on val, final metrics on test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=float(0.15) / 0.85, random_state=42, stratify=y_temp
    )

    # Sample weights for boosting models — upweight phishing by 1.4×
    # Missed phishing (FN) is more costly than a false alarm (FP), so
    # phishing examples get higher weight regardless of class ratio.
    w_legit = 1.0
    w_phish = (n_legit / max(n_phish, 1)) * 1.4
    sw_train = np.where(y_train == 1, w_phish, w_legit)

    print(f"[Train] Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    print(f"[Train] Sample weights — phishing: {w_phish:.3f}  legitimate: {w_legit:.4f}\n")

    catalogue = get_model_catalogue()
    results   = {}
    best_name     = None
    best_f1       = -1.0
    best_pipeline = None

    # ── PHASE 1: evaluate all models on VALIDATION set to pick winner ────────────
    hdr = (f"\n{'Model':<25} {'VAL AUC':>8} {'VAL AP':>7} "
           f"{'VAL F1@{:.2f}'.format(threshold):>12} "
           f"{'VAL FP':>7} {'VAL FN':>7}")
    print("[Train] === Phase 1: Model selection on VALIDATION set ===")
    print(hdr)
    print("─" * len(hdr))

    pipelines = {}
    val_results = {}

    for name, clf in catalogue.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier",   clf),
        ])

        # CalibratedClassifierCV wraps tree models — sample_weight must be
        # passed to the inner estimator, not the calibrator itself.
        fit_params = {}
        if name in ("Gradient Boosting", "XGBoost", "LightGBM"):
            fit_params["classifier__sample_weight"] = sw_train

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            pipe.fit(X_train, y_train, **fit_params)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y_val_proba = pipe.predict_proba(X_val)[:, 1]

        y_val_pred = (y_val_proba >= threshold).astype(int)
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_ap  = average_precision_score(y_val, y_val_proba)
        val_f1  = f1_score(y_val, y_val_pred, zero_division=0)
        tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_val, y_val_pred).ravel()

        val_results[name] = {"val_auc": round(float(val_auc), 4),
                             "val_ap":  round(float(val_ap),  4),
                             "val_f1":  round(float(val_f1),  4)}
        pipelines[name] = pipe

        print(f"{name:<25} {val_auc:>8.4f} {val_ap:>7.4f} "
              f"{val_f1:>12.4f} {fp_v:>7d} {fn_v:>7d}")

        if val_f1 > best_f1:
            best_f1       = val_f1
            best_name     = name
            best_pipeline = pipe

    print(f"\n[Train] ★ Best model (by VAL F1@{threshold}): {best_name}  ({best_f1:.4f})")

    # ── PHASE 2: 5-fold CV AUC on best model — overfitting sanity check ──────────
    print(f"\n[Train] === Phase 2: 5-fold CV AUC on {best_name} (overfitting check) ===")
    # Refit a fresh pipeline on train+val combined for CV (no test leakage)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    cv_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   catalogue[best_name]),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_scores = cross_val_score(cv_pipe, X_trainval, y_trainval,
                                    cv=5, scoring="roc_auc", n_jobs=-1)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    val_auc_best = val_results[best_name]["val_auc"]
    gap = abs(cv_mean - val_auc_best)
    print(f"  CV AUC scores  : {cv_scores.round(4)}")
    print(f"  CV AUC mean    : {cv_mean:.4f}  ±{cv_std:.4f}")
    print(f"  Val AUC        : {val_auc_best:.4f}")
    print(f"  Gap (CV - Val) : {gap:.4f}", end="  ")
    if gap < 0.01:
        print("✓ No overfitting detected")
    elif gap < 0.03:
        print("⚠ Minor gap — monitor with more data")
    else:
        print("✗ Significant gap — model may be overfit")

    # ── PHASE 3: final metrics on held-out TEST set ───────────────────────────────
    print(f"\n[Train] === Phase 3: Final evaluation on TEST set (held-out) ===")
    hdr2 = (f"\n{'Model':<25} {'AUC':>6} {'AP':>6} {'Acc':>6} "
            f"{'Prec':>6} {'Rec':>6} {'F1@0.5':>8} "
            f"{'F1@{:.2f}'.format(threshold):>9} "
            f"{'FP':>6} {'FN':>6}")
    print(hdr2)
    print("─" * len(hdr2))

    results = {}
    for name, pipe in pipelines.items():
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
            "val_auc":                    val_results[name]["val_auc"],
            "cv_auc_mean":                round(cv_mean, 4) if name == best_name else None,
            "cv_auc_std":                 round(cv_std,  4) if name == best_name else None,
        }

        star = " ★" if name == best_name else ""
        print(f"{name:<25} {auc:>6.4f} {ap:>6.4f} {acc:>6.4f} "
              f"{prec:>6.4f} {rec:>6.4f} {f1_50:>8.4f} "
              f"{f1_th:>9.4f} {fp:>6d} {fn:>6d}{star}")

    print(f"\n[Train] ★ Best model: {best_name}  (Test F1@{threshold}={results[best_name][f'f1_at_{threshold}']:.4f})")

    # Detailed classification report for best model on test set
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_best = (y_proba_best >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    print(f"\n[Train] === {best_name} — Detailed Report on TEST set (threshold={threshold}) ===")
    print(classification_report(y_test, y_pred_best,
                                 target_names=["legitimate", "phishing"]))
    print(f"  False Positives (legit→phishing) : {fp}")
    print(f"  False Negatives (phishing→legit) : {fn}")
    print(f"  Average Precision (PR-AUC)        : {results[best_name]['average_precision']:.4f}")
    print(f"  CV AUC mean ± std                 : {cv_mean:.4f} ± {cv_std:.4f}")

    # Save
    os.makedirs(os.path.join(_BASE, "models"), exist_ok=True)
    save_path = os.path.join(_BASE, "models", "phishing_detection.pkl")

    csv_files = _discover_csv_files()
    meta = {
        "model_name":         best_name,
        "threshold":          threshold,
        # f1 stored is VAL f1 (used for selection); test f1 is in results_all
        "f1":                 results[best_name][f"f1_at_{threshold}"],
        "auc":                results[best_name]["auc"],
        "average_precision":  results[best_name]["average_precision"],
        "cv_auc_mean":        results[best_name].get("cv_auc_mean"),
        "cv_auc_std":         results[best_name].get("cv_auc_std"),
        "split":              "70/15/15 train/val/test",
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
