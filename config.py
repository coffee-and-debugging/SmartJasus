"""
config.py — Shared constants for CatchFish.

Loaded once at import time by preprocessing.py, features.py,
rules.py, train.py, and app.py.
"""

import json
import os
import re

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "phishing_detection.pkl")


# ── Reference data (loaded from dataset/*.json) ────────────────────────────────

def _load_json_set(filename: str, key: str) -> set:
    path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return set(json.load(f).get(key, []))
    return set()


LEGITIMATE_DOMAINS: set = _load_json_set("legitimate_domains.json", "domains")
SUSPICIOUS_TLDS: set    = _load_json_set("suspicious_tlds.json", "tlds")
URL_SHORTENERS: set     = _load_json_set("url_shorteners.json", "shorteners")


# ── Urgency / deception lexicon ────────────────────────────────────────────────
# Must be identical across train.py, features.py, and app.py.

URGENT_KEYWORDS = [
    "urgent",            "immediate",          "action required",   "verify now",
    "security alert",    "account suspended",  "password expired",  "click here",
    "limited time",      "offer expires",      "verify account",    "confirm identity",
    "unusual activity",  "unauthorized access","your account",      "win a prize",
    "congratulations you","claim now",         "update your",       "log in now",
    "sign in now",       "confirm your",       "verify",            "suspended",
]


# ── Label normalisation ────────────────────────────────────────────────────────

LABEL_MAP = {
    "1": 1, "phishing": 1, "spam": 1, "malicious": 1,
    "0": 0, "legitimate": 0, "ham": 0, "safe": 0, "benign": 0,
}


# ── Compiled regex patterns ────────────────────────────────────────────────────

HTML_TAG_RE   = re.compile(r"<[^>]+>")
MULTI_WS_RE   = re.compile(r"[ \t]+")
URL_RE        = re.compile(r"https?://[^\s]+", re.IGNORECASE)
SENDER_RE     = re.compile(r"[\w.\-+]+@([\w.\-]+\.\w{2,})")
SYNTHETIC_CSV = re.compile(r"^dataset\d+\.csv$", re.IGNORECASE)


# ── Numeric feature names (order matters — must match preprocessor) ────────────

NUMERIC_FEATURES = [
    "has_attachment",     "links_count",        "urgent_keywords",
    "email_length",       "subject_length",     "link_density",
    "domain_age",         "special_chars",      "html_tags",
    "legitimate_domain",  "suspicious_tld",
    "ip_url_count",       "shortener_url_count",
    "https_url_count",    "http_url_count",     "http_ratio",
    "domain_length",      "domain_has_digits",  "domain_has_hyphen",
]
