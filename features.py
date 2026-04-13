"""
features.py — Feature engineering and sklearn preprocessor for CatchFish.

Two public entry points:

  extract_features(df)     — used by train.py on a DataFrame
  build_preprocessor()     — returns the sklearn ColumnTransformer used inside the Pipeline
  extract_email_features() — used by app.py to build a single-row inference dict
  analyze_urls()           — URL breakdown (used by app.py for the API response)
"""

import re

import pandas as pd
import tldextract
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    LEGITIMATE_DOMAINS,
    NUMERIC_FEATURES,
    SUSPICIOUS_TLDS,
    URGENT_KEYWORDS,
    URL_SHORTENERS,
)


# ── Domain helpers ─────────────────────────────────────────────────────────────

def _domain_age_score(domain: str) -> int:
    """
    Proxy for domain trustworthiness.
    Known-good domains → 30 (fixed high score).
    Others → length of the registered-domain label (e.g. 'paypal' = 6).
    Long random labels like 'x7k2q9v3' are a common phishing signal.
    """
    d = str(domain).lower().strip()
    if d in LEGITIMATE_DOMAINS:
        return 30
    reg = tldextract.extract(d).domain
    return max(1, len(reg)) if reg else 1


# ── Batch feature engineering (train-time) ─────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 19 numeric features to df in-place (returns a copy).
    Column order must stay consistent with NUMERIC_FEATURES in config.py.
    """
    df = df.copy()

    # email / subject size
    df["email_length"]   = df["email_text"].apply(lambda x: len(str(x)))
    df["subject_length"] = df["subject"].apply(lambda x: len(str(x)))
    df["link_density"]   = df["links_count"] / (df["email_length"] + 1)

    # domain signals
    df["legitimate_domain"] = df["sender_domain"].apply(
        lambda x: int(str(x).lower().strip() in LEGITIMATE_DOMAINS)
    )
    df["suspicious_tld"] = df["sender_domain"].apply(
        lambda x: int(any(str(x).lower().strip().endswith(t) for t in SUSPICIOUS_TLDS))
    )
    df["domain_length"]     = df["sender_domain"].apply(lambda x: len(str(x)))
    df["domain_has_digits"] = df["sender_domain"].apply(
        lambda x: int(any(c.isdigit() for c in str(x)))
    )
    df["domain_has_hyphen"] = df["sender_domain"].apply(
        lambda x: int("-" in str(x))
    )
    df["domain_age"] = df["sender_domain"].apply(_domain_age_score)

    # URL signals
    _IP_URL  = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
    _HTTPS   = re.compile(r"https://")
    _HTTP    = re.compile(r"http://")

    df["ip_url_count"]        = df["email_text"].apply(
        lambda t: len(_IP_URL.findall(str(t).lower()))
    )
    df["shortener_url_count"] = df["email_text"].apply(
        lambda t: sum(1 for s in URL_SHORTENERS if s in str(t).lower())
    )
    df["https_url_count"] = df["email_text"].apply(
        lambda t: len(_HTTPS.findall(str(t).lower()))
    )
    df["http_url_count"] = df["email_text"].apply(
        lambda t: len(_HTTP.findall(str(t).lower()))
    )
    df["http_ratio"] = df["http_url_count"] / (df["links_count"] + 1)

    # text signals
    _SPECIAL = re.compile(r"""[!$%^&*()_+|~=`{}\[\]:"'<>?,./]""")
    _HTML    = re.compile(r"<[^>]+>")

    df["special_chars"] = df["email_text"].apply(
        lambda x: len(_SPECIAL.findall(str(x)))
    )
    df["html_tags"] = df["email_text"].apply(
        lambda x: len(_HTML.findall(str(x).lower()))
    )

    return df


# ── sklearn preprocessor ───────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Four-branch ColumnTransformer fed into the sklearn Pipeline:

      email_text    → HashingVectorizer  (2^16 buckets, 1-2 grams)
      subject       → HashingVectorizer  (2^16 buckets, 1-2 grams)
      sender_domain → HashingVectorizer  (512 buckets — encodes TLD / depth patterns)
      numeric (19)  → StandardScaler
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


# ── Single-email feature extraction (inference-time) ──────────────────────────

def analyze_urls(text: str) -> dict:
    """
    Break down URLs found in email text.
    Returns counts and sample lists used by the API response.
    """
    text_lower = str(text).lower()
    all_urls = re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", text_lower)

    ip_urls       = [u for u in all_urls if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", u)]
    shortener_urls = [u for u in all_urls if any(s in u for s in URL_SHORTENERS)]
    https_urls    = [u for u in all_urls if u.startswith("https://")]
    http_urls     = [u for u in all_urls if u.startswith("http://")]
    url_domains   = [
        m.group(1)
        for u in all_urls
        for m in [re.search(r"https?://([^/\s]+)", u)]
        if m
    ]

    return {
        "total_urls":           len(all_urls),
        "ip_url_count":         len(ip_urls),
        "shortener_url_count":  len(shortener_urls),
        "https_url_count":      len(https_urls),
        "http_url_count":       len(http_urls),
        "http_ratio":           len(http_urls) / (len(all_urls) + 1),
        "url_domains":          url_domains,
        "raw_urls":             all_urls[:10],
        "ip_urls":              ip_urls[:5],
    }


def extract_email_features(
    email_text,
    subject="",
    has_attachment=None,
    links_count=None,
    sender_domain=None,
    urgent_keywords=None,
) -> tuple[dict, dict]:
    """
    Build the feature dict for a single email at inference time.
    Schema is identical to extract_features() — no train/inference skew.

    Returns:
        features      — dict ready to wrap in pd.DataFrame([features])
        url_analysis  — URL breakdown dict for the API response
    """
    email_text    = str(email_text) if email_text else ""
    subject       = str(subject) if subject else ""
    sender_domain = str(sender_domain).lower().strip() if sender_domain else ""

    url_analysis  = analyze_urls(email_text)

    links_count    = int(links_count) if links_count is not None else url_analysis["total_urls"]
    has_attachment = int(has_attachment) if has_attachment is not None else 0

    if urgent_keywords is None:
        urgent_keywords = int(any(kw in email_text.lower() for kw in URGENT_KEYWORDS))
    else:
        urgent_keywords = int(urgent_keywords)

    # domain signals — subdomains of known-good domains also trusted
    legitimate_domain = int(
        sender_domain in LEGITIMATE_DOMAINS or
        any(sender_domain.endswith("." + d) for d in LEGITIMATE_DOMAINS)
    )
    suspicious_tld    = int(any(sender_domain.endswith(t) for t in SUSPICIOUS_TLDS))
    domain_has_digits = int(any(c.isdigit() for c in sender_domain))
    domain_has_hyphen = int("-" in sender_domain)
    domain_length     = len(sender_domain)
    domain_age        = _domain_age_score(sender_domain)

    email_length   = len(email_text)
    subject_length = len(subject)
    link_density   = links_count / (email_length + 1)
    special_chars  = len(re.findall(r"""[!$%^&*()_+|~=`{}\[\]:"'<>?,./]""", email_text))
    html_tags      = len(re.findall(r"<[^>]+>", email_text.lower()))

    features = {
        "email_text":           email_text,
        "subject":              subject,
        "sender_domain":        sender_domain,
        "has_attachment":       has_attachment,
        "links_count":          links_count,
        "urgent_keywords":      urgent_keywords,
        "email_length":         email_length,
        "subject_length":       subject_length,
        "link_density":         link_density,
        "domain_age":           domain_age,
        "special_chars":        special_chars,
        "html_tags":            html_tags,
        "legitimate_domain":    legitimate_domain,
        "suspicious_tld":       suspicious_tld,
        "ip_url_count":         url_analysis["ip_url_count"],
        "shortener_url_count":  url_analysis["shortener_url_count"],
        "https_url_count":      url_analysis["https_url_count"],
        "http_url_count":       url_analysis["http_url_count"],
        "http_ratio":           url_analysis["http_ratio"],
        "domain_length":        domain_length,
        "domain_has_digits":    domain_has_digits,
        "domain_has_hyphen":    domain_has_hyphen,
    }
    return features, url_analysis
