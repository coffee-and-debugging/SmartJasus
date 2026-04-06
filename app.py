"""
SmartJasus — Flask Application
================================
Serves the phishing detection API, local SMTP mail server,
dashboard UI, and all data-access endpoints.
"""

import json
import logging
import os
import re
import smtplib
import socket
import warnings
import imaplib
from datetime import datetime, timezone
from email import policy
from email.header import decode_header
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime

import joblib
import numpy as np
import pandas as pd
import tldextract
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from local_mail_server import MailStore, LocalMailHandler, LocalSMTPServer

load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Environment ────────────────────────────────────────────────────────────────
LOCAL_SMTP_HOST = os.getenv("LOCAL_SMTP_HOST", "127.0.0.1")
LOCAL_SMTP_PORT = int(os.getenv("LOCAL_SMTP_PORT", "1025"))
LOCAL_DOMAIN = os.getenv("LOCAL_DOMAIN", "local.com")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "").strip()
EMAIL_APP_PASSWORD = "".join(os.getenv("EMAIL_APP_PASSWORD", "").split())
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com").strip()
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
IMAP_FOLDER = os.getenv("IMAP_FOLDER", "INBOX").strip()
AUTO_SYNC_ON_START = os.getenv("AUTO_SYNC_ON_START", "false").lower() == "true"
APP_HOST = os.getenv("APP_HOST", "0.0.0.0").strip()
APP_PORT = int(os.getenv("APP_PORT", "5000"))
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
PHISHING_THRESHOLD = float(os.getenv("PHISHING_THRESHOLD", "0.60"))
TRUSTED_DOMAIN_REDUCTION = float(os.getenv("TRUSTED_DOMAIN_REDUCTION", "0.18"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "emailserver")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ── Load reference data from datasets/ ─────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.join(_BASE, "dataset")


def _load_json_set(filename: str, key: str) -> set:
    path = os.path.join(_DATASET, filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return set(json.load(f).get(key, []))
    return set()


LEGITIMATE_DOMAINS: set = _load_json_set("legitimate_domains.json", "domains")
SUSPICIOUS_TLDS: set = _load_json_set("suspicious_tlds.json", "tlds")
URL_SHORTENERS: set = _load_json_set("url_shorteners.json", "shorteners")

# Ensure local domain is always trusted
LEGITIMATE_DOMAINS.add(LOCAL_DOMAIN)

URGENT_PHRASES = [
    "urgent", "immediate", "action required", "verify now",
    "security alert", "account suspended", "password expired",
    "click here", "limited time", "offer expires", "verify account",
    "confirm identity", "unusual activity", "unauthorized access",
    "your account", "win a prize", "congratulations you", "claim now",
    "update your", "log in now", "sign in now", "confirm your",
]

# ── Database & services ────────────────────────────────────────────────────────
mail_store = MailStore(
    db_host=DB_HOST, db_port=DB_PORT,
    db_name=DB_NAME, db_user=DB_USER, db_password=DB_PASSWORD,
)
local_smtp_server = None

print("[SmartJasus] Server starting …")

# ── Model loading ──────────────────────────────────────────────────────────────
_model_pipeline = None
_model_meta = {}


def load_model() -> None:
    global _model_pipeline, _model_meta
    model_path = os.path.join(_BASE, "models", "phishing_detection.pkl")
    if not os.path.exists(model_path):
        from train import train_and_save_model
        train_and_save_model(threshold=PHISHING_THRESHOLD)

    payload = joblib.load(model_path)

    # Support both old format (Pipeline) and new format (dict with pipeline+meta)
    if isinstance(payload, dict) and "pipeline" in payload:
        _model_pipeline = payload["pipeline"]
        _model_meta = payload.get("meta", {})
    else:
        _model_pipeline = payload  # legacy plain pipeline
        _model_meta = {"model_name": "legacy", "threshold": PHISHING_THRESHOLD}

    print(f"[SmartJasus] Model loaded: {_model_meta.get('model_name','unknown')}  "
          f"AUC={_model_meta.get('auc','—')}  threshold={PHISHING_THRESHOLD}")


load_model()


# ── Feature extraction ─────────────────────────────────────────────────────────

def analyze_urls(text: str) -> dict:
    text_lower = str(text).lower()
    all_urls = re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", text_lower)

    ip_urls = [u for u in all_urls if re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", u)]
    shortener_urls = [u for u in all_urls if any(s in u for s in URL_SHORTENERS)]
    https_urls = [u for u in all_urls if u.startswith("https://")]
    http_urls = [u for u in all_urls if u.startswith("http://")]

    url_domains = []
    for url in all_urls:
        m = re.search(r"https?://([^/\s]+)", url)
        if m:
            url_domains.append(m.group(1))

    return {
        "total_urls": len(all_urls),
        "ip_url_count": len(ip_urls),
        "shortener_url_count": len(shortener_urls),
        "https_url_count": len(https_urls),
        "http_url_count": len(http_urls),
        "http_ratio": len(http_urls) / (len(all_urls) + 1),
        "url_domains": url_domains,
        "raw_urls": all_urls[:10],
        "ip_urls": ip_urls[:5],
    }


def extract_features_from_email(
    email_text, subject="", has_attachment=None,
    links_count=None, sender_domain=None, urgent_keywords=None,
) -> tuple:
    """Return (features_dict, url_analysis_dict). Matches train.py schema exactly."""
    email_text = str(email_text) if email_text else ""
    subject = str(subject) if subject else ""
    sender_domain = str(sender_domain).lower().strip() if sender_domain else ""

    url_analysis = analyze_urls(email_text)

    links_count = int(links_count) if links_count is not None else url_analysis["total_urls"]
    has_attachment = int(has_attachment) if has_attachment is not None else 0

    if urgent_keywords is None:
        urgent_keywords = int(any(phrase in email_text.lower() for phrase in URGENT_PHRASES))
    else:
        urgent_keywords = int(urgent_keywords)

    legitimate_domain = 1 if sender_domain in LEGITIMATE_DOMAINS else 0
    suspicious_tld = 1 if any(sender_domain.endswith(t) for t in SUSPICIOUS_TLDS) else 0
    domain_has_digits = int(any(c.isdigit() for c in sender_domain))
    domain_has_hyphen = int("-" in sender_domain)
    domain_length = len(sender_domain)
    domain_age = 30 if sender_domain in LEGITIMATE_DOMAINS else max(1, abs(hash(sender_domain)) % 8 + 1)

    email_length = len(email_text)
    subject_length = len(subject)
    link_density = links_count / (email_length + 1)
    special_chars = len(re.findall(r"[!$%^&*()_+|~=`{}\[\]:\"\'<>?,./]", email_text))
    html_tags = len(re.findall(r"<[^>]+>", email_text.lower()))

    features = {
        "email_text": email_text,
        "subject": subject,
        "sender_domain": sender_domain,
        "has_attachment": has_attachment,
        "links_count": links_count,
        "urgent_keywords": urgent_keywords,
        "email_length": email_length,
        "subject_length": subject_length,
        "link_density": link_density,
        "domain_age": domain_age,
        "special_chars": special_chars,
        "html_tags": html_tags,
        "legitimate_domain": legitimate_domain,
        "suspicious_tld": suspicious_tld,
        "ip_url_count": url_analysis["ip_url_count"],
        "shortener_url_count": url_analysis["shortener_url_count"],
        "https_url_count": url_analysis["https_url_count"],
        "http_url_count": url_analysis["http_url_count"],
        "http_ratio": url_analysis["http_ratio"],
        "domain_length": domain_length,
        "domain_has_digits": domain_has_digits,
        "domain_has_hyphen": domain_has_hyphen,
    }
    return features, url_analysis


def apply_rule_adjustments(raw_probability: float, features: dict) -> tuple:
    """Hybrid rule engine. Returns (adjusted_probability, rules_list)."""
    prob = raw_probability
    rules = []
    domain = features.get("sender_domain", "")

    if features.get("legitimate_domain", 0) == 1 and domain:
        old = prob
        prob = max(0.05, prob - TRUSTED_DOMAIN_REDUCTION)
        rules.append(f"Trusted domain '{domain}': -{TRUSTED_DOMAIN_REDUCTION:.2f} ({old:.3f}→{prob:.3f})")

    if features.get("ip_url_count", 0) > 0:
        old = prob
        prob = min(0.99, prob + 0.20)
        rules.append(f"IP-based URL detected: +0.20 ({old:.3f}→{prob:.3f})")

    if features.get("suspicious_tld", 0) == 1:
        old = prob
        prob = min(0.99, prob + 0.12)
        rules.append(f"Suspicious TLD: +0.12 ({old:.3f}→{prob:.3f})")

    if features.get("shortener_url_count", 0) > 0:
        old = prob
        prob = min(0.99, prob + 0.10)
        rules.append(f"URL shortener detected: +0.10 ({old:.3f}→{prob:.3f})")

    if features.get("domain_has_digits", 0) == 1 and features.get("legitimate_domain", 0) == 0:
        old = prob
        prob = min(0.99, prob + 0.06)
        rules.append(f"Digits in sender domain: +0.06 ({old:.3f}→{prob:.3f})")

    return prob, rules


def predict_from_payload(data: dict) -> dict:
    """Full pipeline: features → ML → rules → verdict. Stores all details."""
    features, url_analysis = extract_features_from_email(
        email_text=data.get("email_text", ""),
        subject=data.get("subject", ""),
        has_attachment=data.get("has_attachment"),
        links_count=data.get("links_count"),
        sender_domain=data.get("sender_domain", ""),
        urgent_keywords=data.get("urgent_keywords"),
    )

    input_df = pd.DataFrame([features])
    raw_proba_arr = _model_pipeline.predict_proba(input_df)
    raw_probability = float(raw_proba_arr[0][1])

    adjusted_probability, rules_applied = apply_rule_adjustments(raw_probability, features)
    prediction = "phishing" if adjusted_probability >= PHISHING_THRESHOLD else "legitimate"
    confidence = adjusted_probability if prediction == "phishing" else (1.0 - adjusted_probability)

    display_features = {k: v for k, v in features.items() if k not in ("email_text", "subject")}

    return {
        "prediction": prediction,
        "probability": adjusted_probability,
        "raw_probability": raw_probability,
        "confidence": confidence,
        "threshold": PHISHING_THRESHOLD,
        "model_used": _model_meta.get("model_name", "unknown"),
        "rule_adjustments": rules_applied,
        "features_used": display_features,
        "url_analysis": {
            "total_urls": url_analysis["total_urls"],
            "ip_url_count": url_analysis["ip_url_count"],
            "shortener_url_count": url_analysis["shortener_url_count"],
            "raw_urls": url_analysis["raw_urls"],
            "ip_urls": url_analysis.get("ip_urls", []),
        },
    }


# ── Email parsing helpers ──────────────────────────────────────────────────────

def _decode_mime(value: str) -> str:
    if not value:
        return ""
    parts = []
    for fragment, encoding in decode_header(value):
        if isinstance(fragment, bytes):
            parts.append(fragment.decode(encoding or "utf-8", errors="replace"))
        else:
            parts.append(fragment)
    return "".join(parts)


def _parse_imap_date(meta) -> str | None:
    if not meta:
        return None
    if isinstance(meta, bytes):
        meta = meta.decode(errors="ignore")
    m = re.search(r'INTERNALDATE "([^"]+)"', str(meta))
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%d-%b-%Y %H:%M:%S %z")
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _parse_email_message(message, received_at_override=None) -> dict:
    sender = _decode_mime(message.get("From", "unknown@local"))
    recipient = _decode_mime(message.get("To", EMAIL_ADDRESS or "unknown@local"))
    subject = _decode_mime(message.get("Subject", ""))

    body = ""
    has_attachment = 0
    if message.is_multipart():
        parts = []
        for part in message.walk():
            if part.get_content_disposition() == "attachment":
                has_attachment = 1
                continue
            if part.get_content_type() == "text/plain":
                parts.append(part.get_content() or "")
        body = "\n".join(p for p in parts if p)
    else:
        body = message.get_content() or ""

    addr = parseaddr(sender)[1]
    sender_domain = addr.split("@", 1)[1].lower() if "@" in addr else ""

    received_at = received_at_override
    if not received_at and (dh := message.get("Date")):
        try:
            received_at = parsedate_to_datetime(dh).astimezone(timezone.utc).isoformat()
        except Exception:
            pass
    if not received_at:
        received_at = datetime.now(timezone.utc).isoformat()

    return {
        "sender": sender, "recipient": recipient, "subject": subject,
        "body": body, "has_attachment": has_attachment,
        "sender_domain": sender_domain, "received_at": received_at,
    }


def scan_and_store_email(payload: dict, source_uid=None, source_mailbox=None) -> int | None:
    result = predict_from_payload({
        "email_text": payload["body"],
        "subject": payload["subject"],
        "has_attachment": payload["has_attachment"],
        "sender_domain": payload["sender_domain"],
    })
    features_json = json.dumps(result.get("features_used", {}))
    email_id = mail_store.save_email(
        sender=payload["sender"],
        recipient=payload["recipient"] or (EMAIL_ADDRESS or "unknown@local"),
        subject=payload["subject"],
        body=payload["body"],
        has_attachment=payload["has_attachment"],
        links_count=result["features_used"].get("links_count", 0),
        sender_domain=payload["sender_domain"],
        received_at=payload.get("received_at"),
        source_uid=source_uid,
        source_mailbox=source_mailbox,
        prediction=result["prediction"],
        probability=result["probability"],
        confidence=result["confidence"],
        features_json=features_json,
    )
    if email_id:
        mail_store.save_scan_log(
            email_id=email_id,
            prediction=result["prediction"],
            raw_probability=result.get("raw_probability", result["probability"]),
            adjusted_probability=result["probability"],
            confidence=result["confidence"],
            rule_adjustments=json.dumps(result.get("rule_adjustments", [])),
            features_snapshot=features_json,
        )
    return email_id


def sync_inbox(limit: int = 20) -> int:
    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        raise ValueError("EMAIL_ADDRESS and EMAIL_APP_PASSWORD must be set in .env")
    imported = 0
    with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
        imap.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        imap.select(IMAP_FOLDER, readonly=True)
        status, data = imap.uid("search", None, "ALL")
        if status != "OK" or not data or not data[0]:
            return imported
        uids = data[0].split()[-max(1, limit):]
        for uid in uids:
            status, msg_data = imap.uid("fetch", uid, "(RFC822 INTERNALDATE)")
            if status != "OK" or not msg_data:
                continue
            raw_email = received_at = None
            for chunk in msg_data:
                if isinstance(chunk, tuple) and len(chunk) >= 2:
                    raw_email = chunk[1]
                    received_at = _parse_imap_date(chunk[0])
                    break
            if not raw_email:
                continue
            message = BytesParser(policy=policy.default).parsebytes(raw_email)
            parsed = _parse_email_message(message, received_at_override=received_at)
            row_id = scan_and_store_email(
                parsed, source_uid=f"imap:{uid.decode()}", source_mailbox=IMAP_FOLDER
            )
            if row_id:
                imported += 1
    return imported


def start_local_smtp_server() -> None:
    global local_smtp_server
    if local_smtp_server is not None:
        return
    handler = LocalMailHandler(mail_store, predict_from_payload)
    local_smtp_server = LocalSMTPServer(handler, host=LOCAL_SMTP_HOST, port=LOCAL_SMTP_PORT)
    try:
        local_smtp_server.start()
    except OSError as exc:
        if getattr(exc, "errno", None) == 98:
            local_smtp_server = None
        else:
            raise


# ── UI Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/mail-dashboard")
def mail_dashboard():
    return render_template("mail_dashboard.html", default_from_address=EMAIL_ADDRESS)


@app.route("/web-icon.png")
def web_icon():
    return send_from_directory("templates", "icon.png")


# ── Prediction Routes ──────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        resp = jsonify({"status": "preflight"})
        resp.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST",
        })
        return resp
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    try:
        return jsonify(predict_from_payload(data))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/detect", methods=["POST"])
def detect_alias():
    """Backward-compatible alias for browser extension."""
    data = request.get_json() or {}
    result = predict_from_payload({
        "email_text": data.get("email_text") or data.get("text", ""),
        "subject": data.get("subject", ""),
        "has_attachment": data.get("has_attachment"),
        "links_count": data.get("links_count"),
        "sender_domain": data.get("sender_domain", ""),
        "urgent_keywords": data.get("urgent_keywords"),
    })
    return jsonify({"result": result["prediction"], **result})


# ── Local Mail Routes ──────────────────────────────────────────────────────────

@app.route("/api/send-local", methods=["POST"])
def send_local_email():
    """Send email through the local SMTP server (@local.com)."""
    data = request.get_json() or {}
    from_addr = (data.get("from_address") or EMAIL_ADDRESS or "").strip()
    to_addr = (data.get("to_address") or "").strip()
    subject = data.get("subject", "No Subject")
    body = data.get("body", "")

    if not from_addr:
        return jsonify({"error": "from_address is required"}), 400
    if not to_addr:
        return jsonify({"error": "to_address is required"}), 400

    try:
        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(LOCAL_SMTP_HOST, LOCAL_SMTP_PORT, timeout=10) as smtp:
            smtp.send_message(msg)
        return jsonify({"status": "sent", "from": from_addr, "to": to_addr})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/send-email", methods=["POST"])
def send_email():
    """Send email through Gmail SMTP."""
    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        return jsonify({"error": "EMAIL_ADDRESS and EMAIL_APP_PASSWORD must be set"}), 400
    data = request.get_json() or {}
    to_addr = (data.get("to_address") or "").strip()
    if not to_addr:
        return jsonify({"error": "to_address is required"}), 400
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_addr
        msg["Subject"] = data.get("subject", "")
        msg.set_content(data.get("body", ""))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            smtp.ehlo()
            if SMTP_USE_TLS:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        return jsonify({"status": "sent", "from": EMAIL_ADDRESS, "to": to_addr})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── User Management ────────────────────────────────────────────────────────────

@app.route("/api/users", methods=["GET"])
def get_users():
    return jsonify({"users": mail_store.get_users(), "domain": LOCAL_DOMAIN})


@app.route("/api/users/add", methods=["POST"])
def add_user():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip().lower()
    if not username or not re.match(r"^[a-z0-9._-]+$", username):
        return jsonify({"error": "Invalid username"}), 400
    email = f"{username}@{LOCAL_DOMAIN}"
    added = mail_store.add_user(username, email)
    return jsonify({"status": "created" if added else "exists", "email": email})


# ── Data / Inbox Routes ────────────────────────────────────────────────────────

@app.route("/api/sync-inbox", methods=["POST", "GET"])
def sync_inbox_endpoint():
    try:
        limit = max(1, min(int(request.args.get("limit", 20)), 1000))
        imported = sync_inbox(limit=limit)
        return jsonify({"status": "ok", "imported": imported, "folder": IMAP_FOLDER})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/mail-config", methods=["GET"])
def mail_config_status():
    return jsonify({
        "email_address_set": bool(EMAIL_ADDRESS),
        "app_password_set": bool(EMAIL_APP_PASSWORD),
        "smtp_host": SMTP_HOST, "smtp_port": SMTP_PORT,
        "imap_host": IMAP_HOST, "imap_port": IMAP_PORT,
        "imap_folder": IMAP_FOLDER,
        "auto_sync_on_start": AUTO_SYNC_ON_START,
        "local_domain": LOCAL_DOMAIN,
        "phishing_threshold": PHISHING_THRESHOLD,
        "virustotal_enabled": bool(VIRUSTOTAL_API_KEY),
        "model_name": _model_meta.get("model_name", "unknown"),
        "model_auc": _model_meta.get("auc", None),
    })


@app.route("/api/local-server-status", methods=["GET"])
def local_server_status():
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.settimeout(0.5)
    try:
        result = sc.connect_ex((LOCAL_SMTP_HOST, LOCAL_SMTP_PORT))
        return jsonify({
            "local_server_up": result == 0,
            "host": LOCAL_SMTP_HOST,
            "port": LOCAL_SMTP_PORT,
        })
    finally:
        sc.close()


@app.route("/api/mailbox/<path:recipient>", methods=["GET"])
def mailbox(recipient):
    limit = max(1, min(int(request.args.get("limit", 50)), 5000))
    return jsonify({"recipient": recipient, "items": mail_store.get_mailbox(recipient, limit=limit)})


@app.route("/api/alerts", methods=["GET"])
def alerts():
    limit = max(1, min(int(request.args.get("limit", 100)), 500))
    return jsonify({"items": mail_store.get_alerts(limit=limit)})


@app.route("/api/activity", methods=["GET"])
def activity():
    limit = max(1, min(int(request.args.get("limit", 100)), 5000))
    return jsonify({"items": mail_store.get_recent_activity(limit=limit)})


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Detailed scan logs with rule adjustments and full feature snapshots."""
    limit = max(1, min(int(request.args.get("limit", 200)), 1000))
    return jsonify({"logs": mail_store.get_scan_logs(limit=limit)})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(mail_store.get_stats())


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return trained model metadata."""
    return jsonify({
        "model_name": _model_meta.get("model_name", "unknown"),
        "threshold": _model_meta.get("threshold", PHISHING_THRESHOLD),
        "auc": _model_meta.get("auc"),
        "f1": _model_meta.get("f1"),
        "results_all": _model_meta.get("results_all", {}),
    })


# ── Domain Reputation ──────────────────────────────────────────────────────────

@app.route("/api/domain-reputation", methods=["POST"])
def domain_reputation():
    data = request.get_json() or {}
    domain = (data.get("domain") or "").strip().lower()
    if not domain:
        return jsonify({"error": "domain is required"}), 400

    ext = tldextract.extract(domain)
    main_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else domain

    is_trusted = domain in LEGITIMATE_DOMAINS or main_domain in LEGITIMATE_DOMAINS
    has_susp_tld = any(domain.endswith(t) for t in SUSPICIOUS_TLDS)
    has_digits = any(c.isdigit() for c in (ext.domain or ""))
    has_hyphen = "-" in (ext.domain or "")
    domain_len = len(main_domain)

    # Risk score: 50 = neutral baseline
    risk_score = 50
    risk_flags = []

    if is_trusted:
        risk_score -= 40
        risk_flags.append("Known legitimate domain")
    if has_susp_tld:
        risk_score += 30
        risk_flags.append("Suspicious TLD")
    if has_digits:
        risk_score += 15
        risk_flags.append("Digits in domain name")
    if has_hyphen:
        risk_score += 10
        risk_flags.append("Hyphen in domain name")
    if domain_len > 30:
        risk_score += 20
        risk_flags.append("Unusually long domain name")
    if ext.subdomain and len(ext.subdomain.split(".")) > 2:
        risk_score += 10
        risk_flags.append("Deep subdomain chain")

    risk_score = max(0, min(100, risk_score))
    risk_level = "low" if risk_score < 40 else ("medium" if risk_score < 70 else "high")

    result = {
        "domain": domain,
        "main_domain": main_domain,
        "is_trusted": is_trusted,
        "has_suspicious_tld": has_susp_tld,
        "has_digits": has_digits,
        "has_hyphen": has_hyphen,
        "domain_length": domain_len,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_flags": risk_flags,
        "virustotal": None,
    }

    # VirusTotal lookup
    if VIRUSTOTAL_API_KEY:
        try:
            import requests as req_lib
            vt_resp = req_lib.get(
                f"https://www.virustotal.com/api/v3/domains/{main_domain}",
                headers={"x-apikey": VIRUSTOTAL_API_KEY},
                timeout=10,
            )
            if vt_resp.status_code == 200:
                attrs = vt_resp.json().get("data", {}).get("attributes", {})
                stats = attrs.get("last_analysis_stats", {})
                cats = attrs.get("categories", {})
                result["virustotal"] = {
                    "malicious": stats.get("malicious", 0),
                    "suspicious": stats.get("suspicious", 0),
                    "clean": stats.get("harmless", 0),
                    "undetected": stats.get("undetected", 0),
                    "categories": list(set(cats.values()))[:4],
                    "reputation": attrs.get("reputation", None),
                }
                # Boost risk score if VT shows malicious
                if stats.get("malicious", 0) > 0:
                    result["risk_score"] = min(100, result["risk_score"] + stats["malicious"] * 5)
                    result["risk_flags"].append(f"VirusTotal: {stats['malicious']} engine(s) flagged malicious")
                    result["risk_level"] = "high"
            elif vt_resp.status_code == 404:
                result["virustotal"] = {"error": "Domain not found in VirusTotal database"}
            elif vt_resp.status_code == 401:
                result["virustotal"] = {"error": "Invalid VirusTotal API key"}
            else:
                result["virustotal"] = {"error": f"VT API returned HTTP {vt_resp.status_code}"}
        except Exception as vt_err:
            result["virustotal"] = {"error": str(vt_err)}

    return jsonify(result)


# ── Legacy persist endpoint ────────────────────────────────────────────────────

@app.route("/api/persist-records", methods=["POST"])
def persist_records():
    data = request.get_json() or {}
    records = data.get("records", [])
    if not records:
        return jsonify({"status": "ok", "persisted": 0})
    persisted = 0
    for record in records:
        try:
            if not mail_store.get_email_by_fingerprint(
                sender=record.get("sender", ""),
                subject=record.get("subject", ""),
                received_at=record.get("received_at", ""),
            ):
                mail_store.save_email(
                    sender=record.get("sender", ""),
                    recipient=record.get("recipient", ""),
                    subject=record.get("subject", ""),
                    body=record.get("body", ""),
                    has_attachment=int(record.get("has_attachment", 0)),
                    links_count=int(record.get("links_count", 0)),
                    sender_domain=record.get("sender_domain", ""),
                    prediction=record.get("prediction", "legitimate"),
                    probability=float(record.get("probability", 0)),
                    confidence=float(record.get("confidence", 0)),
                )
                persisted += 1
        except Exception:
            continue
    return jsonify({"status": "ok", "persisted": persisted})


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_local_smtp_server()
        if AUTO_SYNC_ON_START:
            try:
                sync_inbox(limit=20)
            except Exception:
                pass
        print(f"[SmartJasus] Web    : http://127.0.0.1:{APP_PORT}")
        print(f"[SmartJasus] SMTP   : {LOCAL_SMTP_HOST}:{LOCAL_SMTP_PORT}  (@{LOCAL_DOMAIN})")
        print(f"[SmartJasus] Model  : {_model_meta.get('model_name','?')}  threshold={PHISHING_THRESHOLD}")
        print(f"[SmartJasus] VT API : {'enabled' if VIRUSTOTAL_API_KEY else 'disabled'}")

    app.run(debug=True, host=APP_HOST, port=APP_PORT, threaded=True)
