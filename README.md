# SmartJasus

SmartJasus is a Flask-based phishing detection platform that combines:

- an ML classifier for email risk scoring
- a local SMTP ingestion service for end-to-end testing
- optional IMAP inbox synchronization from a real mailbox
- PostgreSQL-backed activity storage
- web dashboards and a browser extension integration path

The application can be used in fully local mode, internet-connected mode, or mixed mode.

## What The Project Does

SmartJasus analyzes email-like input and classifies it as either phishing or legitimate.

It supports these operational flows:

1. Manual check flow:
   You provide subject/body and metadata in UI or API, and the model returns a prediction with probability and confidence.
2. Local SMTP flow:
   App starts a local SMTP server, receives messages, scans them, and stores results in PostgreSQL.
3. IMAP sync flow:
   App can pull emails from your configured mailbox (for example Gmail IMAP), scan them, and store them.
4. External SMTP send flow:
   App can send an email using configured SMTP credentials.

## Current Architecture

### High-level components

- Backend API: Flask app in app.py
- ML model and training: train.py and models/phishing_detection_model.pkl
- Local SMTP + storage logic: local_mail_server.py
- Database: PostgreSQL (table emails)
- Frontend templates:
  - templates/index.html: main SOC dashboard and analytics
  - templates/mail_dashboard.html: mailbox inspector + manual check page
- Browser extension (optional): Extension/

### Runtime architecture

1. Process starts from app.py.
2. .env is loaded for mail, app, and DB configuration.
3. PostgreSQL store is initialized and schema ensured (create DB/table/indexes if missing).
4. ML model is loaded from models/phishing_detection_model.pkl.
5. Local SMTP server is started on LOCAL_SMTP_HOST:LOCAL_SMTP_PORT.
6. Flask serves dashboards and APIs on APP_HOST:APP_PORT.
7. Frontend periodically fetches activity/status and renders analytics.

## Machine Learning Component

### Model type

- Random Forest classifier persisted as models/phishing_detection_model.pkl

### Auto-model behavior

- On startup, if model file is missing, app imports train_and_save_model from train.py and trains/saves automatically.

### Inference function

- Core scoring path is predict_from_payload in app.py.
- Response returns:
  - prediction: phishing or legitimate
  - probability: phishing-class probability
  - confidence: max class probability
  - features_used: engineered, model-facing features

### Feature engineering details

Features come from extract_features_from_email and extract_domain_features in app.py.
Current extracted features include:

- email_text
- subject
- has_attachment
- links_count
- sender_domain
- urgent_keywords
- email_length
- subject_length
- link_density
- domain_age (kept for model compatibility)
- special_chars
- html_tags
- domain_length
- subdomain_count
- hyphen_count
- digit_count

Heuristics include:

- URL detection via regex when links_count is not supplied
- urgent phrase detection from a fixed phrase list
- sender domain extraction from address or URL patterns

## Mail Server Component

### Local SMTP server

- Implemented with aiosmtpd Controller in local_mail_server.py
- Started from start_local_smtp_server in app.py
- Default bind: 127.0.0.1:1025 (configurable)
- Incoming DATA is parsed, scanned by ML, and stored in emails table

Important:

- This local SMTP server does not require internet connection.
- If UI shows offline, it means local port binding/probe issue, not internet connectivity requirement.

### IMAP sync (optional internet-dependent)

- sync_inbox in app.py connects to IMAP_HOST:IMAP_PORT
- Auth uses EMAIL_ADDRESS and EMAIL_APP_PASSWORD from .env
- Pulls messages from IMAP_FOLDER, parses and stores scanned records

### External SMTP send (optional internet-dependent)

- Endpoint /api/send-email uses SMTP_HOST/SMTP_PORT credentials
- Supports STARTTLS based on SMTP_USE_TLS

## Database Component

### Engine

- PostgreSQL via psycopg2-binary

### Initialization behavior

- MailStore in local_mail_server.py auto-initializes DB on app startup.
- It attempts to:
  - create database if missing
  - create emails table if missing
  - create indexes if missing

No manual migration script is required for normal setup.

### emails table columns

- id SERIAL PRIMARY KEY
- received_at TEXT
- sender TEXT
- recipient TEXT
- subject TEXT
- body TEXT
- has_attachment INTEGER
- links_count INTEGER
- sender_domain TEXT
- source_uid TEXT UNIQUE
- source_mailbox TEXT
- prediction TEXT
- probability REAL
- confidence REAL
- created_at TIMESTAMP

## API Reference

### UI pages

- GET /
- GET /mail-dashboard
- GET /web-icon.png

### Prediction APIs

- POST /predict
- POST /detect (compat alias)

### Mail and sync APIs

- POST /api/send-local
- POST /api/send-email
- POST or GET /api/sync-inbox
- GET /api/mail-config
- GET /api/local-server-status
- GET /api/mailbox/<recipient>
- GET /api/alerts
- GET /api/activity
- POST /api/persist-records

## Frontend Behavior Notes

- Main dashboard (index.html):
  - auto-loads activity
  - renders SOC stats and threat feed
  - supports persistent card expansion until outside click
- Mail dashboard (mail_dashboard.html):
  - currently configured for manual checking workflow
  - from/to fields editable
  - check button posts to /predict
  - inbox card expansion persists across refresh until outside click

## Repository Structure

- app.py: Flask app, APIs, model inference, IMAP sync, SMTP send
- local_mail_server.py: PostgreSQL store + local SMTP handler
- train.py: model training routine
- requirements.txt: Python dependencies
- templates/: web UI pages
- Extension/: browser extension source
- models/: saved model artifacts
- datasets/ and dataset.csv: training/evaluation datasets
- DATABASE_SETUP.md: PostgreSQL setup and verification guide

## Setup Guide

### 1) Prerequisites

- Python 3.9+
- PostgreSQL 13+ (or compatible)
- pip and virtualenv

### 2) Create virtual environment

On Linux or macOS:

python3 -m venv .venv
source .venv/bin/activate

On Windows (PowerShell):

python -m venv .venv
.venv\Scripts\Activate.ps1

### 3) Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

### 4) Configure .env

Create .env in project root and set at minimum:

EMAIL_ADDRESS=your_email@example.com
EMAIL_APP_PASSWORD=your_app_password

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true

IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_FOLDER=INBOX

DB_HOST=localhost
DB_PORT=5432
DB_NAME=emailserver
DB_USER=postgres
DB_PASSWORD=your_password

LOCAL_SMTP_HOST=127.0.0.1
LOCAL_SMTP_PORT=1025

APP_HOST=0.0.0.0
APP_PORT=5000

AUTO_SYNC_ON_START=false

Notes:

- EMAIL_APP_PASSWORD can contain spaces in .env; app normalizes and strips them before login.
- Set AUTO_SYNC_ON_START=true only when internet and mailbox access are expected at startup.

### 5) Start the application

python app.py

Expected startup logs include web server and local SMTP bind target.

### 6) Open dashboards

- Main dashboard: http://127.0.0.1:5000/
- Mail dashboard: http://127.0.0.1:5000/mail-dashboard

## Verification Checklist

1. Local API health check:

curl http://127.0.0.1:5000/api/local-server-status

Expect local_server_up true.

2. Manual model check:

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"email_text":"Please verify your account now","subject":"Urgent security alert"}'

3. DB activity check:

curl "http://127.0.0.1:5000/api/activity?limit=5"

## Troubleshooting

### Local server shows offline

- Confirm app.py is currently running.
- Check if another process already uses LOCAL_SMTP_PORT.
- Verify probe endpoint directly:
  curl http://127.0.0.1:5000/api/local-server-status

If local_server_up is false while app is running, port binding likely failed or another SMTP process owns the port.

### IMAP sync fails

- Verify EMAIL_ADDRESS, EMAIL_APP_PASSWORD, IMAP_HOST, IMAP_PORT.
- Ensure mailbox/provider allows IMAP and app-password login.

### SMTP send fails

- Verify SMTP_HOST, SMTP_PORT, TLS setting, app password.
- For Gmail, use App Password and keep 2FA enabled on the account.

### DB connection errors

- Verify DB service is running.
- Check DB_USER/DB_PASSWORD and host/port.
- See DATABASE_SETUP.md for detailed DB diagnostics.

## Security Notes

- Do not commit .env to version control.
- Use app passwords, not account primary passwords.
- Restrict CORS and host binding for production deployment.
- Current setup is optimized for local testing/dev, not hardened production operation.

## License

This project is licensed under the MIT License.
