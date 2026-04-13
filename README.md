# CatchFish

CatchFish is an end-to-end phishing detection system that combines machine learning, email ingestion, and security dashboards in one Flask application.

It is trained on the real-world CSV datasets currently present in `dataset/` (Enron, Ling, Nazario, Nigerian_Fraud, SpamAssasin).

Usage modes:

1. Offline/local analysis mode (manual checks + local SMTP flow)
2. Internet-connected mailbox mode (IMAP sync + external SMTP send)
3. Mixed SOC mode (both local SMTP and real mailbox monitoring)

## 1. Project Purpose

CatchFish classifies email content as either phishing or legitimate and tracks scanned records in PostgreSQL for monitoring and investigation.

Core outcomes:

1. Detect suspicious email patterns with a model trained on real-world phishing and legitimate email data.
2. Capture and persist email activity from multiple ingestion channels.
3. Provide operator-facing dashboards with live feed, alerts, and trends.

## 2. System Architecture

### 2.1 Components

1. API and orchestration layer
   File: app.py
   Responsibilities:
   - load environment and startup services
   - expose prediction and mail APIs
   - run IMAP sync and SMTP send endpoints
   - serve dashboards

2. ML training pipeline
   File: train.py
   Responsibilities:
   - read and merge all discovered real-world dataset CSVs from dataset/
   - clean and derive required features
   - engineer additional features
   - train a scikit-learn Logistic Regression pipeline
   - persist the model artifact at models/phishing_detection.pkl

3. Local SMTP + persistence layer
   File: local_mail_server.py
   Responsibilities:
   - run local SMTP server via aiosmtpd
   - parse incoming messages
   - call ML prediction callback
   - write normalized results to PostgreSQL

4. UI layer
   Files:
   - templates/index.html
   - templates/mail_dashboard.html
   Responsibilities:
   - monitoring and activity visualization
   - manual phishing check
   - local server status and feed exploration

5. Browser extension (optional integration)
   Directory: Extension/

### 2.2 Data stores and artifacts

1. PostgreSQL database
   - table: emails
   - stores metadata, body, prediction scores, source identifiers

2. Model artifact
   - models/phishing_detection.pkl

3. Real-world training data
   - dataset/Enron.csv
   - dataset/Ling.csv
   - dataset/Nazario.csv
   - dataset/Nigerian_Fraud.csv
   - dataset/SpamAssasin.csv

4. Reference data (dataset/)
   - legitimate_domains.json
   - suspicious_tlds.json
   - url_shorteners.json

## 3. End-to-End Workflow

### 3.1 Startup workflow

When python app.py starts:

1. .env values are loaded.
2. MailStore initializes PostgreSQL (create DB/table/indexes if needed).
3. Model loading runs:
   - if models/phishing_detection.pkl exists, load it
   - if missing, trigger training from train.py, then load
4. Local SMTP service starts at LOCAL_SMTP_HOST:LOCAL_SMTP_PORT.
5. Flask server starts at APP_HOST:APP_PORT.
6. Optional IMAP auto-sync runs if AUTO_SYNC_ON_START=true.

### 3.2 Ingestion workflows

CatchFish ingests mail-like content through four paths:

1. Manual prediction
   - UI or API POST to /predict
   - extract features -> model -> JSON result

2. Local SMTP ingest
   - mail arrives at local SMTP server
   - LocalMailHandler parses message
   - prediction callback runs
   - record is stored in PostgreSQL

3. IMAP sync ingest
   - /api/sync-inbox reads configured mailbox via IMAP
   - parsed messages are scored
   - new records stored with source_uid and source_mailbox

4. External SMTP send
   - /api/send-email sends outbound email using provider credentials

### 3.3 Monitoring workflow

1. Dashboards periodically request /api/activity and status endpoints.
2. Stats and threat cards are rendered client-side.
3. Operators inspect message details and prediction confidence.

## 4. Machine Learning: Detailed Internals

### 4.1 Real-world dataset pipeline

train.py loads and merges all discovered real-world CSV files from dataset/:

| File               | Source                      |
|--------------------|-----------------------------|
| Enron.csv          | Real corporate email corpus |
| Ling.csv           | Legitimate email corpus     |
| Nazario.csv        | Real phishing samples       |
| Nigerian_Fraud.csv | Advance-fee fraud emails    |
| SpamAssasin.csv    | SpamAssasin benchmark       |

### 4.2 Feature derivation from raw dataset data

Raw dataset columns: sender, body, subject, label (0/1)

Derived required inputs:

1. email_text      — body column after HTML stripping and whitespace normalisation
2. subject         — subject column (cleaned)
3. sender_domain   — domain extracted from sender email address via regex
4. has_attachment  — defaulted to 0 (not present in dataset files)
5. links_count     — count of https?:// URLs in email_text via regex
6. urgent_keywords — count of 16 urgency keyword hits in subject + body
7. label           — already 0/1 integer in all files (1=phishing, 0=legitimate)

### 4.3 Feature engineering during training

Function: extract_additional_features(df) in train.py

Generated engineered features:

1. email_length        — len(email_text)
2. subject_length      — len(subject)
3. link_density        — links_count / (email_length + 1)
4. domain_age          — hash-based proxy: 30 if trusted domain, else 1–8
5. special_chars       — count of punctuation/special symbols in email_text
6. html_tags           — count of <...> style tags in lowercase email_text
7. legitimate_domain   — 1 if sender_domain in legitimate_domains.json
8. suspicious_tld      — 1 if sender_domain ends with a suspicious TLD
9. domain_length       — len(sender_domain)
10. domain_has_digits  — 1 if any digit in sender_domain
11. domain_has_hyphen  — 1 if hyphen in sender_domain
12. ip_url_count       — count of IP-address-based URLs
13. shortener_url_count— count of known URL shortener domains
14. https_url_count    — count of https:// links
15. http_url_count     — count of http:// links
16. http_ratio         — http_url_count / (links_count + 1)

### 4.4 Preprocessing pipeline design

The model uses a scikit-learn Pipeline with a ColumnTransformer.

Text channels:

1. email_text -> HashingVectorizer
   - n_features = 2^16
   - alternate_sign = False
   - stop_words = english
   - ngram_range = (1, 2)

2. subject -> HashingVectorizer (same config)

Categorical channel:

3. sender_domain -> HashingVectorizer
   - n_features = 512
   - alternate_sign = False

Numeric channel:

4. 19 numeric features -> StandardScaler

### 4.5 Model configuration

CatchFish uses a single classifier:

1. Logistic Regression — max_iter=1000, C=1.0, class_weight=balanced, solver=lbfgs

The trained pipeline is saved to models/phishing_detection.pkl.

### 4.6 Train/validation flow

1. Split: stratified 70 / 15 / 15 train/validation/test
2. Fit Logistic Regression pipeline on the train split.
3. Evaluate with threshold=0.60:
   - AUC, Accuracy, Precision, Recall, F1, FP, FN
4. Save the trained pipeline with joblib.dump(..., compress=3).

### 4.7 Inference internals in app.py

Primary function: predict_from_payload(data)

Pipeline:

1. extract_email_features(...) fills missing metadata and computes engineered fields.
2. Build single-row pandas DataFrame.
3. Call model.predict_proba.
4. Apply rule adjustments (trusted domain, IP URLs, suspicious TLD, shorteners).
5. Return prediction, probability, confidence, features_used, url_analysis.

### 4.8 Prediction semantics

- prediction   — phishing or legitimate
- probability  — model-estimated phishing likelihood (after rule adjustments)
- confidence   — certainty for chosen class
- threshold    — 0.60 (configurable via PHISHING_THRESHOLD in .env)

### 4.9 ML limitations and caveats

1. domain_age is a hash-based placeholder, not real WHOIS registration age.
2. has_attachment is always 0 in training data (not present in dataset files).
3. HashingVectorizer is non-invertible; token-level interpretability is limited.
4. Concept drift is expected in phishing campaigns; periodic retraining recommended.

## 5. Retraining

To retrain the model locally (e.g. after sklearn upgrade or adding new data):

python train.py

This will:
1. Load and merge all discovered CSVs from dataset/
2. Train the Logistic Regression pipeline
3. Save it to models/phishing_detection.pkl

To retrain on Google Colab (recommended for slow machines):
- Upload dataset/ folder to Google Drive
- Open CatchFish.ipynb in Colab
- Set DATASET_PATH and run all cells
- Download the generated .pkl and place in models/

## 6. Mail Server and Messaging Internals

### 6.1 Local SMTP server

Implementation:

- LocalSMTPServer wraps aiosmtpd Controller
- LocalMailHandler.handle_DATA parses RFC822 content

Flow:

1. Parse From/Subject/body
2. Detect attachment presence
3. Infer sender domain
4. Parse Date header for received_at fallback
5. Predict phishing probability
6. Save one record per recipient in envelope.rcpt_tos

### 6.2 External SMTP endpoint

Endpoint: /api/send-email

Behavior:

1. Validate EMAIL_ADDRESS and EMAIL_APP_PASSWORD
2. Build EmailMessage
3. Connect SMTP_HOST:SMTP_PORT with STARTTLS
4. Login and send

### 6.3 IMAP sync internals

Function: sync_inbox(limit=20)

1. Connect IMAP4_SSL(IMAP_HOST, IMAP_PORT)
2. Login with EMAIL_ADDRESS and EMAIL_APP_PASSWORD
3. Select IMAP_FOLDER readonly
4. Search UID ALL, fetch latest limit UIDs
5. Fetch RFC822 + INTERNALDATE
6. Parse each message and score
7. Store with source_uid=imap:<uid>

## 7. Database Layer Internals

Database bootstrap is automatic in MailStore._init_db.

### 7.1 Schema

Table: emails

1. id SERIAL PRIMARY KEY
2. received_at TEXT NOT NULL
3. sender TEXT NOT NULL
4. recipient TEXT NOT NULL
5. subject TEXT
6. body TEXT
7. has_attachment INTEGER NOT NULL DEFAULT 0
8. links_count INTEGER NOT NULL DEFAULT 0
9. sender_domain TEXT
10. source_uid TEXT UNIQUE
11. source_mailbox TEXT
12. prediction TEXT NOT NULL
13. probability REAL NOT NULL
14. confidence REAL NOT NULL
15. created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

### 7.2 Indexes

1. idx_emails_recipient(recipient, received_at desc)
2. idx_emails_source_uid(source_uid) where source_uid is not null
3. idx_emails_prediction(prediction, received_at desc)
4. idx_emails_received_at(received_at desc)

## 8. API Reference

### 8.1 UI routes

1. GET /
2. GET /alerts
3. GET /dataset
4. GET /feature-extraction
5. GET /post-ml
6. GET /model
7. GET /domain-intelligence
8. GET /virustotal-file-scanner
9. GET /mail-dashboard
10. GET /web-icon.png

### 8.2 Prediction routes

1. POST /predict
2. OPTIONS /predict
3. POST /detect (compat alias)

### 8.3 Mail and data routes

1. POST /api/send-local
2. POST /api/send-email
3. POST or GET /api/sync-inbox
4. GET /api/mail-config
5. GET /api/local-server-status
6. GET /api/mailbox/<recipient>
7. GET /api/alerts
8. GET /api/activity
9. GET /api/logs
10. GET /api/stats
11. GET /api/model-info
12. POST /api/domain-reputation
13. POST /api/scan-file
14. POST /api/persist-records
15. POST /api/users/add
16. GET /api/users

## 9. Repository Structure

```
dataset/
  Enron.csv                     Real corporate email corpus
  Ling.csv                      Legitimate email corpus
  Nazario.csv                   Real phishing samples
  Nigerian_Fraud.csv            Advance-fee fraud emails
  SpamAssasin.csv               SpamAssasin benchmark
  legitimate_domains.json       Trusted domain list
  suspicious_tlds.json          High-risk TLD list
  url_shorteners.json           Known URL shortener domains

models/
  phishing_detection.pkl   Trained model artifact

templates/
  index.html                    SOC monitoring dashboard
   alerts.html                   Alert feed and triage UI
   dataset.html                  Dataset analytics page
   domain_intelligence.html      Domain reputation page
   feature_extraction.html       Feature extraction page
   mail_dashboard.html           Mail compose and inbox UI
   model.html                    Model details page
   post_ml.html                  Post-ML rules page
   virustotal_file_scanner.html  File scanner page

Extension/
  manifest.json                 Chrome extension manifest
  popup.html / popup.js         Extension popup
  content.js / background.js    Extension scripts

app.py                          Flask app, endpoints, inference, IMAP sync
train.py                        ML training pipeline on real-world data
local_mail_server.py            PostgreSQL store, SMTP ingest logic
CatchFish.ipynb                 Notebook workflow
requirements.txt                Python dependencies
DATABASE_SETUP.md               PostgreSQL setup guide
.env                            Environment config (not committed)
```

## 10. Setup and Run

### 10.1 Prerequisites

1. Python 3.9+
2. PostgreSQL 13+
3. pip and virtual environment support

### 10.2 Environment setup

Linux/macOS:

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 10.3 .env example

```
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

VIRUSTOTAL_API_KEY=your_virustotal_api_key

PHISHING_THRESHOLD=0.60
AUTO_SYNC_ON_START=false
```

### 10.4 Run

```
python app.py
```

Open:

1. http://127.0.0.1:5000/
2. http://127.0.0.1:5000/mail-dashboard

## 11. Verification Checklist

1. Verify local SMTP status:

```
curl http://127.0.0.1:5000/api/local-server-status
```

Expected: local_server_up should be true.

2. Verify prediction endpoint:

```
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text":"Please verify your account now","subject":"Urgent security alert"}'
```

3. Verify model info:

```
curl http://127.0.0.1:5000/api/model-info
```

4. Optional: trigger IMAP sync:

```
curl -X POST "http://127.0.0.1:5000/api/sync-inbox?limit=5"
```

## 12. Troubleshooting

### 12.1 Local SMTP shows offline

1. Ensure app.py process is running.
2. Check LOCAL_SMTP_PORT conflicts.
3. Query /api/local-server-status directly.

### 12.2 Model file missing on startup

If models/phishing_detection.pkl is missing:

1. app.py will automatically call train.py to rebuild it.
2. Or retrain manually: python train.py
3. Or use CatchFish.ipynb on Google Colab and copy the .pkl here.

### 12.3 IMAP auth failures

1. Verify EMAIL_ADDRESS and EMAIL_APP_PASSWORD.
2. Ensure provider allows IMAP and app password usage.

### 12.4 SMTP send failures

1. Verify SMTP_HOST, SMTP_PORT, TLS setting.
2. For Gmail, use App Password with account 2FA enabled.

### 12.5 PostgreSQL connection issues

1. Verify DB service is running.
2. Validate DB credentials in .env.
3. See DATABASE_SETUP.md for full diagnostics.

## 13. Security and Operational Notes

1. Never commit .env with real credentials.
2. This project is configured for development/local SOC simulation.
3. Restrict CORS, host binding, and credential handling before production deployment.

## 14. License

This project is licensed under the MIT License.
