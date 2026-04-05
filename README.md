# SmartJasus

SmartJasus is an end-to-end phishing detection system that combines machine learning, email ingestion, and security dashboards in one Flask application.

It is designed for three common usage modes:

1. Offline/local analysis mode (manual checks + local SMTP flow)
2. Internet-connected mailbox mode (IMAP sync + external SMTP send)
3. Mixed SOC mode (both local SMTP and real mailbox monitoring)

## 1. Project Purpose

SmartJasus classifies email content as either phishing or legitimate and tracks scanned records in PostgreSQL for monitoring and investigation.

Core outcomes:

1. Detect suspicious email patterns with a trained gradient boosting model.
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
   - read dataset.csv
   - engineer features
   - train scikit-learn pipeline
   - persist model artifact at models/phishing_detection_model.pkl

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
   - models/phishing_detection_model.pkl

3. Training data
   - dataset.csv (primary training source in current code)

## 3. End-to-End Workflow

### 3.1 Startup workflow

When python app.py starts:

1. .env values are loaded.
2. MailStore initializes PostgreSQL (create DB/table/indexes if needed).
3. Model loading runs:
   - if model file exists, load it
   - if missing, trigger train_and_save_model() from train.py, then load
4. Local SMTP service starts at LOCAL_SMTP_HOST:LOCAL_SMTP_PORT.
5. Flask server starts at APP_HOST:APP_PORT.
6. Optional IMAP auto-sync runs if AUTO_SYNC_ON_START=true.

### 3.2 Ingestion workflows

SmartJasus ingests mail-like content through four paths:

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
   - useful for integration testing and external message delivery

### 3.3 Monitoring workflow

1. Dashboards periodically request /api/activity and status endpoints.
2. Stats and threat cards are rendered client-side.
3. Operators inspect message details and prediction confidence.

## 4. Machine Learning: Detailed Internals

This section describes exactly how ML works in the current code.

## 4.1 Training data contract

train.py expects dataset.csv with at least these columns:

1. email_text
2. subject
3. has_attachment
4. links_count
5. sender_domain
6. urgent_keywords
7. label (string class: phishing or legitimate)

Label conversion:

- phishing -> 1
- legitimate -> 0

## 4.2 Feature engineering during training

Function: extract_additional_features(df) in train.py

Generated engineered features:

1. email_length
   len(email_text)

2. subject_length
   len(subject)

3. link_density
   links_count / (email_length + 1)

4. domain_age
   placeholder proxy: hash(sender_domain) % 30
   Note: this is not WHOIS age; it is a deterministic placeholder feature.

5. special_chars
   count of punctuation/special symbols in email_text

6. html_tags
   count of <...> style tags in lowercase email_text

## 4.3 Preprocessing pipeline design

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
   - n_features = 100
   - alternate_sign = False

Numeric channel:

4. numeric features -> StandardScaler
   - has_attachment
   - links_count
   - urgent_keywords
   - email_length
   - subject_length
   - link_density
   - domain_age
   - special_chars
   - html_tags

Combined representation:

- ColumnTransformer concatenates all transformed channels into one feature space.

## 4.4 Classifier internals

Current classifier: GradientBoostingClassifier

Hyperparameters in train.py:

1. n_estimators = 150
2. learning_rate = 0.1
3. max_depth = 5
4. random_state = 42
5. subsample = 0.8
6. max_features = sqrt

Conceptually, gradient boosting builds trees sequentially to minimize classification loss by fitting each new learner to residual errors of the ensemble.

If f_m(x) is the ensemble after m stages:

f_m(x) = f_{m-1}(x) + eta * h_m(x)

where:

- eta is learning_rate
- h_m(x) is the new tree fitted to gradient/residual signal

## 4.5 Train/validation flow

1. Split: train_test_split with
   - test_size = 0.2
   - random_state = 42
   - stratify = y

2. Fit pipeline on train split.
3. Evaluate on test split using:
   - accuracy_score
   - f1_score
   - classification_report
4. Save model with joblib.dump(..., compress=3).

## 4.6 Inference internals in app.py

Primary function: predict_from_payload(data)

Pipeline:

1. extract_features_from_email(...) fills missing metadata and computes engineered fields.
2. Build single-row pandas DataFrame.
3. Call model.predict and model.predict_proba.
4. Return:
   - prediction: phishing/legitimate
   - probability: probability of phishing class
   - confidence: max(probabilities)
   - features_used: non-text engineered metadata

## 4.7 Runtime feature extraction behavior

extract_features_from_email in app.py has smart fallbacks:

1. links_count
   - if missing, regex-extract URLs from email_text

2. sender_domain
   - if missing, infer from email address or URL-like patterns

3. urgent_keywords
   - if missing, detect phrase hits from a predefined urgency phrase list

4. email_length/subject_length/link_density/special_chars/html_tags
   - computed every request

Domain helpers:

- extract_domain_features(domain) uses tldextract and adds:
  - domain_length
  - subdomain_count
  - hyphen_count
  - digit_count

Note:

- The saved model was trained with the features defined in train.py.
- app.py also computes some additional domain-derived fields for diagnostics and compatibility.

## 4.8 Prediction semantics

Returned values mean:

1. prediction
   - final class label from estimator

2. probability
   - model-estimated phishing likelihood

3. confidence
   - max class probability (model certainty for chosen class)

Interpretation guideline:

- high probability + phishing -> high-risk candidate
- high confidence + legitimate -> likely benign
- borderline probabilities should be reviewed with context and sender validation

## 4.9 ML limitations and caveats

1. domain_age is currently a placeholder hash feature, not real registration age.
2. HashingVectorizer is non-invertible; token-level interpretability is limited.
3. Model quality depends on dataset representativeness and label quality.
4. Concept drift is expected in phishing campaigns; periodic retraining is required.

## 4.10 Recommended ML improvement roadmap

1. Replace placeholder domain_age with real WHOIS/domain intelligence feature.
2. Add calibration (Platt/Isotonic) for better probability reliability.
3. Add precision/recall and confusion matrix reporting in training logs.
4. Version model artifacts with timestamp and dataset hash metadata.
5. Add drift monitoring and scheduled retraining workflow.

## 5. Mail Server and Messaging Internals

### 5.1 Local SMTP server

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

### 5.2 External SMTP endpoint

Endpoint: /api/send-email

Behavior:

1. Validate EMAIL_ADDRESS and EMAIL_APP_PASSWORD
2. Build EmailMessage
3. Connect SMTP_HOST:SMTP_PORT
4. EHLO
5. STARTTLS if SMTP_USE_TLS=true
6. EHLO again
7. Login and send

### 5.3 IMAP sync internals

Function: sync_inbox(limit=20)

1. Connect IMAP4_SSL(IMAP_HOST, IMAP_PORT)
2. Login with EMAIL_ADDRESS and EMAIL_APP_PASSWORD
3. Select IMAP_FOLDER readonly
4. Search UID ALL, fetch latest limit UIDs
5. Fetch RFC822 + INTERNALDATE
6. Parse each message and score
7. Store with source_uid=imap:<uid>

## 6. Database Layer Internals

Database bootstrap is automatic in MailStore._init_db:

1. _create_database()
2. _create_tables()

### 6.1 Schema

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

### 6.2 Indexes

1. idx_emails_recipient(recipient, received_at desc)
2. idx_emails_source_uid(source_uid) where source_uid is not null
3. idx_emails_prediction(prediction, received_at desc)
4. idx_emails_received_at(received_at desc)

### 6.3 Dedup behavior

save_email handles source_uid unique collisions:

1. catches IntegrityError
2. updates received_at for existing source_uid
3. avoids duplicate row insertion for same source_uid

## 7. API Reference

### 7.1 UI routes

1. GET /
2. GET /mail-dashboard
3. GET /web-icon.png

### 7.2 Prediction routes

1. POST /predict
2. OPTIONS /predict
3. POST /detect (compat alias)

### 7.3 Mail and data routes

1. POST /api/send-local
2. POST /api/send-email
3. POST or GET /api/sync-inbox
4. GET /api/mail-config
5. GET /api/local-server-status
6. GET /api/mailbox/<recipient>
7. GET /api/alerts
8. GET /api/activity
9. POST /api/persist-records

## 8. Repository Structure

1. app.py
   Flask app, endpoint layer, inference orchestrator, IMAP sync

2. train.py
   ML training pipeline, evaluation, artifact export

3. local_mail_server.py
   PostgreSQL store, schema bootstrap, SMTP ingest logic

4. templates/
   UI pages for SOC dashboard and manual check console

5. Extension/
   Optional Chrome extension files

6. models/
   Trained model artifact(s)

7. dataset.csv
   Current training dataset source in code

8. DATABASE_SETUP.md
   Detailed PostgreSQL setup and verification

## 9. Setup and Run

### 9.1 Prerequisites

1. Python 3.9+
2. PostgreSQL 13+
3. pip and virtual environment support

### 9.2 Environment setup

Linux/macOS:

python3 -m venv .venv
source .venv/bin/activate

Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1

Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

### 9.3 .env example

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

1. EMAIL_APP_PASSWORD is normalized by stripping spaces before auth.
2. AUTO_SYNC_ON_START=true enables startup mailbox sync.

### 9.4 Run

python app.py

Open:

1. http://127.0.0.1:5000/
2. http://127.0.0.1:5000/mail-dashboard

## 10. Verification Checklist

1. Verify local SMTP status:

curl http://127.0.0.1:5000/api/local-server-status

Expected: local_server_up should be true.

2. Verify prediction endpoint:

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"email_text":"Please verify your account now","subject":"Urgent security alert"}'

3. Verify activity retrieval:

curl "http://127.0.0.1:5000/api/activity?limit=5"

4. Optional: trigger IMAP sync:

curl -X POST "http://127.0.0.1:5000/api/sync-inbox?limit=5"

## 11. Troubleshooting

### 11.1 Local SMTP shows offline

1. Ensure app.py process is running.
2. Check LOCAL_SMTP_PORT conflicts.
3. Query /api/local-server-status directly.
4. If port is occupied by another process, free or change port.

### 11.2 Model load or training errors

1. Confirm dataset.csv exists and has required columns.
2. Check scikit-learn/pandas installation.
3. Remove stale model artifact and restart to force retrain when needed.

### 11.3 IMAP auth failures

1. Verify EMAIL_ADDRESS and EMAIL_APP_PASSWORD.
2. Ensure provider allows IMAP and app password usage.
3. Verify IMAP host/port values.

### 11.4 SMTP send failures

1. Verify SMTP_HOST, SMTP_PORT, TLS setting.
2. For Gmail, use App Password with account 2FA enabled.

### 11.5 PostgreSQL connection issues

1. Verify DB service is running.
2. Validate DB credentials in .env.
3. Check DB user permissions for DB/table creation.
4. See DATABASE_SETUP.md for full diagnostics.

## 12. Security and Operational Notes

1. Never commit .env with real credentials.
2. This project is configured for development/local SOC simulation.
3. Restrict CORS, host binding, and credential handling before production deployment.
4. Add proper secret management and audit logging for production use.

## 13. License

This project is licensed under the MIT License.
