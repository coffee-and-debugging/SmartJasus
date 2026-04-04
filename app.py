from flask import Flask, request, jsonify, render_template, g
from flask import send_from_directory
import joblib
from flask_cors import CORS
import pandas as pd
import os
import re
import warnings
import imaplib
import smtplib
import socket
import logging
from email.message import EmailMessage
from email import policy
from email.parser import BytesParser
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
import tldextract
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone

from local_mail_server import MailStore, LocalMailHandler, LocalSMTPServer

load_dotenv()

warnings.filterwarnings('ignore')
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

LOCAL_SMTP_HOST = os.getenv('LOCAL_SMTP_HOST', '127.0.0.1')
LOCAL_SMTP_PORT = int(os.getenv('LOCAL_SMTP_PORT', '1025'))
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', '').strip()
EMAIL_APP_PASSWORD = ''.join(os.getenv('EMAIL_APP_PASSWORD', '').split())
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com').strip()
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
IMAP_HOST = os.getenv('IMAP_HOST', 'imap.gmail.com').strip()
IMAP_PORT = int(os.getenv('IMAP_PORT', '993'))
IMAP_FOLDER = os.getenv('IMAP_FOLDER', 'INBOX').strip()
AUTO_SYNC_ON_START = os.getenv('AUTO_SYNC_ON_START', 'false').lower() == 'true'
APP_HOST = os.getenv('APP_HOST', '0.0.0.0').strip()
APP_PORT = int(os.getenv('APP_PORT', '5000'))
REQUEST_LOG_ENABLED = os.getenv('REQUEST_LOG_ENABLED', 'true').lower() == 'true'
REQUEST_LOG_EXCLUDE = {
    '/api/local-server-status',
    '/favicon.ico',
}

# PostgreSQL Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '5432'))
DB_NAME = os.getenv('DB_NAME', 'emailserver')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

mail_store = MailStore(
    db_host=DB_HOST,
    db_port=DB_PORT,
    db_name=DB_NAME,
    db_user=DB_USER,
    db_password=DB_PASSWORD
)
local_smtp_server = None

print("[SmartJasus] Server starting, It may take a while...")

model = None
def load_model():
    global model
    if not os.path.exists('models/phishing_detection_model.pkl'):
        from train import train_and_save_model
        train_and_save_model()
    model = joblib.load('models/phishing_detection_model.pkl')

load_model()

def extract_domain_features(domain):
    """Extract features from domain name"""
    if not domain:
        return {
            'domain_length': 0,
            'subdomain_count': 0,
            'hyphen_count': 0,
            'digit_count': 0
        }
    
    extracted = tldextract.extract(domain)
    main_domain = f"{extracted.domain}.{extracted.suffix}"
    
    return {
        'domain_length': len(main_domain),
        'subdomain_count': len(extracted.subdomain.split('.')),
        'hyphen_count': main_domain.count('-'),
        'digit_count': sum(c.isdigit() for c in main_domain)
    }

def extract_features_from_email(email_text, subject="", has_attachment=None, links_count=None, sender_domain=None, urgent_keywords=None):
    """
    Extract features from available email data with smart defaults for missing values
    """
   
    features = {
        'email_text': email_text if email_text else "",
        'subject': subject if subject else "",
        'has_attachment': 0,
        'links_count': 0,
        'sender_domain': "",
        'urgent_keywords': 0,
        
        'email_length': 0,
        'subject_length': 0,
        'link_density': 0,
        'domain_age': 0,
        'special_chars': 0,
        'html_tags': 0
    }
    
    
    if has_attachment is not None:
        features['has_attachment'] = int(has_attachment)
    
    
    if links_count is None and email_text:
        
        links = re.findall(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', email_text.lower())
        features['links_count'] = len(links)
    elif links_count is not None:
        features['links_count'] = int(links_count)
    
    
    if sender_domain is None and email_text:
       
        domain_match = re.search(
            r'[\w\.-]+@([\w\.-]+\.\w{2,})|https?://([\w\.-]+\.\w{2,})', 
            email_text.lower())
        if domain_match:
            features['sender_domain'] = domain_match.group(1) or domain_match.group(2)
    
    
    if urgent_keywords is None and email_text:
        urgent_phrases = ['urgent', 'immediate', 'action required', 'verify now', 
                         'security alert', 'account suspended', 'password expired',
                         'click here', 'limited time', 'offer expires']
        features['urgent_keywords'] = int(any(phrase in email_text.lower() for phrase in urgent_phrases))
    elif urgent_keywords is not None:
        features['urgent_keywords'] = int(urgent_keywords)
    
   
    features['email_length'] = len(features['email_text'])
    features['subject_length'] = len(features['subject'])
    features['link_density'] = features['links_count'] / (features['email_length'] + 1)
    
    
    domain_features = extract_domain_features(features['sender_domain'])
    features.update(domain_features)
    
   
    features['special_chars'] = len(re.findall(r'[!$%^&*()_+|~=`{}\[\]:";\'<>?,./]', features['email_text']))
    
   
    features['html_tags'] = len(re.findall(r'<[^>]+>', features['email_text'].lower()))
    
    return features


def predict_from_payload(data):
    """Run prediction using existing model and feature extraction logic."""
    features = extract_features_from_email(
        email_text=data.get('email_text', ''),
        subject=data.get('subject', ''),
        has_attachment=data.get('has_attachment'),
        links_count=data.get('links_count'),
        sender_domain=data.get('sender_domain'),
        urgent_keywords=data.get('urgent_keywords')
    )

    input_data = pd.DataFrame([features])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    return {
        'prediction': 'phishing' if prediction[0] == 1 else 'legitimate',
        'probability': float(probability[0][1]),
        'confidence': float(max(probability[0])),
        'features_used': {
            k: v for k, v in features.items()
            if k not in ['email_text', 'subject']
        }
    }


def decode_mime_words(value):
    if not value:
        return ""
    decoded_parts = []
    for fragment, encoding in decode_header(value):
        if isinstance(fragment, bytes):
            decoded_parts.append(fragment.decode(encoding or 'utf-8', errors='replace'))
        else:
            decoded_parts.append(fragment)
    return ''.join(decoded_parts)


def parse_imap_internal_date(meta):
    if not meta:
        return None
    if isinstance(meta, bytes):
        meta = meta.decode(errors='ignore')
    match = re.search(r'INTERNALDATE "([^"]+)"', str(meta))
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), '%d-%b-%Y %H:%M:%S %z')
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def parse_email_message(message, received_at_override=None):
    sender = decode_mime_words(message.get('From', 'unknown@local'))
    recipient = decode_mime_words(message.get('To', EMAIL_ADDRESS or 'unknown@local'))
    subject = decode_mime_words(message.get('Subject', ''))

    body = ""
    has_attachment = 0
    if message.is_multipart():
        text_parts = []
        for part in message.walk():
            disposition = part.get_content_disposition()
            if disposition == 'attachment':
                has_attachment = 1
                continue
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                text_parts.append(part.get_content() or '')
        body = "\n".join([p for p in text_parts if p])
    else:
        body = message.get_content() or ""

    email_addr = parseaddr(sender)[1]
    sender_domain = email_addr.split('@', 1)[1].lower() if '@' in email_addr else ''
    
    # Prefer an explicit received time from IMAP/internal sources when available.
    received_at = received_at_override
    # Fall back to email Date header if needed.
    date_header = message.get('Date')
    if not received_at and date_header:
        try:
            dt = parsedate_to_datetime(date_header)
            received_at = dt.astimezone(timezone.utc).isoformat()
        except:
            pass
    
    # Fallback to current time if no Date header
    if not received_at:
        from datetime import datetime
        received_at = datetime.now(timezone.utc).isoformat()
    
    return {
        'sender': sender,
        'recipient': recipient,
        'subject': subject,
        'body': body,
        'has_attachment': has_attachment,
        'sender_domain': sender_domain,
        'received_at': received_at
    }


def scan_and_store_email(payload, source_uid=None, source_mailbox=None):
    prediction = predict_from_payload(
        {
            'email_text': payload['body'],
            'subject': payload['subject'],
            'has_attachment': payload['has_attachment'],
            'sender_domain': payload['sender_domain'],
        }
    )
    return mail_store.save_email(
        sender=payload['sender'],
        recipient=payload['recipient'] or (EMAIL_ADDRESS or 'unknown@local'),
        subject=payload['subject'],
        body=payload['body'],
        has_attachment=payload['has_attachment'],
        links_count=prediction['features_used'].get('links_count', 0),
        sender_domain=payload['sender_domain'],
        received_at=payload.get('received_at'),
        source_uid=source_uid,
        source_mailbox=source_mailbox,
        prediction=prediction['prediction'],
        probability=prediction['probability'],
        confidence=prediction['confidence'],
    )


def sync_inbox(limit=20):
    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        raise ValueError('EMAIL_ADDRESS and EMAIL_APP_PASSWORD must be set in .env')

    imported = 0
    with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
        imap.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        imap.select(IMAP_FOLDER, readonly=True)

        status, data = imap.uid('search', None, 'ALL')
        if status != 'OK' or not data or not data[0]:
            return imported

        uids = data[0].split()[-max(1, limit):]
        for uid in uids:
            status, msg_data = imap.uid('fetch', uid, '(RFC822 INTERNALDATE)')
            if status != 'OK' or not msg_data:
                continue

            raw_email = None
            received_at = None
            for chunk in msg_data:
                if isinstance(chunk, tuple) and len(chunk) >= 2:
                    raw_email = chunk[1]
                    received_at = parse_imap_internal_date(chunk[0])
                    break
            if not raw_email:
                continue

            message = BytesParser(policy=policy.default).parsebytes(raw_email)
            parsed = parse_email_message(message, received_at_override=received_at)
            row_id = scan_and_store_email(
                parsed,
                source_uid=f'imap:{uid.decode()}',
                source_mailbox=IMAP_FOLDER
            )
            if row_id:
                imported += 1

    return imported


def start_local_smtp_server():
    """Start local SMTP service for end-to-end local mail flow."""
    global local_smtp_server
    if local_smtp_server is not None:
        return

    handler = LocalMailHandler(mail_store, predict_from_payload)
    local_smtp_server = LocalSMTPServer(
        handler,
        host=LOCAL_SMTP_HOST,
        port=LOCAL_SMTP_PORT
    )
    try:
        local_smtp_server.start()
    except OSError as exc:
        # Avoid hard failure when another local SMTP instance is already bound.
        if getattr(exc, 'errno', None) == 98:
            local_smtp_server = None
        else:
            raise

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/mail-dashboard')
def mail_dashboard():
    return render_template('mail_dashboard.html', default_from_address=EMAIL_ADDRESS)


@app.route('/web-icon.png')
def web_icon():
    return send_from_directory('templates', 'icon.png')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    try:
       
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'preflight'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response

        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data received'}), 400

        response = predict_from_payload(data)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect', methods=['POST'])
def detect_alias():
    """Backward-compatible alias used by browser extension."""
    data = request.get_json() or {}
    normalized = {
        'email_text': data.get('email_text') or data.get('text', ''),
        'subject': data.get('subject', ''),
        'has_attachment': data.get('has_attachment'),
        'links_count': data.get('links_count'),
        'sender_domain': data.get('sender_domain'),
        'urgent_keywords': data.get('urgent_keywords')
    }
    result = predict_from_payload(normalized)
    return jsonify({'result': result['prediction'], **result})


@app.route('/api/send-local', methods=['POST'])
def send_local_email():
    """Send an email through Local SMTP Server for local user-to-user testing."""
    try:
        data = request.get_json() or {}
        from_address = (EMAIL_ADDRESS or '').strip()
        to_address = (data.get('to_address') or '').strip()
        subject = data.get('subject', '')
        body = data.get('body', '')

        if not from_address:
            return jsonify({'error': 'EMAIL_ADDRESS must be set in .env'}), 400

        if not to_address:
            return jsonify({'error': 'to_address is required'}), 400

        msg = EmailMessage()
        msg['From'] = from_address
        msg['To'] = to_address
        msg['Subject'] = subject
        msg.set_content(body)

        with smtplib.SMTP(LOCAL_SMTP_HOST, LOCAL_SMTP_PORT, timeout=10) as smtp:
            smtp.send_message(msg)

        return jsonify({'status': 'sent'})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/send-email', methods=['POST'])
def send_email():
    """Send email through configured SMTP server using .env credentials."""
    try:
        if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
            return jsonify({'error': 'EMAIL_ADDRESS and EMAIL_APP_PASSWORD must be set in .env'}), 400

        data = request.get_json() or {}
        to_address = (data.get('to_address') or '').strip()
        subject = data.get('subject', '')
        body = data.get('body', '')

        if not to_address:
            return jsonify({'error': 'to_address is required'}), 400

        msg = EmailMessage()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_address
        msg['Subject'] = subject
        msg.set_content(body)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            smtp.ehlo()
            if SMTP_USE_TLS:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        return jsonify({'status': 'sent', 'from': EMAIL_ADDRESS, 'to': to_address})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/sync-inbox', methods=['POST', 'GET'])
def sync_inbox_endpoint():
    try:
        limit = int(request.args.get('limit', 20))
        limit = max(1, min(limit, 1000))
        imported = sync_inbox(limit=limit)
        return jsonify({'status': 'ok', 'imported': imported, 'folder': IMAP_FOLDER})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/mail-config', methods=['GET'])
def mail_config_status():
    return jsonify(
        {
            'email_address_set': bool(EMAIL_ADDRESS),
            'app_password_set': bool(EMAIL_APP_PASSWORD),
            'smtp_host': SMTP_HOST,
            'smtp_port': SMTP_PORT,
            'imap_host': IMAP_HOST,
            'imap_port': IMAP_PORT,
            'imap_folder': IMAP_FOLDER,
            'auto_sync_on_start': AUTO_SYNC_ON_START,
        }
    )


@app.route('/api/local-server-status', methods=['GET'])
def local_server_status():
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.settimeout(0.5)
    try:
        result = socket_client.connect_ex((LOCAL_SMTP_HOST, LOCAL_SMTP_PORT))
        return jsonify(
            {
                'local_server_up': result == 0,
                'host': LOCAL_SMTP_HOST,
                'port': LOCAL_SMTP_PORT,
            }
        )
    finally:
        socket_client.close()


@app.route('/api/mailbox/<path:recipient>', methods=['GET'])
def mailbox(recipient):
    limit = int(request.args.get('limit', 50))
    limit = max(1, min(limit, 5000))
    items = mail_store.get_mailbox(recipient, limit=limit)
    return jsonify({'recipient': recipient, 'items': items})


@app.route('/api/alerts', methods=['GET'])
def alerts():
    limit = int(request.args.get('limit', 100))
    limit = max(1, min(limit, 500))
    items = mail_store.get_alerts(limit=limit)
    return jsonify({'items': items})


@app.route('/api/activity', methods=['GET'])
def activity():
    limit = int(request.args.get('limit', 100))
    limit = max(1, min(limit, 5000))
    items = mail_store.get_recent_activity(limit=limit)
    return jsonify({'items': items})


@app.route('/api/persist-records', methods=['POST'])
def persist_records():
    """
    Persist email records to database - called periodically from frontend
    """
    try:
        data = request.get_json() or {}
        records = data.get('records', [])
        
        if not records:
            return jsonify({'status': 'ok', 'persisted': 0})
        
        # Track which records were already in DB and which are new
        persisted_count = 0
        for record in records:
            try:
                # Check if record already exists by id or received_at + sender + subject
                existing = mail_store.get_email_by_fingerprint(
                    sender=record.get('sender', ''),
                    subject=record.get('subject', ''),
                    received_at=record.get('received_at', '')
                )
                
                if not existing:
                    # Only save new records
                    mail_store.save_email(
                        sender=record.get('sender', ''),
                        recipient=record.get('recipient', ''),
                        subject=record.get('subject', ''),
                        body=record.get('body', ''),
                        has_attachment=int(record.get('has_attachment', 0)),
                        links_count=int(record.get('links_count', 0)),
                        sender_domain=record.get('sender_domain', ''),
                        source_uid=None,
                        source_mailbox=None,
                        prediction=record.get('prediction', 'legitimate'),
                        probability=float(record.get('probability', 0)),
                        confidence=float(record.get('confidence', 0))
                    )
                    persisted_count += 1
            except Exception as e:
                continue
        
        return jsonify({'status': 'ok', 'persisted': persisted_count})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Under debug reloader, start SMTP only in the reloader child process.
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        start_local_smtp_server()
        if AUTO_SYNC_ON_START:
            try:
                sync_inbox(limit=20)
            except Exception:
                pass

        print(f"[SmartJasus] Web server: http://127.0.0.1:{APP_PORT} (bind {APP_HOST}:{APP_PORT})")
        print(f"[SmartJasus] Local SMTP: {LOCAL_SMTP_HOST}:{LOCAL_SMTP_PORT}")

    app.run(debug=True, host=APP_HOST, port=APP_PORT, threaded=True)



# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Your account has been compromised. Click here to secure it!\", \"subject\": \"Urgent: Account Security Alert\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Can you review my document?\"}"



# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Reset your password now to avoid lockout. Click here to secure it!\", \"subject\": \"Urgent: Account Security Alert\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Final notice: subscription expired.\", \"subject\": \"Unusual Login Attempt\", \"has_attachment\": 1, \"links_count\": 2, \"sender_domain\": \"travelprizes.org\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Monthly newsletter - May Edition\", \"subject\": \"Company Newsletter\", \"has_attachment\": 0, \"links_count\": 0, \"sender_domain\": \"company.com\", \"urgent_keywords\": 0}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Reset your password now to avoid lockout.\", \"subject\": \"Password Reset Required\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Your Netflix account has been suspended.\", \"subject\": \"Netflix Account Notice\", \"has_attachment\": 1, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"our account is on hold. Log in now to avoid suspension.\"}"