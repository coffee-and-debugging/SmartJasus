import json
import os
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from typing import Callable, Dict, List, Optional
from contextlib import contextmanager

from aiosmtpd.controller import Controller

QUIET_STARTUP = os.getenv('QUIET_STARTUP', 'true').lower() == 'true'
LOCAL_DOMAIN = os.getenv('LOCAL_DOMAIN', 'local.com')

# Default local users created on first start
DEFAULT_LOCAL_USERS = [
    ('admin', f'admin@{LOCAL_DOMAIN}'),
    ('analyst', f'analyst@{LOCAL_DOMAIN}'),
    ('user1', f'user1@{LOCAL_DOMAIN}'),
    ('user2', f'user2@{LOCAL_DOMAIN}'),
]

# Load reference data from datasets/ folder (same as app.py / train.py)
_BASE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = os.path.join(_BASE, 'datasets')

def _load_json_set(filename: str, key: str) -> set:
    path = os.path.join(_DATASETS, filename)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return set(json.load(f).get(key, []))
    return set()

LEGITIMATE_DOMAINS: set = _load_json_set('legitimate_domains.json', 'domains')
LEGITIMATE_DOMAINS.add(LOCAL_DOMAIN)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MailStore:
    def __init__(self, db_host: str, db_port: int, db_name: str, db_user: str, db_password: str):
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self) -> None:
        self._create_database()
        self._create_tables()
        self._seed_default_users()
        if not QUIET_STARTUP:
            print(f"✓ Database initialized: {self.db_name}@{self.db_host}:{self.db_port}")

    def _create_database(self) -> None:
        try:
            conn = psycopg2.connect(
                host=self.db_host, port=self.db_port,
                database='postgres', user=self.db_user, password=self.db_password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.db_name}'")
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {self.db_name}")
                if not QUIET_STARTUP:
                    print(f"  ✓ Created database '{self.db_name}'")
            cursor.close()
            conn.close()
        except Exception as e:
            if not QUIET_STARTUP:
                print(f"  ⚠ Could not create database: {e}")

    def _create_tables(self) -> None:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                # ── emails table ───────────────────────────────────────────────
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS emails (
                        id SERIAL PRIMARY KEY,
                        received_at TEXT NOT NULL,
                        sender TEXT NOT NULL,
                        recipient TEXT NOT NULL,
                        subject TEXT,
                        body TEXT,
                        has_attachment INTEGER NOT NULL DEFAULT 0,
                        links_count INTEGER NOT NULL DEFAULT 0,
                        sender_domain TEXT,
                        source_uid TEXT UNIQUE,
                        source_mailbox TEXT,
                        prediction TEXT NOT NULL,
                        probability REAL NOT NULL,
                        confidence REAL NOT NULL,
                        features_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Add features_json column if upgrading from older schema
                cursor.execute("""
                    DO $$ BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='emails' AND column_name='features_json'
                        ) THEN
                            ALTER TABLE emails ADD COLUMN features_json TEXT;
                        END IF;
                    END $$;
                """)

                # ── local users table ──────────────────────────────────────────
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS local_users (
                        id SERIAL PRIMARY KEY,
                        username TEXT NOT NULL UNIQUE,
                        email TEXT NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # ── scan logs table (detailed analysis trail) ──────────────────
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scan_logs (
                        id SERIAL PRIMARY KEY,
                        email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
                        scanned_at TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        raw_probability REAL NOT NULL,
                        adjusted_probability REAL NOT NULL,
                        confidence REAL NOT NULL,
                        rule_adjustments TEXT,
                        features_snapshot TEXT,
                        error_info TEXT
                    )
                """)

                # ── indexes ────────────────────────────────────────────────────
                for idx_sql in [
                    "CREATE INDEX IF NOT EXISTS idx_emails_recipient ON emails(recipient, received_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_emails_source_uid ON emails(source_uid) WHERE source_uid IS NOT NULL",
                    "CREATE INDEX IF NOT EXISTS idx_emails_prediction ON emails(prediction, received_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_emails_received_at ON emails(received_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_scan_logs_email_id ON scan_logs(email_id)",
                ]:
                    cursor.execute(idx_sql)

                cursor.close()
        except Exception as e:
            raise Exception(f"Failed to create tables: {e}")

    def _seed_default_users(self) -> None:
        """Create default local users if they don't exist."""
        for username, email in DEFAULT_LOCAL_USERS:
            self.add_user(username, email, silent=True)

    # ── User management ────────────────────────────────────────────────────────

    def add_user(self, username: str, email: str, silent: bool = False) -> bool:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO local_users (username, email) VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING id",
                    (username.lower(), email.lower())
                )
                inserted = cursor.fetchone() is not None
                cursor.close()
                if inserted and not silent and not QUIET_STARTUP:
                    print(f"  ✓ Created local user: {email}")
                return inserted
        except Exception:
            return False

    def get_users(self) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, created_at FROM local_users ORDER BY username")
            cols = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    def user_exists(self, email: str) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM local_users WHERE email = %s", (email.lower(),))
            exists = cursor.fetchone() is not None
            cursor.close()
        return exists

    # ── Email storage ──────────────────────────────────────────────────────────

    def save_email(
        self,
        sender: str,
        recipient: str,
        subject: str,
        body: str,
        has_attachment: int,
        links_count: int,
        sender_domain: str,
        prediction: str,
        probability: float,
        confidence: float,
        received_at: str = None,
        source_uid: str = None,
        source_mailbox: str = None,
        features_json: str = None,
    ) -> Optional[int]:
        if not received_at:
            received_at = utc_now_iso()

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO emails (
                        received_at, sender, recipient, subject, body,
                        has_attachment, links_count, sender_domain,
                        source_uid, source_mailbox,
                        prediction, probability, confidence, features_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        received_at, sender, recipient, subject, body,
                        int(has_attachment), int(links_count), sender_domain,
                        source_uid, source_mailbox,
                        prediction, float(probability), float(confidence), features_json,
                    ),
                )
                row_id = cursor.fetchone()[0]
                cursor.close()
                return row_id
            except psycopg2.IntegrityError:
                conn.rollback()
                if source_uid and received_at:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE emails SET received_at = %s WHERE source_uid = %s",
                        (received_at, source_uid),
                    )
                    conn.commit()
                    cursor.close()
                cursor.close()
                return None

    def save_scan_log(
        self,
        email_id: int,
        prediction: str,
        raw_probability: float,
        adjusted_probability: float,
        confidence: float,
        rule_adjustments: str = None,
        features_snapshot: str = None,
        error_info: str = None,
    ) -> None:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO scan_logs (
                        email_id, scanned_at, prediction,
                        raw_probability, adjusted_probability, confidence,
                        rule_adjustments, features_snapshot, error_info
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        email_id, utc_now_iso(), prediction,
                        float(raw_probability), float(adjusted_probability), float(confidence),
                        rule_adjustments, features_snapshot, error_info,
                    ),
                )
                cursor.close()
        except Exception:
            pass  # Logs should never crash the main flow

    def get_mailbox(self, recipient: str, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM emails WHERE recipient = %s ORDER BY received_at DESC LIMIT %s",
                (recipient, limit),
            )
            cols = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    def get_alerts(self, limit: int = 200, recipient: str = None) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            if recipient:
                cursor.execute(
                    "SELECT * FROM emails WHERE prediction = 'phishing' AND LOWER(recipient) = %s ORDER BY received_at DESC LIMIT %s",
                    (recipient.lower(), limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM emails WHERE prediction = 'phishing' ORDER BY received_at DESC LIMIT %s",
                    (limit,),
                )
            cols = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    def get_recent_activity(self, limit: int = 100) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM emails ORDER BY received_at DESC LIMIT %s",
                (limit,),
            )
            cols = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    def get_scan_logs(self, limit: int = 200) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sl.*, e.sender, e.recipient, e.subject
                FROM scan_logs sl
                LEFT JOIN emails e ON sl.email_id = e.id
                ORDER BY sl.scanned_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            cols = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    def get_email_by_fingerprint(self, sender: str, subject: str, received_at: str) -> Optional[int]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM emails WHERE sender = %s AND subject = %s AND received_at = %s LIMIT 1",
                (sender, subject, received_at),
            )
            row = cursor.fetchone()
            cursor.close()
        return row[0] if row else None

    def get_stats(self) -> Dict:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE prediction = 'phishing') AS phishing,
                    COUNT(*) FILTER (WHERE prediction = 'legitimate') AS legitimate,
                    ROUND(AVG(confidence)::numeric, 4) AS avg_confidence
                FROM emails
            """)
            row = cursor.fetchone()
            cursor.close()
        if row:
            return {
                'total': row[0], 'phishing': row[1],
                'legitimate': row[2], 'avg_confidence': float(row[3] or 0),
            }
        return {'total': 0, 'phishing': 0, 'legitimate': 0, 'avg_confidence': 0.0}


# ── SMTP Handler ───────────────────────────────────────────────────────────────

class LocalMailHandler:
    def __init__(self, mail_store: MailStore, predict_email: Callable[[Dict], Dict]):
        self.mail_store = mail_store
        self.predict_email = predict_email

    async def handle_DATA(self, server, session, envelope):
        import json
        message = BytesParser(policy=policy.default).parsebytes(envelope.content)

        sender = message.get("From", envelope.mail_from or "unknown@local")
        subject = message.get("Subject", "")

        body = ""
        has_attachment = 0
        if message.is_multipart():
            text_parts = []
            for part in message.walk():
                disposition = part.get_content_disposition()
                if disposition == "attachment":
                    has_attachment = 1
                    continue
                if part.get_content_type() == "text/plain":
                    text_parts.append(part.get_content() or "")
            body = "\n".join([p for p in text_parts if p])
        else:
            body = message.get_content() or ""

        sender_domain = ""
        if "@" in sender:
            sender_domain = sender.split("@", 1)[1].strip().lower().rstrip(">")

        received_at = None
        date_header = message.get("Date")
        if date_header:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(date_header)
                received_at = dt.isoformat()
            except Exception:
                pass
        if not received_at:
            received_at = utc_now_iso()

        for recipient in envelope.rcpt_tos:
            try:
                prediction = self.predict_email({
                    "email_text": body,
                    "subject": subject,
                    "has_attachment": has_attachment,
                    "sender_domain": sender_domain,
                })
                features_json = json.dumps(prediction.get('features_used', {}))
                email_id = self.mail_store.save_email(
                    sender=sender,
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    has_attachment=has_attachment,
                    links_count=prediction["features_used"].get("links_count", 0),
                    sender_domain=sender_domain,
                    received_at=received_at,
                    prediction=prediction["prediction"],
                    probability=prediction["probability"],
                    confidence=prediction["confidence"],
                    features_json=features_json,
                )
                if email_id:
                    self.mail_store.save_scan_log(
                        email_id=email_id,
                        prediction=prediction["prediction"],
                        raw_probability=prediction.get("raw_probability", prediction["probability"]),
                        adjusted_probability=prediction["probability"],
                        confidence=prediction["confidence"],
                        rule_adjustments=json.dumps(prediction.get("rule_adjustments", [])),
                        features_snapshot=features_json,
                    )
            except Exception:
                pass  # Do not crash the SMTP session

        return "250 Message accepted for local delivery"


class LocalSMTPServer:
    def __init__(self, handler: LocalMailHandler, host: str = "127.0.0.1", port: int = 1025):
        self.host = host
        self.port = port
        self._controller = Controller(handler, hostname=self.host, port=self.port)

    def start(self) -> None:
        self._controller.start()

    def stop(self) -> None:
        self._controller.stop()
