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
        """Context manager for database connections."""
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
        """Initialize database: create database and tables if they don't exist."""
        # First, create the database if it doesn't exist
        self._create_database()
        
        # Then create tables and indexes
        self._create_tables()
        
        if not QUIET_STARTUP:
            print(f"✓ Database initialized: {self.db_name}@{self.db_host}:{self.db_port}")

    def _create_database(self) -> None:
        """Create the database if it doesn't exist."""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database='postgres',
                user=self.db_user,
                password=self.db_password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.db_name}'")
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {self.db_name}")
                if not QUIET_STARTUP:
                    print(f"  ✓ Created database '{self.db_name}'")
            else:
                if not QUIET_STARTUP:
                    print(f"  ✓ Database '{self.db_name}' exists")
            
            cursor.close()
            conn.close()
        except Exception as e:
            if not QUIET_STARTUP:
                print(f"  ⚠ Could not create database: {e}")
            # Try to continue anyway - maybe database already exists

    def _create_tables(self) -> None:
        """Create tables and indexes if they don't exist."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                
                # Create emails table
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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                if not QUIET_STARTUP:
                    print(f"  ✓ Created/verified tables")
                
                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emails_recipient
                    ON emails(recipient, received_at DESC)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emails_source_uid
                    ON emails(source_uid)
                    WHERE source_uid IS NOT NULL
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emails_prediction
                    ON emails(prediction, received_at DESC)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emails_received_at
                    ON emails(received_at DESC)
                """)
                
                if not QUIET_STARTUP:
                    print(f"  ✓ Created/verified indexes")
                cursor.close()
        except Exception as e:
            raise Exception(f"Failed to create tables: {e}")

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
    ) -> int:
        # Use provided received_at or current time as fallback
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
                        prediction, probability, confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        received_at,
                        sender,
                        recipient,
                        subject,
                        body,
                        int(has_attachment),
                        int(links_count),
                        sender_domain,
                        source_uid,
                        source_mailbox,
                        prediction,
                        float(probability),
                        float(confidence),
                    ),
                )
                row_id = cursor.fetchone()[0]
                cursor.close()
                return row_id
            except psycopg2.IntegrityError:
                # source_uid already exists (duplicate IMAP email)
                conn.rollback()
                if source_uid and received_at:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        UPDATE emails
                        SET received_at = %s
                        WHERE source_uid = %s
                        """,
                        (received_at, source_uid),
                    )
                    conn.commit()
                    cursor.close()
                cursor.close()
                return None

    def get_mailbox(self, recipient: str, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM emails
                WHERE recipient = %s
                ORDER BY received_at DESC
                LIMIT %s
                """,
                (recipient, limit),
            )
            cols = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    def get_alerts(self, limit: int = 100) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM emails
                WHERE prediction = 'phishing'
                ORDER BY received_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            cols = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    def get_recent_activity(self, limit: int = 100) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM emails
                ORDER BY received_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            cols = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    def get_email_by_fingerprint(self, sender: str, subject: str, received_at: str) -> Optional[Dict]:
        """Check if an email already exists by sender+subject+received_at fingerprint"""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id
                FROM emails
                WHERE sender = %s AND subject = %s AND received_at = %s
                LIMIT 1
                """,
                (sender, subject, received_at),
            )
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else None


class LocalMailHandler:
    def __init__(self, mail_store: MailStore, predict_email: Callable[[Dict], Dict]):
        self.mail_store = mail_store
        self.predict_email = predict_email

    async def handle_DATA(self, server, session, envelope):
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
                    text_parts.append(part.get_content())
            body = "\n".join([p for p in text_parts if p])
        else:
            body = message.get_content() or ""

        sender_domain = ""
        if "@" in sender:
            sender_domain = sender.split("@", 1)[1].strip().lower().rstrip(">")

        # Extract received date from email headers
        received_at = None
        date_header = message.get("Date")
        if date_header:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(date_header)
                received_at = dt.isoformat()
            except:
                pass
        
        if not received_at:
            received_at = utc_now_iso()

        for recipient in envelope.rcpt_tos:
            prediction = self.predict_email(
                {
                    "email_text": body,
                    "subject": subject,
                    "has_attachment": has_attachment,
                    "sender_domain": sender_domain,
                }
            )
            self.mail_store.save_email(
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
            )

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
