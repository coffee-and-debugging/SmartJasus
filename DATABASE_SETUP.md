# PostgreSQL Setup Guide For CatchFish

This document reflects the current implementation in app.py and local_mail_server.py.

Important update:

- You do not need a separate migration script.
- Database, table, and indexes are auto-initialized by the application at startup.

## 1. Prerequisites

- PostgreSQL installed
- PostgreSQL server running
- A PostgreSQL user with create database permission (or an already-created target database)
- Python environment with dependencies from requirements.txt

Install PostgreSQL (examples):

- Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib
- macOS: brew install postgresql
- Windows: install from postgresql.org installer

Start PostgreSQL service (examples):

- Ubuntu/Debian: sudo systemctl start postgresql
- macOS: brew services start postgresql

## 2. Configure Database Variables In .env

Set these values in project .env:

DB_HOST=localhost
DB_PORT=5432
DB_NAME=emailserver
DB_USER=postgres
DB_PASSWORD=your_password

Notes:

- DB_USER and DB_PASSWORD must be valid for your PostgreSQL instance.
- DB_NAME will be created automatically if user permissions allow it.

## 3. Verify PostgreSQL Access Before Running App

Run:

psql -U postgres -h localhost -p 5432 -c "SELECT 1;"

If this fails, fix authentication or service state first.

## 4. Install Python Dependencies

Run:

pip install -r requirements.txt

This includes psycopg2-binary, so no extra DB package step is required.

## 5. Start CatchFish

Run:

python app.py

At startup, MailStore auto-runs:

1. create database if needed
2. create emails table if needed
3. create indexes if needed

If QUIET_STARTUP=false is set in environment, you will see explicit DB init logs.

## 6. Validate That Schema Exists

Check table exists:

psql -U postgres -d emailserver -h localhost -c "\dt"

Check indexes:

psql -U postgres -d emailserver -h localhost -c "\di"

## 7. Current Database Schema

Table: emails

- id SERIAL PRIMARY KEY
- received_at TEXT NOT NULL
- sender TEXT NOT NULL
- recipient TEXT NOT NULL
- subject TEXT
- body TEXT
- has_attachment INTEGER NOT NULL DEFAULT 0
- links_count INTEGER NOT NULL DEFAULT 0
- sender_domain TEXT
- source_uid TEXT UNIQUE
- source_mailbox TEXT
- prediction TEXT NOT NULL
- probability REAL NOT NULL
- confidence REAL NOT NULL
- created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

Indexes created by app:

- idx_emails_recipient on recipient, received_at desc
- idx_emails_source_uid on source_uid where source_uid is not null
- idx_emails_prediction on prediction, received_at desc
- idx_emails_received_at on received_at desc

## 8. Commands Removed From Older Docs

The following old step is obsolete and should not be used:

- python migrate_db.py

Reason:

- current codebase has no required external migration command for normal startup
- DB bootstrap is built into MailStore._init_db

## 9. Troubleshooting

### Connection refused

Symptom:

- could not connect to server

Checks:

1. PostgreSQL service is running
2. DB_HOST and DB_PORT are correct
3. firewall is not blocking local port

### Password authentication failed

Checks:

1. DB_USER and DB_PASSWORD in .env are correct
2. pg_hba.conf authentication method permits your login mode

### Permission denied to create database

If DB_USER cannot create DB:

1. create DB manually:
   psql -U postgres -h localhost -c "CREATE DATABASE emailserver;"
2. keep DB_NAME pointing to this DB
3. restart app

### Database exists but writes fail

Checks:

1. table emails exists in DB_NAME
2. DB user has insert/select/update privileges

### Port conflict

Check 5432 usage:

lsof -i :5432

If PostgreSQL runs on a different port, update DB_PORT in .env.

## 10. Backup And Restore

Backup:

pg_dump -U postgres -d emailserver > emailserver_backup.sql

Restore:

psql -U postgres -d emailserver < emailserver_backup.sql

## 11. Recommended First Run Checklist

1. Confirm psql connectivity with SELECT 1
2. Start app.py
3. Open dashboard and hit refresh analytics
4. Call /api/activity and verify records are returned
5. Confirm /api/local-server-status returns local_server_up true
