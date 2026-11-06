#!/usr/bin/env python3
"""Idempotently copy existing SQLite history into MongoDB Atlas.

Dry-run by default. Add --apply to upsert existing alerts, samples, heatmaps,
audit events, and camera settings. Video, evidence images, source paths, and
credentials stay on the local volume.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _rows(conn: sqlite3.Connection, sql: str):
    return conn.execute(sql).fetchall()


def migrate(database_path: Path, cameras_path: Path, apply: bool,
            mongo_url: str, mongo_db: str, batch_size: int = 500) -> dict[str, int]:
    if not database_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {database_path}")
    counts = {"alerts": 0, "samples": 0, "heatmaps": 0,
              "audit_events": 0, "cameras": 0}
    client = None
    db = None
    if apply:
        if not mongo_url:
            raise ValueError("MONGODB_URL is required with --apply")
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=7000,
                             connectTimeoutMS=7000, appname="GodavariHistoryMigration")
        client.admin.command("ping")
        db = client[mongo_db]

    conn = sqlite3.connect(str(database_path))
    conn.row_factory = sqlite3.Row
    try:
        alert_columns = {r[1] for r in conn.execute("PRAGMA table_info(alerts)")}
        alerts = []
        if alert_columns:
            for row in _rows(conn, "SELECT * FROM alerts"):
                item = dict(row)
                try:
                    item["context"] = json.loads(item.get("context") or "{}")
                except (TypeError, json.JSONDecodeError):
                    item["context"] = {}
                item["_id"] = item["id"]
                alerts.append(item)

        samples = []
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='samples'").fetchone():
            for row in _rows(conn, "SELECT * FROM samples"):
                item = dict(row)
                for field in ("zones", "lines", "transitions"):
                    try:
                        item[field] = json.loads(item.get(field) or "[]")
                    except (TypeError, json.JSONDecodeError):
                        item[field] = []
                item["_id"] = f"{item['cam_id']}:{item['ts']:.6f}"
                samples.append(item)

        heatmaps = []
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='heatmaps'").fetchone():
            for row in _rows(conn, "SELECT * FROM heatmaps"):
                item = dict(row)
                try:
                    item["grid"] = json.loads(item.pop("grid", "{}") or "{}")
                except (TypeError, json.JSONDecodeError):
                    item["grid"] = {}
                item["_id"] = f"{item['cam_id']}:{item['ts']:.6f}"
                heatmaps.append(item)

        audits = []
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit_log'").fetchone():
            for row in _rows(conn, "SELECT * FROM audit_log"):
                item = dict(row)
                try:
                    item["detail"] = json.loads(item.get("detail") or "{}")
                except (TypeError, json.JSONDecodeError):
                    item["detail"] = {}
                item["_id"] = item.pop("id")
                audits.append(item)

        cameras = []
        if cameras_path.is_file():
            from app.mongo_sync import camera_document
            try:
                raw = json.loads(cameras_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = []
            if isinstance(raw, list):
                cameras = [doc for doc in (camera_document(c) for c in raw)
                           if doc is not None]

        groups = {"alerts": alerts, "samples": samples, "heatmaps": heatmaps,
                  "audit_events": audits, "cameras": cameras}
        for name, documents in groups.items():
            counts[name] = len(documents)
            if apply and documents:
                for start in range(0, len(documents), batch_size):
                    operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True)
                                  for doc in documents[start:start + batch_size]]
                    db[name].bulk_write(operations, ordered=False)
    finally:
        conn.close()
        if client:
            client.close()
    return counts


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path,
                        default=Path(os.environ.get("CROWD_DATA_DIR", "data")) / "crowd.db")
    parser.add_argument("--cameras", type=Path,
                        default=Path(os.environ.get("CROWD_DATA_DIR", "data")) / "cameras.json")
    parser.add_argument("--apply", action="store_true", help="perform the MongoDB upserts; default is dry-run")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 5000:
        parser.error("--batch-size must be between 1 and 5000")
    counts = migrate(args.sqlite, args.cameras, args.apply,
                     os.environ.get("MONGODB_URL", ""),
                     os.environ.get("MONGODB_DB_NAME", "godavari"), args.batch_size)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run",
                      "upserted_source_documents": counts}, indent=2))


if __name__ == "__main__":
    main()
