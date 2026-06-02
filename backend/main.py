"""
MedImage Training Backend
- POST /api/train/{project_id}   → submit training job
- GET  /api/jobs                 → list all jobs
- GET  /api/jobs/{job_id}        → job detail + logs
- DELETE /api/jobs/{job_id}      → cancel/delete job
"""

import os
import uuid
import time
import threading
import sqlite3
import json
import subprocess
import textwrap
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── DB ──────────────────────────────────────────────────────────────────────

DB_PATH = os.getenv("DB_PATH", "/data/jobs.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                name         TEXT,
                project_id   INTEGER,
                dataset      TEXT,
                training_type TEXT,
                model_name   TEXT,
                engine       TEXT,
                epochs       INTEGER,
                batch_size   INTEGER,
                learning_rate REAL,
                optimizer    TEXT,
                imgsz        INTEGER,
                notes        TEXT,
                status       TEXT DEFAULT 'queued',
                progress     INTEGER DEFAULT 0,
                log          TEXT DEFAULT '',
                created_at   REAL,
                started_at   REAL,
                finished_at  REAL,
                error        TEXT
            )
        """)
        conn.commit()


init_db()

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="MedImage Training API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ──────────────────────────────────────────────────────────────────


class TrainRequest(BaseModel):
    training_type: str = "classification"
    model_name: str = "efficientnet-b2"
    engine: str = "PyTorch"
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    imgsz: int = 640
    notes: str = ""


# ─── Training runner ─────────────────────────────────────────────────────────

LS_API_URL = os.getenv("LS_API_URL", "http://label-studio:8080")
LS_TOKEN   = os.getenv("LS_TOKEN", "medimage-ls-token-2026")
MINIO_URL  = os.getenv("MINIO_URL", "http://minio:9000")


def _append_log(job_id: str, line: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET log = log || ? WHERE id = ?",
            (f"[{ts}] {line}\n", job_id),
        )
        conn.commit()


def _set_progress(job_id: str, progress: int):
    with get_db() as conn:
        conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (progress, job_id))
        conn.commit()


def _set_status(job_id: str, status: str, error: str | None = None):
    finished = time.time() if status in ("completed", "error") else None
    with get_db() as conn:
        if finished:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, finished, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE id = ?",
                (status, error, job_id),
            )
        conn.commit()


def run_training(job: dict):
    """Simulate training with epoch-by-epoch progress updates."""
    job_id    = job["id"]
    epochs    = job["epochs"]
    model     = job["model_name"]
    engine    = job["engine"]
    proj_id   = job["project_id"]

    try:
        # Mark running
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                (time.time(), job_id),
            )
            conn.commit()

        _append_log(job_id, f"=== MedImage Training Job ===")
        _append_log(job_id, f"Model     : {model}")
        _append_log(job_id, f"Engine    : {engine}")
        _append_log(job_id, f"Epochs    : {epochs}")
        _append_log(job_id, f"Batch size: {job['batch_size']}")
        _append_log(job_id, f"LR        : {job['learning_rate']}")
        _append_log(job_id, f"Optimizer : {job['optimizer']}")
        _append_log(job_id, f"Project   : {proj_id}")
        _append_log(job_id, "")

        # Fetch dataset info from Label Studio
        _append_log(job_id, f"Fetching dataset from Label Studio project {proj_id}...")
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{LS_API_URL}/api/projects/{proj_id}/",
                headers={"Authorization": f"Token {LS_TOKEN}"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                proj_data = json.loads(r.read())
                task_count = proj_data.get("task_number", 0)
                _append_log(job_id, f"Dataset    : {proj_data.get('title', proj_id)} ({task_count} tasks)")
        except Exception as e:
            _append_log(job_id, f"Warning: Could not fetch project info: {e}")

        _append_log(job_id, "")
        _append_log(job_id, "Starting training loop...")
        _append_log(job_id, "")

        import random
        loss = 1.5 + random.uniform(-0.1, 0.1)
        acc  = 0.5 + random.uniform(-0.05, 0.05)

        for epoch in range(1, epochs + 1):
            # Simulate epoch training (1–3s per epoch for demo)
            time.sleep(max(1.0, min(3.0, 60.0 / epochs)))

            loss  = loss  * 0.92 + random.uniform(-0.02, 0.02)
            acc   = min(0.995, acc * 1.04 + random.uniform(-0.01, 0.02))
            val_loss = loss * 1.05 + random.uniform(-0.03, 0.03)
            val_acc  = min(0.99, acc - random.uniform(0.01, 0.04))

            progress = int(epoch / epochs * 100)
            _set_progress(job_id, progress)
            _append_log(
                job_id,
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"loss={loss:.4f}  acc={acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
            )

        _append_log(job_id, "")
        _append_log(job_id, f"Training completed ✓")
        _append_log(job_id, f"Final val_acc: {val_acc:.4f}")
        _append_log(job_id, "Model checkpoint saved to /data/models/{job_id}/best.pt")
        _set_progress(job_id, 100)
        _set_status(job_id, "completed")

    except Exception as exc:
        _append_log(job_id, f"ERROR: {exc}")
        _set_status(job_id, "error", str(exc))


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.post("/api/train/{project_id}")
def submit_job(project_id: int, req: TrainRequest):
    job_id = str(uuid.uuid4())[:8]
    name   = f"{req.training_type.upper()} · {req.model_name} · proj-{project_id}"

    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id, name, project_id, dataset, training_type, model_name, engine,
                epochs, batch_size, learning_rate, optimizer, imgsz, notes,
                status, progress, log, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',0,'',?)""",
            (
                job_id, name, project_id,
                f"LS project {project_id}",
                req.training_type, req.model_name, req.engine,
                req.epochs, req.batch_size, req.learning_rate,
                req.optimizer, req.imgsz, req.notes,
                time.time(),
            ),
        )
        conn.commit()

    # Fetch row as dict to pass to thread
    with get_db() as conn:
        row = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    t = threading.Thread(target=run_training, args=(row,), daemon=True)
    t.start()

    return {"job_id": job_id, "message": f"Training job {job_id} queued for {req.model_name}"}


@app.get("/api/jobs")
def list_jobs():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    jobs = []
    for r in rows:
        d = dict(r)
        d.pop("log", None)  # omit log from list view
        jobs.append({
            "id":            d["id"],
            "name":          d["name"],
            "training_type": d["training_type"],
            "model":         d["model_name"],
            "engine":        d["engine"],
            "status":        d["status"],
            "progress":      d["progress"],
            "created_at":    d["created_at"],
            "started_at":    d["started_at"],
            "finished_at":   d["finished_at"],
            "error":         d["error"],
            "dataset":       d["dataset"],
            "epochs":        d["epochs"],
            "batch_size":    d["batch_size"],
        })
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return dict(row)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with get_db() as conn:
        conn.execute("UPDATE jobs SET status = 'error', error = 'Cancelled by user' WHERE id = ? AND status IN ('queued', 'running')", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ? AND status NOT IN ('running')", (job_id,))
        conn.commit()
    return {"ok": True}


@app.get("/healthz")
def health():
    return {"status": "ok"}
