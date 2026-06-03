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
import random
import math
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
                error        TEXT,
                hidden_in_jobs   INTEGER DEFAULT 0,
                hidden_in_models INTEGER DEFAULT 0
            )
        """)
        # Migrate existing DB: add columns if missing
        for col, defval in [
            ("hidden_in_jobs",   "INTEGER DEFAULT 0"),
            ("hidden_in_models", "INTEGER DEFAULT 0"),
            ("source",           "TEXT DEFAULT 'trained'"),
            ("source_url",       "TEXT DEFAULT ''"),
            ("lora_rank",        "INTEGER DEFAULT 16"),
            ("quantization",     "TEXT DEFAULT '4bit'"),
            ("max_seq_len",      "INTEGER DEFAULT 2048"),
            ("chat_template",    "TEXT DEFAULT 'alpaca'"),
            ("grad_accum",       "INTEGER DEFAULT 4"),
            ("text_dataset",     "TEXT DEFAULT ''"),
        ]:
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {defval}")
            except Exception:
                pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_datasets (
                id         TEXT PRIMARY KEY,
                name       TEXT,
                format     TEXT DEFAULT 'alpaca',
                path       TEXT,
                row_count  INTEGER DEFAULT 0,
                size_bytes INTEGER DEFAULT 0,
                created_at REAL
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
    # LLM-specific fields
    lora_rank: int = 16
    quantization: str = "4bit"      # 4bit | 8bit | full
    max_seq_len: int = 2048
    chat_template: str = "alpaca"   # alpaca | chatml | llama3 | gemma
    grad_accum: int = 4
    text_dataset: str = ""          # text_dataset id or name


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


def run_llm_training(job: dict):
    """Simulate Unsloth/TRL LLM or VLM fine-tuning."""
    job_id     = job["id"]
    epochs     = max(1, job["epochs"])
    model      = job["model_name"]
    engine     = job["engine"]
    lora_rank  = job.get("lora_rank", 16)
    quant      = job.get("quantization", "4bit")
    max_seq    = job.get("max_seq_len", 2048)
    tmpl       = job.get("chat_template", "alpaca")
    grad_accum = job.get("grad_accum", 4)
    t_type     = job.get("training_type", "llm-text")
    text_ds    = job.get("text_dataset", "custom dataset")

    try:
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE id=?",
                (time.time(), job_id),
            )
            conn.commit()

        is_vlm = (t_type == "vlm-finetune")
        _append_log(job_id, f"=== MedImage {'VLM' if is_vlm else 'LLM'} Fine-tuning ===")
        _append_log(job_id, f"Model      : {model}")
        _append_log(job_id, f"Engine     : {engine}")
        _append_log(job_id, f"Quantize   : {quant} {'(QLoRA)' if quant != 'full' else '(full fine-tune)'}")
        _append_log(job_id, f"LoRA rank  : r={lora_rank}  alpha={lora_rank*2}")
        _append_log(job_id, f"Max seq len: {max_seq}")
        _append_log(job_id, f"Chat tmpl  : {tmpl}")
        _append_log(job_id, f"Grad accum : {grad_accum}")
        _append_log(job_id, f"Epochs     : {epochs}")
        _append_log(job_id, f"Dataset    : {text_ds}")
        _append_log(job_id, "")

        # Phase 1: Load model + tokenizer
        _set_progress(job_id, 5)
        _append_log(job_id, f"[1/6] Loading {model} with Unsloth FastLanguageModel ...")
        time.sleep(random.uniform(2.0, 4.0))
        if is_vlm:
            _append_log(job_id, f"  ✓ Vision encoder loaded (patch_size=14, num_patches=729)")
        _append_log(job_id, f"  ✓ Base model loaded in {quant} mode ({random.randint(3, 8)} GB VRAM)")
        _append_log(job_id, f"  ✓ Tokenizer loaded (vocab_size={random.randint(32000, 128000)})")
        _append_log(job_id, "")

        # Phase 2: Apply LoRA
        _set_progress(job_id, 12)
        _append_log(job_id, f"[2/6] Applying LoRA adapters (PEFT) ...")
        time.sleep(random.uniform(0.5, 1.5))
        trainable = round(random.uniform(0.8, 2.5), 2)
        total = round(random.uniform(7.0, 14.0), 1)
        _append_log(job_id, f"  trainable params: {trainable}M / {total}B ({trainable/total*100:.2f}%) -- LoRA r={lora_rank}")
        _append_log(job_id, f"  target modules  : q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")
        _append_log(job_id, "")

        # Phase 3: Prepare dataset
        _set_progress(job_id, 20)
        _append_log(job_id, f"[3/6] Preparing dataset ({tmpl} format) ...")
        time.sleep(random.uniform(1.0, 2.5))
        n_train = random.randint(800, 5000)
        n_val   = int(n_train * 0.1)
        _append_log(job_id, f"  ✓ Loaded {n_train + n_val} samples from {text_ds}")
        _append_log(job_id, f"  ✓ Train: {n_train}  |  Val: {n_val}")
        _append_log(job_id, f"  ✓ Avg token length: {random.randint(128, 512)} tokens")
        _append_log(job_id, f"  ✓ Tokenized & packed to max_seq_len={max_seq}")
        _append_log(job_id, "")

        # Phase 4: Training steps
        _set_progress(job_id, 25)
        _append_log(job_id, f"[4/6] Training (SFTTrainer with DeepSpeed Zero-2 offload) ...")
        _append_log(job_id, "")

        steps_per_epoch = max(10, n_train // (job.get("batch_size", 2) * grad_accum))
        total_steps = steps_per_epoch * epochs
        loss = 2.4 + random.uniform(-0.2, 0.2)
        lr   = job.get("learning_rate", 2e-4)
        step_delay = max(0.3, min(2.0, 60.0 / total_steps))

        for step in range(1, total_steps + 1):
            time.sleep(step_delay)
            loss  = loss  * random.uniform(0.96, 0.999) + random.uniform(-0.01, 0.01)
            loss  = max(0.05, loss)
            pct   = int(25 + (step / total_steps) * 65)
            _set_progress(job_id, pct)

            if step == 1 or step % max(1, total_steps // 20) == 0 or step == total_steps:
                cur_epoch = int((step - 1) / steps_per_epoch) + 1
                tps       = random.randint(180, 420)
                perp      = round(math.exp(loss), 3)
                _append_log(
                    job_id,
                    f"Step [{step:>4}/{total_steps}] epoch={cur_epoch}/{epochs}  "
                    f"loss={loss:.4f}  perplexity={perp:.3f}  "
                    f"lr={lr:.2e}  tokens/s={tps}"
                )

        _append_log(job_id, "")

        # Phase 5: Eval
        _set_progress(job_id, 92)
        _append_log(job_id, f"[5/6] Running evaluation on validation set ...")
        time.sleep(random.uniform(1.0, 2.0))
        eval_loss = loss * random.uniform(1.02, 1.08)
        _append_log(job_id, f"  eval_loss={eval_loss:.4f}  eval_perplexity={math.exp(eval_loss):.3f}")
        _append_log(job_id, "")

        # Phase 6: Save adapter
        _set_progress(job_id, 97)
        _append_log(job_id, f"[6/6] Saving LoRA adapter ...")
        time.sleep(random.uniform(0.5, 1.5))
        _append_log(job_id, f"  ✓ Adapter saved to /data/models/{job_id}/lora_adapter/")
        _append_log(job_id, f"  ✓ Tokenizer saved")
        _append_log(job_id, "")
        _append_log(job_id, "Fine-tuning complete ✓")
        _append_log(job_id, f"  Base model : {model}")
        _append_log(job_id, f"  Adapter    : LoRA r={lora_rank} ({quant})")
        _append_log(job_id, f"  Final loss : {loss:.4f}  |  perplexity: {math.exp(loss):.3f}")
        _set_progress(job_id, 100)
        _set_status(job_id, "completed")

    except Exception as exc:
        _append_log(job_id, f"ERROR: {exc}")
        _set_status(job_id, "error", str(exc))


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.post("/api/train/{project_id}")
def submit_job(project_id: int, req: TrainRequest):
    job_id = str(uuid.uuid4())[:8]
    is_llm = req.training_type in ("llm-text", "vlm-finetune")
    if is_llm:
        ds_label = req.text_dataset or f"dataset-{project_id}"
        name = f"{req.training_type.upper()} · {req.model_name} · {ds_label}"
    else:
        name = f"{req.training_type.upper()} · {req.model_name} · proj-{project_id}"

    dataset_label = req.text_dataset if (is_llm and req.text_dataset) else f"LS project {project_id}"

    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id, name, project_id, dataset, training_type, model_name, engine,
                epochs, batch_size, learning_rate, optimizer, imgsz, notes,
                lora_rank, quantization, max_seq_len, chat_template, grad_accum, text_dataset,
                status, progress, log, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',0,'',?)""",
            (
                job_id, name, project_id, dataset_label,
                req.training_type, req.model_name, req.engine,
                req.epochs, req.batch_size, req.learning_rate,
                req.optimizer, req.imgsz, req.notes,
                req.lora_rank, req.quantization, req.max_seq_len,
                req.chat_template, req.grad_accum, req.text_dataset,
                time.time(),
            ),
        )
        conn.commit()

    with get_db() as conn:
        row = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    target_fn = run_llm_training if is_llm else run_training
    threading.Thread(target=target_fn, args=(row,), daemon=True).start()

    return {"job_id": job_id, "message": f"Training job {job_id} queued for {req.model_name}"}


@app.get("/api/jobs")
def list_jobs(view: str = "jobs"):
    if view == "models":
        where = "WHERE status = 'completed' AND hidden_in_models = 0"
    else:
        where = "WHERE hidden_in_jobs = 0"
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM jobs {where} ORDER BY created_at DESC"
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
            "source":        d.get("source", "trained"),
        })
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return dict(row)


class PatchJobRequest(BaseModel):
    name: str | None = None
    notes: str | None = None

@app.patch("/api/jobs/{job_id}")
def patch_job(job_id: str, req: PatchJobRequest):
    with get_db() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if req.name is not None:
            conn.execute("UPDATE jobs SET name = ? WHERE id = ?", (req.name, job_id))
        if req.notes is not None:
            conn.execute("UPDATE jobs SET notes = ? WHERE id = ?", (req.notes, job_id))
        conn.commit()
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, from_view: str = "jobs"):
    with get_db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if from_view == "models":
            conn.execute("UPDATE jobs SET hidden_in_models = 1 WHERE id = ?", (job_id,))
        else:
            # Cancel if still active, then hide from jobs view
            if row["status"] in ("queued", "running"):
                conn.execute("UPDATE jobs SET status = 'error', error = 'Cancelled by user' WHERE id = ?", (job_id,))
            conn.execute("UPDATE jobs SET hidden_in_jobs = 1 WHERE id = ?", (job_id,))
        conn.commit()
    return {"ok": True}


# ─── Import pretrained model ─────────────────────────────────────────────────

class ImportModelRequest(BaseModel):
    name: str
    training_type: str = "classification"
    model_name: str
    engine: str = "PyTorch"
    source_type: str = "huggingface"   # huggingface | url | builtin
    source_url: str = ""
    notes: str = ""


def _run_import(job_id: str, req: ImportModelRequest):
    """Simulate model download / weight loading."""
    try:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (time.time(), job_id))
            conn.commit()

        steps = [
            (10, f"Resolving source: {req.source_url or req.source_type} …"),
            (30, f"Downloading weights for {req.model_name} …"),
            (55, "Verifying checksum …"),
            (75, "Loading model architecture …"),
            (90, "Registering model in library …"),
            (100, f"Import complete ✓  ({req.model_name})"),
        ]
        for pct, msg in steps:
            time.sleep(random.uniform(0.5, 1.2))
            _set_progress(job_id, pct)
            _append_log(job_id, msg)

        _append_log(job_id, f"Model checkpoint saved to /data/models/{job_id}/weights.pt")
        _set_status(job_id, "completed")
    except Exception as exc:
        _append_log(job_id, f"ERROR: {exc}")
        _set_status(job_id, "error", str(exc))


@app.post("/api/models/import")
def import_model(req: ImportModelRequest):
    job_id = str(uuid.uuid4())[:8]
    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id, name, project_id, dataset, training_type, model_name, engine,
                epochs, batch_size, learning_rate, optimizer, imgsz, notes,
                status, progress, log, created_at, source, source_url)
               VALUES (?,?,0,'pretrained',?,?,?,0,0,0,'—',0,?,'queued',0,'',?,?,?)""",
            (
                job_id, req.name,
                req.training_type, req.model_name, req.engine,
                req.notes, time.time(),
                "imported", req.source_url,
            ),
        )
        conn.commit()

    threading.Thread(target=_run_import, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


# ─── Settings ───────────────────────────────────────────────────────────────

@app.get("/api/settings/{key}")
def get_setting(key: str):
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="not found")
    return {"value": row["value"]}


@app.post("/api/settings/{key}")
def save_setting(key: str, body: dict):
    value = str(body.get("value", ""))
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, value))
        conn.commit()
    return {"ok": True}


# ─── Modal.com Cloud Deploy ──────────────────────────────────────────────────

import re as _re

_modal_state: dict = {
    "status": "idle",    # idle | deploying | running | stopping | error
    "ray_url": None,
    "logs": [],
    "proc": None,
}


class ModalStartRequest(BaseModel):
    token_id: str
    token_secret: str
    gpu_type: str = "T4"
    num_workers: int = 1


def _modal_script(req: ModalStartRequest) -> str:
    gpu_spec = f'gpu="{req.gpu_type}"' if req.gpu_type != "cpu" else "gpu=None"
    workers_block = ""
    if req.num_workers > 0:
        workers_block = textwrap.dedent(f"""

            @app.function(image=ray_image, {gpu_spec}, timeout=86400, memory=8192)
            def ray_worker(head_ip: str):
                import subprocess, time
                subprocess.Popen(["ray", "start", f"--address={{head_ip}}:6379"])
                time.sleep(86400)
        """)
    return textwrap.dedent(f"""\
        import modal, subprocess, time

        app = modal.App("medimage-ray")

        ray_image = (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install("ray[all]>=2.30")
        )

        @app.function(image=ray_image, {gpu_spec}, timeout=86400, memory=16384)
        @modal.web_server(8265, startup_timeout=300)
        def ray_head():
            subprocess.Popen([
                "ray", "start", "--head",
                "--dashboard-host=0.0.0.0", "--dashboard-port=8265",
                "--port=6379",
            ])
            time.sleep(86400)
        {workers_block}
    """)


def _run_modal(req: ModalStartRequest, script_path: str):
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = req.token_id
    env["MODAL_TOKEN_SECRET"] = req.token_secret
    try:
        proc = subprocess.Popen(
            ["python3", "-m", "modal", "serve", script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _modal_state["proc"] = proc
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if line:
                _modal_state["logs"].append(line)
                m = _re.search(r'https://[^\s]+\.modal\.run[^\s]*', line)
                if m:
                    _modal_state["ray_url"] = m.group(0)
                    _modal_state["status"] = "running"
        proc.wait()
        if _modal_state["status"] not in ("idle", "error"):
            _modal_state["status"] = "idle"
    except Exception as exc:
        _modal_state["status"] = "error"
        _modal_state["logs"].append(f"Fatal: {exc}")


@app.post("/api/modal/start")
def modal_start(req: ModalStartRequest):
    if _modal_state["status"] in ("deploying", "running"):
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Already running — stop first")
    _modal_state.update(status="deploying", ray_url=None, logs=[], proc=None)
    script_path = "/tmp/medimage_modal_ray.py"
    with open(script_path, "w") as f:
        f.write(_modal_script(req))
    threading.Thread(target=_run_modal, args=(req, script_path), daemon=True).start()
    return {"status": "deploying"}


@app.post("/api/modal/stop")
def modal_stop():
    proc = _modal_state.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _modal_state.update(status="idle", proc=None, ray_url=None)
    return {"status": "idle"}


@app.get("/api/modal/status")
def modal_status_ep():
    return {
        "status": _modal_state["status"],
        "ray_url": _modal_state["ray_url"],
        "logs": _modal_state["logs"][-30:],
    }


# ─── Text Datasets ─────────────────────────────────────────────────────────────────

import os as _os

@app.post("/api/text-datasets/upload")
async def upload_text_dataset(file: UploadFile = File(...)):
    """Upload a .jsonl (or .json / .txt) text dataset file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("jsonl", "json", "txt", "csv"):
        raise HTTPException(status_code=400, detail="Unsupported format. Use .jsonl, .json, .txt, or .csv")
    ds_id = str(uuid.uuid4())[:8]
    save_dir = "/data/text-datasets"
    _os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{ds_id}.{ext}"
    raw = await file.read()
    size_bytes = len(raw)
    with open(save_path, "wb") as f:
        f.write(raw)
    # Count rows
    row_count = 0
    try:
        import json as _json
        if ext == "jsonl":
            row_count = sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())
        elif ext == "json":
            data = _json.loads(raw)
            row_count = len(data) if isinstance(data, list) else 1
        else:
            row_count = sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())
    except Exception:
        row_count = 0
    # Detect format
    ds_format = "plain"
    try:
        if ext == "jsonl":
            first = _json.loads(raw.decode("utf-8").splitlines()[0])
            if "conversations" in first or "messages" in first:
                ds_format = "sharegpt"
            elif "instruction" in first:
                ds_format = "alpaca"
            elif "text" in first or "content" in first:
                ds_format = "plain"
    except Exception:
        pass
    with get_db() as conn:
        conn.execute(
            "INSERT INTO text_datasets (id, name, format, path, row_count, size_bytes, created_at) VALUES (?,?,?,?,?,?,?)",
            (ds_id, file.filename, ds_format, save_path, row_count, size_bytes, time.time()),
        )
        conn.commit()
    return {"id": ds_id, "name": file.filename, "format": ds_format, "row_count": row_count, "size_bytes": size_bytes}


@app.get("/api/text-datasets")
def list_text_datasets():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM text_datasets ORDER BY created_at DESC").fetchall()
    results = []
    for r in rows:
        d = dict(r)
        # Add preview rows if file exists
        preview = []
        try:
            import json as _json
            with open(d["path"], "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    try:
                        preview.append(_json.loads(line))
                    except Exception:
                        preview.append({"text": line.strip()})
        except Exception:
            pass
        d["preview"] = preview
        results.append(d)
    return {"datasets": results}


@app.delete("/api/text-datasets/{ds_id}")
def delete_text_dataset(ds_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT path FROM text_datasets WHERE id = ?", (ds_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        try:
            _os.remove(row["path"])
        except Exception:
            pass
        conn.execute("DELETE FROM text_datasets WHERE id = ?", (ds_id,))
        conn.commit()
    return {"ok": True}


@app.get("/healthz")
def health():
    return {"status": "ok"}


# ─── Inference (Playground) ──────────────────────────────────────────────────

_CLS_LABELS = ["Normal", "Pneumonia", "COVID-19", "Atelectasis", "Cardiomegaly", "Pleural Effusion", "Infiltration"]

_DET_LABELS = [
    {"label": "nodule",        "color": "#f59e0b"},
    {"label": "mass",          "color": "#ef4444"},
    {"label": "opacity",       "color": "#6366f1"},
    {"label": "consolidation", "color": "#10b981"},
    {"label": "cardiomegaly",  "color": "#ec4899"},
]

_SEG_LABELS = [
    {"label": "lung_left",  "color": "#6366f1"},
    {"label": "lung_right", "color": "#10b981"},
    {"label": "lesion",     "color": "#ef4444"},
    {"label": "effusion",   "color": "#f59e0b"},
]

from fastapi import UploadFile, File, Form as FForm
from typing import Optional

@app.post("/api/inference")
async def run_inference(
    model_id: str = FForm(...),
    image: Optional[UploadFile] = File(None),
    prompt: str = FForm(""),
    system_prompt: str = FForm(""),
):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND status = 'completed'", (model_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found or not completed")

    job = dict(row)
    training_type = job["training_type"]
    fname = (image.filename if image else None) or prompt[:20] or "upload"
    # seed RNG from filename for reproducible-ish demo results
    rng = random.Random(abs(hash(fname)) % (2**31))

    t_start = time.time()

    if training_type == "classification":
        logits = [rng.gauss(0, 1) for _ in _CLS_LABELS]
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        predictions = sorted(
            [{"label": l, "confidence": round(p, 4)} for l, p in zip(_CLS_LABELS, probs)],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:5]
        elapsed = round((time.time() - t_start) * 1000 + rng.uniform(15, 80), 1)
        return {
            "type": "classification",
            "predictions": predictions,
            "top_label": predictions[0]["label"],
            "top_confidence": predictions[0]["confidence"],
            "inference_time_ms": elapsed,
            "model_name": job["model_name"],
        }

    elif training_type == "detection":
        num_det = rng.randint(1, 4)
        detections = []
        for _ in range(num_det):
            cfg = rng.choice(_DET_LABELS)
            x1 = rng.uniform(0.05, 0.55)
            y1 = rng.uniform(0.05, 0.55)
            x2 = min(0.95, x1 + rng.uniform(0.12, 0.35))
            y2 = min(0.95, y1 + rng.uniform(0.12, 0.35))
            detections.append({
                "label":      cfg["label"],
                "confidence": round(rng.uniform(0.55, 0.97), 3),
                "bbox":       [round(x1,4), round(y1,4), round(x2,4), round(y2,4)],
                "color":      cfg["color"],
            })
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        elapsed = round((time.time() - t_start) * 1000 + rng.uniform(50, 130), 1)
        return {
            "type": "detection",
            "detections": detections,
            "count": len(detections),
            "inference_time_ms": elapsed,
            "model_name": job["model_name"],
        }

    elif training_type == "segmentation":
        chosen = rng.sample(_SEG_LABELS, rng.randint(1, len(_SEG_LABELS)))
        masks = []
        for cfg in chosen:
            cx = rng.uniform(0.2, 0.8)
            cy = rng.uniform(0.2, 0.8)
            rx = rng.uniform(0.06, 0.22)
            ry = rng.uniform(0.06, 0.22)
            n_pts = 10
            polygon = [
                [round(cx + rx * math.cos(2 * math.pi * i / n_pts), 4),
                 round(cy + ry * math.sin(2 * math.pi * i / n_pts), 4)]
                for i in range(n_pts)
            ]
            masks.append({
                "label":      cfg["label"],
                "confidence": round(rng.uniform(0.72, 0.96), 3),
                "area_pct":   round(math.pi * rx * ry * 100, 2),
                "color":      cfg["color"],
                "polygon":    polygon,
            })
        elapsed = round((time.time() - t_start) * 1000 + rng.uniform(80, 200), 1)
        return {
            "type": "segmentation",
            "masks": masks,
            "inference_time_ms": elapsed,
            "model_name": job["model_name"],
        }

    elif training_type in ("llm-text", "vlm-finetune"):
        # Text / VL generation — image is consumed but response is text
        prompts = [
            "Based on the imaging findings, this appears consistent with bilateral lower-lobe consolidation suggesting community-acquired pneumonia. Recommend clinical correlation and CBC with differential.",
            "The scan demonstrates a well-defined hyperdense lesion in the right hepatic lobe measuring approximately 3.2 × 2.8 cm. Differential includes hepatocellular carcinoma vs. metastasis. Recommend MRI with contrast and AFP levels.",
            "Findings are consistent with moderate left pleural effusion with associated compressive atelectasis. No pneumothorax identified. Clinical correlation advised.",
            "The retinal fundus image shows neovascularization at the disc (NVD) and multiple flame hemorrhages consistent with proliferative diabetic retinopathy. Urgent ophthalmology referral recommended.",
            "CT demonstrates a 1.5 cm pulmonary nodule in the right upper lobe with spiculated margins. Per Fleischner Society guidelines, recommend PET-CT and multidisciplinary tumor board review.",
        ]
        response = rng.choice(prompts)
        elapsed = round((time.time() - t_start) * 1000 + rng.uniform(200, 800), 1)
        tokens = len(response.split())
        return {
            "type": training_type,
            "response": response,
            "tokens_generated": tokens,
            "tokens_per_second": round(tokens / (elapsed / 1000), 1),
            "inference_time_ms": elapsed,
            "model_name": job["model_name"],
        }

    raise HTTPException(status_code=400, detail=f"Unknown training_type: {training_type}")
