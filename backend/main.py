"""
MedImage Training Backend
- POST /api/train/{project_id}   → submit training job
- GET  /api/jobs                 → list all jobs
- GET  /api/jobs/{job_id}        → job detail + logs
- DELETE /api/jobs/{job_id}      → cancel/delete job
"""

import os
import sys
import uuid
import time
import shutil
import threading
import sqlite3
import json
import subprocess
import textwrap
import random
import math
import asyncio
import base64
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form as FForm, Request, Body, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.websockets import WebSocketState
import io
import zipfile
import asyncio
import json as _json
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
            ("s3_weights_path",  "TEXT DEFAULT ''"),
            ("modal_url",        "TEXT DEFAULT ''"),
            ("modal_api_key",    "TEXT DEFAULT ''"),
            ("ray_serve_url",    "TEXT DEFAULT ''"),
            ("inference_provider", "TEXT DEFAULT ''"),
            ("cluster",          "TEXT DEFAULT 'ray'"),
            ("pipeline_step",    "TEXT DEFAULT ''"),
            ("updated_at",       "TEXT DEFAULT ''"),
            ("user_id",          "TEXT DEFAULT ''"),
            ("num_gpus",         "INTEGER DEFAULT 1"),
            ("ray_submission_id","TEXT DEFAULT ''"),
            ("bulk_run_id",      "TEXT DEFAULT ''"),
        ]:
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {defval}")
            except Exception:
                pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bulk_runs (
                id              TEXT PRIMARY KEY,
                started_at      REAL NOT NULL,
                finished_at     REAL,
                stopped         INTEGER DEFAULT 0,
                provider        TEXT DEFAULT 'ray',
                deploy_enabled  INTEGER DEFAULT 0,
                total           INTEGER DEFAULT 0,
                ok              INTEGER DEFAULT 0,
                failed          INTEGER DEFAULT 0,
                deployed_count  INTEGER DEFAULT 0,
                user_id         TEXT DEFAULT ''
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bulk_runs_started ON bulk_runs(started_at DESC)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bulk_run_jobs (
                bulk_run_id  TEXT NOT NULL,
                row_key      TEXT NOT NULL,
                job_id       TEXT,
                status       TEXT,
                error        TEXT,
                elapsed_sec  REAL DEFAULT 0,
                deployed     INTEGER DEFAULT 0,
                deploy_url   TEXT,
                PRIMARY KEY (bulk_run_id, row_key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_prefs (
                user_id TEXT NOT NULL,
                key     TEXT NOT NULL,
                value   TEXT NOT NULL DEFAULT '',
                updated_at REAL DEFAULT (strftime('%s','now')),
                PRIMARY KEY (user_id, key)
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_history (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL DEFAULT 'default',
                created_at      REAL NOT NULL,
                mode            TEXT NOT NULL,
                model_id        TEXT,
                model_name      TEXT,
                model_type      TEXT,
                image_name      TEXT,
                thumbnail_key   TEXT,
                image_key       TEXT,
                user_prompt     TEXT,
                system_prompt   TEXT,
                result_json     TEXT NOT NULL,
                inference_time_ms INTEGER
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inference_history_user_time "
            "ON inference_history (user_id, created_at DESC)"
        )
        # Migration: add image_key column if missing (existing DBs from before)
        try:
            conn.execute("ALTER TABLE inference_history ADD COLUMN image_key TEXT")
        except Exception:
            pass  # column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS totp_credentials (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                user_label  TEXT NOT NULL DEFAULT 'Authenticator',
                secret      TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                created_at  INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS modal_credentials (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                token_id    TEXT NOT NULL,
                token_secret TEXT NOT NULL,
                updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS huggingface_tokens (
                user_id     TEXT PRIMARY KEY,
                token       TEXT NOT NULL,
                hf_username TEXT DEFAULT '',
                updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
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


# ─── Ray job reconciliation watchdog ──────────────────────────────────────────
#
# Without this, stuck PENDING jobs accumulate on the Ray cluster and hold GPU
# resources forever — every subsequent job then queues behind them and the
# whole pipeline grinds to a halt (the "TRAIN: UNKNOWN error" symptom).
#
# Every 60s the watchdog:
#   1. Lists all Ray jobs from the dashboard
#   2. Stops any PENDING job older than 5 min (orphans OR jobs that the
#      autoscaler can't place)
#   3. Marks the matching medimage DB job as 'error' if the Ray job is
#      FAILED/STOPPED but the DB still says 'running' (catches the race
#      where run_training() crashes before it can _set_status the error)
#
# Runs as a daemon thread so the API can restart without leaking the loop.

_RAY_RECONCILE_INTERVAL_S = 60
_RAY_STUCK_PENDING_AGE_S  = 300   # 5 min
_ray_stopped_pending: set[str] = set()   # submission IDs we already stopped — avoid re-stopping


def _stop_ray_submission(submission_id: str) -> bool:
    """Ask the Ray dashboard to stop a job. Returns True on success.
    For PENDING jobs that won't stop, also tries DELETE."""
    if not submission_id or not submission_id.startswith("raysubmit_"):
        return False
    try:
        import requests as _req
        r = _req.post(f"{RAY_URL}/api/jobs/{submission_id}/stop", timeout=5)
        if r.ok:
            return True
        # STOP failed — try DELETE (works for some terminal-state jobs)
        r2 = _req.delete(f"{RAY_URL}/api/jobs/{submission_id}", timeout=5)
        return r2.ok
    except Exception:
        return False


def _reconcile_ray_jobs() -> dict:
    """One pass: kill stuck PENDING jobs + sync DB state for finished Ray jobs.
    Returns a summary for logging / API responses.
    """
    import requests as _req
    summary = {"stopped_pending": [], "synced_errors": [], "errors": []}
    # Refresh _active_ray_subs from DB. The module-level set is only
    # populated at startup; without this refresh, jobs created after
    # the API booted aren't tracked and the "thread still alive" defer
    # below would never fire for them.
    try:
        with get_db() as _rc:
            for _r in _rc.execute(
                "SELECT ray_submission_id FROM jobs "
                "WHERE ray_submission_id != '' AND status IN ('running','queued')"
            ).fetchall():
                _active_ray_subs.add(_r["ray_submission_id"])
    except Exception:
        pass
    try:
        r = _req.get(f"{RAY_URL}/api/jobs/?limit=200", timeout=10)
        if not r.ok:
            summary["errors"].append(f"list jobs HTTP {r.status_code}")
            return summary
        data = r.json()
        ray_jobs = data if isinstance(data, list) else data.get("data", [])
    except Exception as e:
        summary["errors"].append(f"list jobs: {e}")
        return summary

    now_ms = time.time() * 1000
    ray_status_by_sub = {}
    for j in ray_jobs:
        sid = j.get("submission_id")
        st  = j.get("status")
        st_ms = j.get("start_time") or 0
        if sid:
            ray_status_by_sub[sid] = (st, st_ms)

    # 1) Stop PENDING jobs that are either:
    #    (a) Older than 5 min and have no active medimage training thread
    #        polling them (DB status = 'running' = thread is alive). These
    #        are orphans the autoscaler can't place.
    #    (b) Owned by a DB row whose status is no longer 'running' or
    #        'queued' (cancelled by user, error, etc). The Ray job lingers
    #        in PENDING forever and holds GPU slots — kill it regardless
    #        of age so the cluster frees up immediately.
    _active_ray_subs = set()
    _terminated_ray_subs = set()
    with get_db() as conn:
        for row in conn.execute(
            "SELECT ray_submission_id, status FROM jobs WHERE ray_submission_id != ''"
        ).fetchall():
            sid_ = row["ray_submission_id"]
            if not sid_:
                continue
            if row["status"] in ("running", "queued"):
                _active_ray_subs.add(sid_)
            else:
                _terminated_ray_subs.add(sid_)

    for j in ray_jobs:
        if j.get("type") != "SUBMISSION":
            continue
        sid  = j.get("submission_id")
        st   = j.get("status")
        st_t = j.get("start_time") or 0
        if st != "PENDING":
            continue
        # A PENDING-in-Ray + running-in-DB job is stale if older than the
        # threshold — the training thread can't possibly be alive because
        # Ray hasn't even started executing the submission yet.
        if sid in _active_ray_subs and (now_ms - st_t) < _RAY_STUCK_PENDING_AGE_S * 1000:
            continue   # young active job — leave alone
        if sid in _ray_stopped_pending:
            continue   # already stopped — don't re-log every 60s
        # (b) DB row already terminated (cancelled/error/etc) — kill
        # immediately. Retry the stop call up to 3 times to defeat the
        # Ray dashboard's intermittent "stop ack" bug for never-allocated
        # jobs (where the job_id is null and the cluster sometimes
        # doesn't reap the submission on the first attempt).
        if sid in _terminated_ray_subs:
            if sid not in _ray_stopped_pending:
                _stop_attempts = 0
                for _ in range(3):
                    if _stop_ray_submission(sid):
                        _stop_attempts += 1
                    import time as _t
                    _t.sleep(0.3)
                _ray_stopped_pending.add(sid)
                summary["stopped_pending"].append(sid)
                _append_log_for_sub(sid, f"[reconcile] Stopped orphaned PENDING job (DB row already terminated, {_stop_attempts}/3 stop acks)")
            continue
        # (a) Stale PENDING with no active thread — only after threshold.
        # Same retry trick to maximise the chance the dashboard reaps
        # the submission.
        if (now_ms - st_t) > _RAY_STUCK_PENDING_AGE_S * 1000:
            _stop_attempts = 0
            for _ in range(3):
                if _stop_ray_submission(sid):
                    _stop_attempts += 1
                import time as _t
                _t.sleep(0.3)
            _ray_stopped_pending.add(sid)
            summary["stopped_pending"].append(sid)
            _append_log_for_sub(sid, f"[reconcile] Stopped stuck PENDING job after {int((now_ms - st_t)/1000)}s (no active training thread, {_stop_attempts}/3 stop acks)")

    # 2) Sync DB for FAILED/STOPPED/SUCCEEDED Ray jobs whose medimage row
    #    is still 'running' or 'queued' (run_training thread crashed
    #    before updating the status).
    finished = ["FAILED", "STOPPED", "SUCCEEDED"]
    for sid, (st, st_ms) in ray_status_by_sub.items():
        if st not in finished:
            continue
        with get_db() as conn:
            row = conn.execute(
                "SELECT id, status, error, started_at, finished_at FROM jobs "
                "WHERE ray_submission_id=? AND status IN ('running','queued') LIMIT 1",
                (sid,),
            ).fetchone()
        if not row:
            continue
        d = dict(row)
        medimage_id = d["id"]
        if st == "SUCCEEDED":
            # Ray says success but DB is still running — odd. Don't claim
            # victory ourselves (the training thread will set weights path
            # + 'completed'). Just log it.
            _append_log(medimage_id, f"[reconcile] Ray job reports SUCCEEDED but DB is still '{d['status']}' — leaving for training thread")
            continue
        # FAILED or STOPPED → mark DB as error.
        # IMPORTANT: set a SHORT message here so the training thread's later
        # _set_status() call overwrites it with the real reason. The previous
        # behavior set a verbose "(reconciled — training thread may have
        # crashed)" message that masked the actual error when the thread
        # eventually did update — confusing for users reading the report.
        # If the training thread is dead (e.g. killed by an API restart),
        # the user will see this short message + can read the full log for
        # the captured stdout.
        # rtdetr-l incident: Ray can report FAILED before the wrapper
        # training thread has had a chance to write its real log lines
        # (log fetch loop returns empty 15/15). Defer until the thread
        # has actually exited, so the captured stdout from the real
        # crash isn't masked by our short "Ray job ended: FAILED" string.
        if sid in _active_ray_subs:
            _append_log(medimage_id, f"[reconcile] Ray reports {st} but training thread still active — deferring error sync")
            continue
        err_msg = f"Ray job ended: {st} (status set by reconciler)"
        _append_log(medimage_id, f"[reconcile] {err_msg} — if the training thread is still alive, it will overwrite this with the real reason; otherwise check the wrapper stdout above")
        _set_status(medimage_id, "error", err_msg)
        summary["synced_errors"].append({"medimage_id": medimage_id, "ray_submission_id": sid, "ray_status": st})

    return summary


def _append_log_for_sub(submission_id: str, line: str):
    """Append a log line to the medimage job that owns this Ray submission."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM jobs WHERE ray_submission_id=? LIMIT 1", (submission_id,),
        ).fetchone()
    if row:
        _append_log(row["id"], line)


def _ray_reconcile_watchdog():
    """Background loop: reconcile Ray cluster state every N seconds."""
    while True:
        try:
            time.sleep(_RAY_RECONCILE_INTERVAL_S)
            _reconcile_ray_jobs()
        except Exception as e:
            print(f"[reconcile] watchdog error: {e}")


# Start the watchdog once when the module loads (API container has a single
# worker process, so we don't worry about multiple instances).
try:
    _t = threading.Thread(target=_ray_reconcile_watchdog, daemon=True, name="ray-reconcile-watchdog")
    _t.start()
    print(f"[reconcile] watchdog started (interval={_RAY_RECONCILE_INTERVAL_S}s, stuck_pending>{_RAY_STUCK_PENDING_AGE_S}s)")
except Exception as _e:
    print(f"[reconcile] failed to start watchdog: {e}")

# ─── Keycloak JWT Auth ────────────────────────────────────────────────────────

_KC_ENABLED = os.getenv("KEYCLOAK_ENABLED", "true").lower() == "true"
_KC_JWKS_URL = os.getenv(
    "KEYCLOAK_JWKS_URL",
    "http://medimage-keycloak-1:8080/realms/h-forge/protocol/openid-connect/certs",
)

_jwks_client = None
if _KC_ENABLED:
    try:
        from jwt import PyJWKClient
        import jwt as _pyjwt
        _jwks_client = PyJWKClient(_KC_JWKS_URL, cache_jwk_set=True, lifespan=300)
    except Exception as _e:
        print(f"[auth] PyJWT not available or JWKS init failed: {_e}")


def _sync_verify_token(token: str):
    signing_key = _jwks_client.get_signing_key_from_jwt(token)
    return _pyjwt.decode(
        token, signing_key.key, algorithms=["RS256"],
        options={"verify_aud": False},
    )


# Paths excluded from JWT auth
_KC_PUBLIC_PATHS = {"/api/health", "/api/diag/minio", "/api/gpu-util"}
_KC_PUBLIC_PREFIXES = ("/api/ls-goto/", "/api/jupyter/", "/api/inference/thumbnail/", "/api/inference/image/", "/api/ray/gpu-stats", "/api/ray/api/", "/api/ray/nodes", "/api/gpu-util")

@app.middleware("http")
async def jwt_auth_middleware(request: Request, call_next):
    if not _KC_ENABLED or _jwks_client is None:
        return await call_next(request)
    # Only protect /api/* paths, but exempt public endpoints
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    if request.url.path in _KC_PUBLIC_PATHS:
        return await call_next(request)
    if any(request.url.path.startswith(p) for p in _KC_PUBLIC_PREFIXES):
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    token = auth[7:]
    try:
        payload = await asyncio.to_thread(_sync_verify_token, token)
    except Exception as exc:
        return JSONResponse({"detail": f"Invalid token: {exc}"}, status_code=401)
    # Stash the user_id (Keycloak 'sub' claim) on request.state so
    # downstream endpoints can scope data per-user (user_prefs, etc).
    request.state.user_id = (payload or {}).get("sub", "") or ""
    return await call_next(request)


def _current_user_id(request: Request) -> str:
    """Return the user_id from the JWT attached to this request.

    Falls back to "default" when Keycloak is disabled (no auth), so the
    user_prefs table still has a stable per-user key.
    """
    uid = getattr(request.state, "user_id", "")
    if uid:
        return uid
    if not _KC_ENABLED:
        return "default"
    return ""


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ─── Keycloak Admin Helpers ───────────────────────────────────────────────────

_KC_BASE = os.getenv("KC_BASE", "http://medimage-keycloak-1:8080/kc")
_KC_REALM = "h-forge"
_KC_MASTER_ADMIN = os.getenv("KEYCLOAK_ADMIN", "admin")
_KC_MASTER_PASS = os.getenv("KEYCLOAK_ADMIN_PASSWORD", "admin")


async def _kc_admin_token() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_KC_BASE}/realms/master/protocol/openid-connect/token",
            data={
                "client_id": "admin-cli",
                "username": _KC_MASTER_ADMIN,
                "password": _KC_MASTER_PASS,
                "grant_type": "password",
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


def _extract_token_payload(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""
    if not token:
        return {}
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return {}


def _require_platform_admin(request: Request) -> dict:
    payload = _extract_token_payload(request)
    roles = payload.get("realm_access", {}).get("roles", [])
    if "platform-admin" not in roles:
        raise HTTPException(status_code=403, detail="Platform admin role required")
    return payload


# ─── User Management (Admin) ──────────────────────────────────────────────────


class UserCreate(BaseModel):
    username: str
    email: str
    firstName: str = ""
    lastName: str = ""
    password: str
    enabled: bool = True


class UserUpdate(BaseModel):
    email: Optional[str] = None
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    enabled: Optional[bool] = None


class ResetPasswordBody(BaseModel):
    password: str


@app.get("/api/admin/users")
async def admin_list_users(request: Request):
    _require_platform_admin(request)
    token = await _kc_admin_token()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users?max=200",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        users = resp.json()
        # Enrich with credential info
        result = []
        for u in users:
            creds_resp = await client.get(
                f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{u['id']}/credentials",
                headers={"Authorization": f"Bearer {token}"},
            )
            creds = creds_resp.json() if creds_resp.is_success else []
            u["has_totp"] = any(c.get("type") == "otp" for c in creds)
            u["totp_credentials"] = [
                {"id": c["id"], "userLabel": c.get("userLabel", "Authenticator"),
                 "createdDate": c.get("createdDate")}
                for c in creds if c.get("type") == "otp"
            ]
            # Get realm roles
            roles_resp = await client.get(
                f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{u['id']}/role-mappings/realm",
                headers={"Authorization": f"Bearer {token}"},
            )
            u["realmRoles"] = [r["name"] for r in (roles_resp.json() if roles_resp.is_success else [])]
            result.append(u)
        return result


@app.post("/api/admin/users", status_code=201)
async def admin_create_user(body: UserCreate, request: Request):
    _require_platform_admin(request)
    token = await _kc_admin_token()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "username": body.username,
                "email": body.email,
                "firstName": body.firstName,
                "lastName": body.lastName,
                "enabled": body.enabled,
                "emailVerified": True,
                "credentials": [{"type": "password", "value": body.password, "temporary": False}],
            },
        )
        if resp.status_code == 409:
            raise HTTPException(409, "Username or email already exists")
        resp.raise_for_status()
        location = resp.headers.get("Location", "")
        user_id = location.rstrip("/").split("/")[-1]
        return {"id": user_id, "username": body.username}


@app.patch("/api/admin/users/{user_id}")
async def admin_update_user(user_id: str, body: UserUpdate, request: Request):
    _require_platform_admin(request)
    token = await _kc_admin_token()
    update_data = {k: v for k, v in body.model_dump().items() if v is not None}
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{user_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=update_data,
        )
        resp.raise_for_status()
    return {"ok": True}


@app.delete("/api/admin/users/{user_id}", status_code=204)
async def admin_delete_user(user_id: str, request: Request):
    payload = _require_platform_admin(request)
    if user_id == payload.get("sub"):
        raise HTTPException(400, "Cannot delete your own account")
    token = await _kc_admin_token()
    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{user_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: str, body: ResetPasswordBody, request: Request):
    _require_platform_admin(request)
    token = await _kc_admin_token()
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{user_id}/reset-password",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"type": "password", "value": body.password, "temporary": False},
        )
        resp.raise_for_status()
    return {"ok": True}


@app.delete("/api/admin/users/{user_id}/totp/{cred_id}", status_code=204)
async def admin_delete_totp(user_id: str, cred_id: str, request: Request):
    _require_platform_admin(request)
    token = await _kc_admin_token()
    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{user_id}/credentials/{cred_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()


# ─── Profile (self-service) ───────────────────────────────────────────────────

@app.get("/api/profile/totp-setup")
async def profile_totp_setup(request: Request):
    """Generate a TOTP secret + QR code for Google Authenticator setup"""
    import pyotp, qrcode, io, base64, time, uuid
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub", "")
    username = payload.get("preferred_username", user_id)

    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    otp_uri = totp.provisioning_uri(name=username, issuer_name="MedImage")

    # Generate QR code PNG as base64
    img = qrcode.make(otp_uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_b64 = base64.b64encode(buf.getvalue()).decode()

    # Store pending secret in DB (expires after 10 min via old cleanup)
    session_token = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO totp_credentials (id, user_id, user_label, secret, status, created_at) VALUES (?,?,?,?,?,?)",
            (session_token, user_id, "pending", secret, "pending", int(time.time()))
        )
        conn.commit()

    return {
        "totpSecret": secret,
        "totpSecretEncoded": secret,
        "totpSecretQrCode": qr_b64,
        "totpSessionToken": session_token,
        "manualUrl": otp_uri,
    }


@app.post("/api/profile/totp-verify")
async def profile_totp_verify(request: Request):
    """Verify a TOTP code and activate the credential"""
    import pyotp, time, uuid
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub", "")
    body = await request.json()

    totp_code = str(body.get("totp", "")).strip()
    secret = str(body.get("totpSecret") or body.get("totpSecretEncoded") or "").strip()
    session_token = str(body.get("totpSessionToken", "")).strip()
    user_label = str(body.get("userLabel", "Authenticator")).strip() or "Authenticator"

    if not totp_code or len(totp_code) < 6:
        raise HTTPException(400, "Enter the 6-digit code from your authenticator app")

    # Look up pending secret — prefer session token, fall back to provided secret
    with get_db() as conn:
        if session_token:
            row = conn.execute(
                "SELECT secret FROM totp_credentials WHERE id=? AND user_id=? AND status='pending'",
                (session_token, user_id)
            ).fetchone()
            if row:
                secret = row["secret"]

        if not secret:
            raise HTTPException(400, "Setup session expired — please start setup again")

        # Verify the code
        totp = pyotp.TOTP(secret)
        if not totp.verify(totp_code, valid_window=1):
            raise HTTPException(400, "Invalid code — check your authenticator and try again")

        # Activate: replace pending record or insert new active one
        new_id = str(uuid.uuid4())
        if session_token:
            conn.execute(
                "UPDATE totp_credentials SET id=?, user_label=?, status='active', created_at=? WHERE id=? AND user_id=?",
                (new_id, user_label, int(time.time() * 1000), session_token, user_id)
            )
        else:
            conn.execute(
                "INSERT INTO totp_credentials (id, user_id, user_label, secret, status, created_at) VALUES (?,?,?,?,?,?)",
                (new_id, user_id, user_label, secret, "active", int(time.time() * 1000))
            )
        # Clean up any other pending entries for this user
        conn.execute(
            "DELETE FROM totp_credentials WHERE user_id=? AND status='pending'",
            (user_id,)
        )
        conn.commit()

    return {"ok": True}


@app.get("/api/profile/totp-credentials")
async def profile_totp_credentials(request: Request):
    """List current user's active TOTP credentials"""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub", "")
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, user_label, created_at FROM totp_credentials WHERE user_id=? AND status='active' ORDER BY created_at",
            (user_id,)
        ).fetchall()
    return [{"id": r["id"], "userLabel": r["user_label"], "createdDate": r["created_at"]} for r in rows]


@app.delete("/api/profile/totp-credentials/{cred_id}", status_code=204)
async def profile_delete_totp(cred_id: str, request: Request):
    """Delete current user's TOTP credential"""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub", "")
    with get_db() as conn:
        result = conn.execute(
            "DELETE FROM totp_credentials WHERE id=? AND user_id=?",
            (cred_id, user_id)
        )
        conn.commit()
    if result.rowcount == 0:
        raise HTTPException(404, "Credential not found")


# ─── Modal Credentials (global, saved in SQLite) ──────────────────────────────

def _load_modal_creds() -> dict | None:
    """Return {token_id, token_secret} from DB, or None if not set."""
    with get_db() as conn:
        row = conn.execute("SELECT token_id, token_secret FROM modal_credentials WHERE id=1").fetchone()
    if row:
        return {"token_id": row["token_id"], "token_secret": row["token_secret"]}
    return None


@app.get("/api/modal/credentials")
def modal_credentials_get():
    with get_db() as conn:
        row = conn.execute(
            "SELECT token_id, updated_at FROM modal_credentials WHERE id=1"
        ).fetchone()
    if not row:
        return {"configured": False}
    return {
        "configured": True,
        "token_id": row["token_id"],
        "updated_at": row["updated_at"],
    }


@app.post("/api/modal/credentials")
async def modal_credentials_save(body: dict):
    token_id = (body.get("token_id") or "").strip()
    token_secret = (body.get("token_secret") or "").strip()
    if not token_id or not token_secret:
        raise HTTPException(400, "token_id and token_secret required")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO modal_credentials (id, token_id, token_secret, updated_at)
               VALUES (1, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(id) DO UPDATE SET
                 token_id=excluded.token_id,
                 token_secret=excluded.token_secret,
                 updated_at=CURRENT_TIMESTAMP""",
            (token_id, token_secret),
        )
        conn.commit()
    return {"ok": True}


@app.delete("/api/modal/credentials")
def modal_credentials_delete():
    with get_db() as conn:
        conn.execute("DELETE FROM modal_credentials WHERE id=1")
        conn.commit()
    return {"ok": True}


@app.post("/api/modal/verify")
def modal_credentials_verify():
    """Test saved Modal credentials by running `modal token inspect`."""
    creds = _load_modal_creds()
    if not creds:
        raise HTTPException(400, "No Modal credentials saved")
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = creds["token_id"]
    env["MODAL_TOKEN_SECRET"] = creds["token_secret"]
    try:
        proc = subprocess.run(
            ["python3", "-m", "modal", "profile", "current"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        ok = proc.returncode == 0
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if not ok and not output:
            output = "Could not connect — check credentials"
        return {"ok": ok, "output": output[:400]}
    except Exception as exc:
        return {"ok": False, "output": str(exc)[:400]}


# ─── HuggingFace Token (per-user) ────────────────────────────────────────────
# Stored per Keycloak user_id. Used to:
#   1) inject HF_TOKEN / HUGGING_FACE_HUB_TOKEN into training env vars
#      so the Ray/Modal worker can pull gated models (e.g. MahmoodLab/UNI)
#   2) authenticate the backend's own calls to huggingface.co (dataset
#      import, model metadata probes) so private repos work too
# The raw token is NEVER returned via GET — only a masked preview + whoami info.


def _load_user_hf_token(user_id: str) -> str:
    """Return the saved HF token for a Keycloak user, or '' if not configured."""
    if not user_id:
        return ""
    with get_db() as conn:
        row = conn.execute(
            "SELECT token FROM huggingface_tokens WHERE user_id=?", (user_id,)
        ).fetchone()
    return (row["token"] if row else "") or ""


def _mask_hf_token(tok: str) -> str:
    """Return a safe preview like 'hf_AbC…xYz' — never the full token."""
    if not tok:
        return ""
    if len(tok) <= 10:
        return "•" * len(tok)
    return f"{tok[:4]}…{tok[-4:]}"


@app.get("/api/profile/huggingface-token")
def profile_hf_token_get(request: Request):
    """Return whether a token is configured for the current user (masked)."""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub", "")
    with get_db() as conn:
        row = conn.execute(
            "SELECT token, hf_username, updated_at FROM huggingface_tokens WHERE user_id=?",
            (user_id,),
        ).fetchone()
    if not row:
        return {"configured": False}
    return {
        "configured": True,
        "mask":       _mask_hf_token(row["token"]),
        "hf_username": row["hf_username"] or "",
        "updated_at": row["updated_at"] or "",
    }


@app.post("/api/profile/huggingface-token")
async def profile_hf_token_save(request: Request):
    """Save or replace the current user's HF token."""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub", "")
    body    = await request.json()
    token   = (body.get("token") or "").strip()
    if not token:
        raise HTTPException(400, "Token is required")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO huggingface_tokens (user_id, token, hf_username, updated_at)
               VALUES (?, ?, '', CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                 token=excluded.token,
                 hf_username='',
                 updated_at=CURRENT_TIMESTAMP""",
            (user_id, token),
        )
        conn.commit()
    return {"ok": True, "mask": _mask_hf_token(token)}


@app.delete("/api/profile/huggingface-token")
def profile_hf_token_delete(request: Request):
    """Remove the current user's HF token."""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub", "")
    with get_db() as conn:
        conn.execute("DELETE FROM huggingface_tokens WHERE user_id=?", (user_id,))
        conn.commit()
    return {"ok": True}


@app.post("/api/profile/huggingface-token/verify")
def profile_hf_token_verify(request: Request):
    """Verify the saved token by calling HF whoami-v2. Caches hf_username."""
    import requests as _req
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub", "")
    token   = _load_user_hf_token(user_id)
    if not token:
        raise HTTPException(400, "No token saved")
    try:
        r = _req.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.status_code == 401:
            return {"ok": False, "error": "Token rejected (401) — invalid or expired"}
        if r.status_code == 403:
            return {"ok": False, "error": "Token forbidden (403) — account suspended?"}
        if not r.ok:
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        info   = r.json()
        user   = info.get("name") or info.get("fullname") or info.get("id") or ""
        orgs   = [o.get("name", "") for o in (info.get("orgs") or []) if o.get("name")]
        # Cache the username for UI display
        with get_db() as conn:
            conn.execute(
                "UPDATE huggingface_tokens SET hf_username=? WHERE user_id=?",
                (user, user_id),
            )
            conn.commit()
        return {
            "ok":          True,
            "hf_username": user,
            "orgs":        orgs,
            "token_type":  info.get("type", "user"),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


@app.get("/api/modal/deployments")
def modal_deployments_list():
    """List all models currently deployed via Modal, enriched with live spec
    from the in-memory deploy state (gpu_type, num_workers, status, logs)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, model_name, modal_url, inference_provider, training_type, "
            "       engine, updated_at "
            "FROM jobs "
            "WHERE inference_provider='modal' AND modal_url != '' "
            "ORDER BY updated_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        state = _modal_deploy_states.get(d["id"], {})
        d["status"]      = state.get("status") or "running"
        d["gpu_type"]    = state.get("gpu_type") or "T4"
        d["num_workers"] = int(state.get("num_workers") or 1)
        d["deployed_at"] = d.pop("updated_at", None)
        d["memory_mb"]          = 16384   # matches @app.cls(... memory=16384) in _modal_model_script
        d["scaledown_window_s"] = 300     # matches scaledown_window=300
        d["min_containers"]     = 1
        out.append(d)
    return {"deployments": out}


class ChangePasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str


@app.post("/api/profile/change-password")
async def profile_change_password(body: ChangePasswordRequest, request: Request):
    """Change current user's password: verifies old password then resets via admin API"""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub")
    username = payload.get("preferred_username", "")

    # Step 1: Verify current password by attempting token exchange
    async with httpx.AsyncClient() as client:
        verify_resp = await client.post(
            f"{_KC_BASE}/realms/{_KC_REALM}/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "h-forge-ui",
                "username": username,
                "password": body.currentPassword,
                "scope": "openid",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if not verify_resp.is_success:
            raise HTTPException(400, "Current password is incorrect")

    # Step 2: Set new password via admin API
    admin_token = await _kc_admin_token()
    async with httpx.AsyncClient() as client:
        reset_resp = await client.put(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{user_id}/reset-password",
            headers={"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"},
            json={"type": "password", "value": body.newPassword, "temporary": False},
        )
        if not reset_resp.is_success:
            raise HTTPException(reset_resp.status_code, "Failed to update password")
    return {"ok": True}



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
    cluster: str = "ray"            # ray | modal
    num_gpus: int = 1               # GPUs per training job (1=sequential, 2-4=data-parallel)


# ─── Training runner ─────────────────────────────────────────────────────────

LS_API_URL = os.getenv("LS_API_URL", "http://label-studio:8080")
LS_TOKEN   = os.getenv("LS_TOKEN", "medimage-ls-token-2026")
MINIO_URL  = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Public URL the Ray cluster (running on a different host) uses to reach
# THIS API. Used by the Ray training wrapper to download the training
# script via HTTP — the script is too big (~97KB source, ~130KB b64) to
# fit in a single env var (Linux ARG_MAX is ~128KB). Default to the same
# host as MinIO since they typically live on the same network.
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", os.getenv("LS_PUBLIC_URL", "http://100.68.3.42:8000").rsplit(":", 1)[0] + ":8000")


def _verify_weights_uploaded(s3_path: str) -> tuple[bool, str]:
    """Confirm an s3://bucket/key path actually has an object in MinIO.

    Ray reporting SUCCEEDED only means the wrapper script exited 0 — it
    does NOT guarantee the training script actually ran end-to-end. The
    common "phantom SUCCEEDED" cases are:
      - main() falls through without matching a (engine, type) branch
      - the engine raises but main() catches it and returns cleanly
      - upload_to_minio() silently no-ops (e.g., no checkpoint to save)

    This check turns a "phantom" success into a real error by verifying
    the expected object exists. Returns (ok, message).
    """
    if not s3_path or not s3_path.startswith("s3://"):
        return False, f"not an s3:// path: {s3_path!r}"
    path = s3_path[5:]
    if "/" not in path:
        return False, f"s3 path missing bucket/key: {s3_path!r}"
    bucket, key = path.split("/", 1)
    try:
        import boto3 as _b3
        from botocore.config import Config as _Cfg
        from botocore.exceptions import ClientError as _CE
        s3 = _b3.client(
            "s3", endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY,
            config=_Cfg(signature_version="s3v4"),
        )
        try:
            resp = s3.head_object(Bucket=bucket, Key=key)
            size = resp.get("ContentLength", 0)
            if size <= 0:
                return False, f"object exists but is empty (0 bytes): s3://{bucket}/{key}"
            return True, f"verified s3://{bucket}/{key} ({size} bytes)"
        except _CE as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False, f"object NOT found in MinIO: s3://{bucket}/{key}"
            return False, f"head_object failed: {e}"
    except Exception as e:
        return False, f"verify failed: {e}"
# Public URL for MinIO accessible from Ray/Modal clusters (set to external/VPN IP).
# Falls back to MINIO_HOST_IP on port 9000 (set in docker-compose) when
# MINIO_PUBLIC_URL isn't explicitly set — the Ray cluster can't resolve
# the docker-internal "minio" hostname.
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", f"http://{os.getenv('MINIO_HOST_IP', '')}:9000" if os.getenv("MINIO_HOST_IP") else MINIO_URL)
# Public URL for Label Studio accessible from Ray cluster
LS_PUBLIC_URL = os.getenv("LS_PUBLIC_URL", LS_API_URL)
# On-prem Ray cluster URL
RAY_URL = os.getenv("RAY_URL", "http://100.68.53.118:8265")
# Explicit host IP the Ray cluster should use to reach MinIO. Overrides
# auto-detection from RAY_URL. Required when the Ray cluster is on a different
# host than docker-compose and MINIO_PUBLIC_URL still points at the docker
# service name (e.g. "http://minio:9000").
MINIO_HOST_IP = os.getenv("MINIO_HOST_IP", "").strip()


def _resolve_minio_url_for_ray() -> str:
    """
    MinIO URL a Ray worker (running outside the docker network) should use.
    Resolution order:
      1. MINIO_PUBLIC_URL — if it does NOT contain the docker service name "minio"
         (i.e. it already points at a real external host).
      2. MINIO_HOST_IP    — explicit override.
      3. Host parsed from RAY_URL with port 9000 (assumes MinIO is on the same
         host as the Ray cluster, which is the common case).
      4. "host.docker.internal" — last-ditch fallback for the API host.
    """
    public = (MINIO_PUBLIC_URL or "").strip()
    if "://" in public:
        try:
            from urllib.parse import urlparse as _urlparse
            host_part = _urlparse(public).hostname or ""
        except Exception:
            host_part = ""
        if host_part and "minio" not in host_part:
            return public
    host = MINIO_HOST_IP
    if not host:
        try:
            from urllib.parse import urlparse as _urlparse
            host = _urlparse(RAY_URL).hostname or ""
        except Exception:
            host = ""
    if not host:
        host = "host.docker.internal"
    return f"http://{host}:9000"


def _ping_host_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if host:port accepts a TCP connection within the timeout."""
    import socket as _socket
    try:
        with _socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _parse_host_port(url: str) -> tuple[str, int]:
    """Parse http(s)://host:port/url and return (host, port)."""
    from urllib.parse import urlparse as _urlparse
    p = _urlparse(url)
    return (p.hostname or "?", p.port or 0)


def _diagnose_minio_for_ray() -> str:
    """
    Check the candidate MinIO host(s) from the worker's point of view and
    return a short human-readable status string for the training logs.
    Used when the dataset download fails, so the user can fix the
    MINIO_HOST_IP / MINIO_PUBLIC_URL env without having to guess.
    """
    import socket as _socket
    from urllib.parse import urlparse as _urlparse
    candidates: list[tuple[str, int]] = []
    for url in (MINIO_PUBLIC_URL, MINIO_URL, _resolve_minio_url_for_ray()):
        try:
            p = _urlparse(url)
            if p.hostname and p.port:
                candidates.append((p.hostname, p.port))
        except Exception:
            pass
    if not candidates:
        return "no MinIO URL configured"
    seen = set()
    lines = []
    for host, port in candidates:
        key = (host, port)
        if key in seen:
            continue
        seen.add(key)
        ok = _ping_host_port(host, port)
        lines.append(f"  {'✓' if ok else '✗'} {host}:{port} {'reachable' if ok else 'UNREACHABLE'}")
    return "\n".join(lines)

# ─── VLM/LLM module shipping (Ray runtime_env) ───────────────────────────────
# vlm_llm_train.py is shipped to workers via Ray's runtime_env["py_modules"].
# Ray requires this to be set at job level (ray.init), not per-task, so we
# resolve the path once at API startup and pass the same runtime_env to every
# ray.init() call — the first one wins (ignore_reinit_error=True on the rest).
# Workers cache the module, so edits to vlm_llm_train.py are picked up on the
# next job without restarting the cluster.
_VLM_LLM_MODULE_PATH: str = ""
try:
    _candidate = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vlm_llm_train.py"
    )
    if os.path.isfile(_candidate):
        _VLM_LLM_MODULE_PATH = _candidate
except Exception:
    pass

_VLM_LLM_RUNTIME_ENV: dict = (
    {
        "py_modules": [_VLM_LLM_MODULE_PATH],
        "env_vars": {
            # The Modal Ray image ships conda's cuda-driver-dev package
            # which puts no-op libcuda.so stubs in
            # /home/ray/anaconda3/lib/stubs/. libtorch_cuda.so has these
            # paths in its DT_RPATH, so LD_LIBRARY_PATH is ignored —
            # torch always loads the stub and cuda.is_available() is
            # False. LD_PRELOAD forces the real driver to be loaded
            # first, ahead of any DT_RPATH lookup, and its symbols
            # occupy the global table before the stub gets a chance.
            "LD_PRELOAD": "/lib/x86_64-linux-gnu/libcuda.so.1",
        },
        # Ray runs this on every worker once, before any task body.
        # It aliases ctypes.RTLD_* onto os.RTLD_* so the baked
        # /app/main.py's `mode=_ct.RTLD_NOW | _ct.RTLD_GLOBAL` works
        # without an image rebuild, and it pre-loads the real
        # libcuda.so.1 to defeat the conda stub. Shipped per-task
        # via py_modules above — main.py is baked into the image
        # and only updates on rebuild.
        "setup_hook": "vlm_llm_train._vlm_setup_hook",
    } if _VLM_LLM_MODULE_PATH else {}
)

# ─── Real Training Helpers ────────────────────────────────────────────────────

def upload_to_minio(local_path, bucket, key, minio_url, minio_access, minio_secret):
    """Upload a local file to MinIO/S3 and return the s3:// URI.
    The bucket is created on demand if it doesn't exist yet.

    NOTE: A *copy* of this function is also defined inside _VISION_TRAIN_SCRIPT
    below — that copy is part of the training script that runs on the Ray
    cluster, and it has its own boto3 import. Don't delete this one without
    also fixing the inline copy (or vice versa).
    """
    import boto3
    from botocore.config import Config
    s3 = boto3.client(
        "s3", endpoint_url=minio_url,
        aws_access_key_id=minio_access, aws_secret_access_key=minio_secret,
        config=Config(signature_version="s3v4"),
    )
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def _convert_ls_json_to_yolo_zip(json_data: list, ls_url: str, ls_token: str,
                                task: str, image_bytes_map: dict | None = None) -> bytes:
    """Convert a Label Studio JSON export into a YOLO-format zip in memory.

    Used by main()'s JSON→YOLO branch in the dataset prep stage. Lives
    at module top level (rather than inside the _VISION_TRAIN_SCRIPT
    raw string) so the API container can call it directly at runtime.

    task='detect'  → cls + cx cy w h (normalised, 5 cols per line)
    task='segment' → cls + polygon points (variable cols; rectanglelabels
                     get expanded to a 4-corner polygon since LS
                     rectanglelabels don't carry their own polygon
                     vertices — that's enough for YOLO-seg's
                     assign_targets stage which only needs the bbox
                     anyway and won't crash on degenerate "boxes" that
                     span the full image).
    """
    import base64 as _b64
    import requests as _req
    from PIL import Image as _Image
    import io as _io
    import tempfile as _tmpf
    import zipfile as _zf
    ls_url = ls_url.rstrip("/")
    tmp = _tmpf.mkdtemp(prefix="yolozip_")
    try:
        images_dir = os.path.join(tmp, "images")
        labels_dir = os.path.join(tmp, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        class_map: dict[str, int] = {}
        for task_item in json_data:
            image_url = task_item.get("data", {}).get("image", "")
            if not image_url:
                continue
            task_id = task_item.get("id", "unknown")
            fname = f"task_{task_id}.jpg"
            img_path = os.path.join(images_dir, fname)
            try:
                if image_bytes_map and fname in image_bytes_map:
                    img_bytes = image_bytes_map[fname]
                elif image_url.startswith("data:"):
                    _, data = image_url.split(",", 1)
                    img_bytes = _b64.b64decode(data)
                else:
                    url = re.sub(r"https?://[^/]+", ls_url, image_url)
                    r = _req.get(url, headers={"Authorization": f"Token {ls_token}"}, timeout=60)
                    r.raise_for_status()
                    img_bytes = r.content
                _Image.open(_io.BytesIO(img_bytes)).convert("RGB").save(img_path, "JPEG")
            except Exception as e:
                print(f"[warn] Cannot fetch image task {task_id}: {e}")
                continue
            yolo_lines: list[str] = []
            for ann in task_item.get("annotations", []):
                for res in ann.get("result", []):
                    val = res.get("value", {})
                    res_type = res.get("type", "")
                    if res_type == "polygonlabels" or "polygonlabels" in val:
                        labels_list = val.get("polygonlabels", [])
                        if not labels_list:
                            continue
                        label = labels_list[0]
                        if label not in class_map:
                            class_map[label] = len(class_map)
                        cls_id = class_map[label]
                        points = val.get("points", [])
                        if len(points) < 3:
                            continue
                        norm_pts = " ".join(f"{p[0]/100:.6f} {p[1]/100:.6f}" for p in points)
                        yolo_lines.append(f"{cls_id} {norm_pts}")
                    elif res_type == "rectanglelabels" or "rectanglelabels" in val:
                        labels_list = val.get("rectanglelabels", [])
                        if not labels_list:
                            continue
                        label = labels_list[0]
                        if label not in class_map:
                            class_map[label] = len(class_map)
                        cls_id = class_map[label]
                        x_pct = float(val.get("x", 0))
                        y_pct = float(val.get("y", 0))
                        w_pct = float(val.get("width", 0))
                        h_pct = float(val.get("height", 0))
                        if task == "segment":
                            # bbox → 4-corner polygon (YOLO-seg format)
                            x1, y1 = x_pct / 100.0, y_pct / 100.0
                            x2, y2 = (x_pct + w_pct) / 100.0, y1
                            x3, y3 = x2, (y_pct + h_pct) / 100.0
                            x4, y4 = x1, y3
                            yolo_lines.append(
                                f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                                f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                            )
                        else:
                            # YOLO detect: cx cy w h
                            cx = (x_pct + w_pct / 2) / 100.0
                            cy = (y_pct + h_pct / 2) / 100.0
                            nw = w_pct / 100.0
                            nh = h_pct / 100.0
                            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            if yolo_lines:
                label_path = os.path.join(labels_dir, fname.replace(".jpg", ".txt"))
                with open(label_path, "w") as lf:
                    lf.write("\n".join(yolo_lines) + "\n")
        if not class_map:
            raise RuntimeError("No labelled bounding boxes found in JSON export")
        names_list = [k for k, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
        yaml_path = os.path.join(tmp, "dataset.yaml")
        with open(yaml_path, "w") as yf:
            yf.write(f"path: .\n")
            yf.write(f"train: images\n")
            yf.write(f"val: images\n")
            yf.write(f"nc: {len(names_list)}\n")
            yf.write(f"names: {json.dumps(names_list)}\n")
        buf = _io.BytesIO()
        with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(tmp):
                for fn in files:
                    full = os.path.join(root, fn)
                    arc = os.path.relpath(full, tmp)
                    zf.write(full, arc)
        return buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# Vision training script (runs on Ray cluster — supports Ultralytics YOLO and PyTorch/TIMM)
_VISION_TRAIN_SCRIPT = r'''
import os, sys, re, json, zipfile, tempfile, shutil, random
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def upload_to_minio(local_path, bucket, key, minio_url, minio_access, minio_secret):
    import boto3
    from botocore.config import Config
    s3 = boto3.client(
        "s3", endpoint_url=minio_url,
        aws_access_key_id=minio_access, aws_secret_access_key=minio_secret,
        config=Config(signature_version="s3v4"),
    )
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
    s3.upload_file(str(local_path), bucket, key)
    print(f"WEIGHTS_UPLOADED: s3://{bucket}/{key}")


def download_images_from_ls(json_data, out_dir, ls_url, ls_token):
    """Parse LS JSON export, download images organised by class label."""
    import base64 as _b64
    import requests as _req
    ls_url = ls_url.rstrip("/")
    downloaded = 0
    for task in json_data:
        image_url = task.get("data", {}).get("image", "")
        if not image_url:
            continue
        # Collect all labels for this task (first annotation, all results)
        labels = set()
        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                val = res.get("value", {})
                for lt in ("rectanglelabels", "polygonlabels", "ellipselabels", "choices"):
                    for lbl in val.get(lt, []):
                        labels.add(lbl)
        if not labels:
            continue
        # Use the first label as the classification target
        label = sorted(labels)[0]
        cls_dir = os.path.join(out_dir, re.sub(r"[^\w\-]", "_", label))
        os.makedirs(cls_dir, exist_ok=True)
        fname = f"task_{task['id']}.jpg"
        img_path = os.path.join(cls_dir, fname)
        try:
            if image_url.startswith("data:"):
                # Inline base64 data URI
                header, data = image_url.split(",", 1)
                img_bytes = _b64.b64decode(data)
            else:
                # HTTP(S) URL — rewrite internal hostname to public URL
                image_url = re.sub(r"https?://[^/]+", ls_url, image_url)
                r = _req.get(image_url, headers={"Authorization": f"Token {ls_token}"}, timeout=60)
                r.raise_for_status()
                img_bytes = r.content
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            downloaded += 1
        except Exception as e:
            print(f"[warn] Cannot fetch image for task {task.get('id')}: {e}")
    return downloaded


def prepare_yolo_dataset_from_json(json_data, out_dir, ls_url, ls_token, task="detect",
                                  image_bytes_map=None):
    """Convert Label Studio JSON export → YOLO detection or segmentation format.

    task='detect'  → cls + cx cy w h (normalised, 5 cols per line)
    task='segment' → cls + polygon points (variable cols; rectanglelabels get
                     converted to a 4-corner polygon since LS rectanglelabels
                     don't carry their own polygon vertices).

    image_bytes_map (optional): dict[str, bytes] mapping filename → raw image
    bytes. When provided, the function uses these instead of fetching via
    ls_url/ls_token. Used by the inline JSON→YOLO zip conversion in
    main() which already has the data-URI images decoded in memory and
    needs to avoid a second round-trip through the Label Studio server.

    Creates out_dir/images/ and out_dir/labels/, then writes dataset.yaml.
    Returns path to dataset.yaml.
    """
    import base64 as _b64
    import requests as _req
    from PIL import Image as _Image
    import io as _io
    ls_url = ls_url.rstrip("/")
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    class_map = {}  # label_str -> int id

    for task_item in json_data:
        image_url = task_item.get("data", {}).get("image", "")
        if not image_url:
            continue
        task_id = task_item.get("id", "unknown")
        fname = f"task_{task_id}.jpg"
        img_path = os.path.join(images_dir, fname)

        # Download / decode image
        try:
            if fname in (image_bytes_map or {}):
                # Caller already decoded the data URI / fetched the URL.
                img_bytes = image_bytes_map[fname]
            elif image_url.startswith("data:"):
                _, data = image_url.split(",", 1)
                img_bytes = _b64.b64decode(data)
            else:
                url = re.sub(r"https?://[^/]+", ls_url, image_url)
                r = _req.get(url, headers={"Authorization": f"Token {ls_token}"}, timeout=60)
                r.raise_for_status()
                img_bytes = r.content
            # Get image dimensions for fallback
            pil_img = _Image.open(_io.BytesIO(img_bytes)).convert("RGB")
            img_w, img_h = pil_img.size
            pil_img.save(img_path, "JPEG")
        except Exception as e:
            print(f"[warn] Cannot fetch image task {task_id}: {e}")
            continue

        # Collect annotations
        yolo_lines = []
        for ann in task_item.get("annotations", []):
            for res in ann.get("result", []):
                val = res.get("value", {})
                res_type = res.get("type", "")

                # ── polygonlabels (actual polygon) ─────────────────────────
                if res_type == "polygonlabels" or "polygonlabels" in val:
                    labels_list = val.get("polygonlabels", [])
                    if not labels_list:
                        continue
                    label = labels_list[0]
                    if label not in class_map:
                        class_map[label] = len(class_map)
                    cls_id = class_map[label]
                    points = val.get("points", [])
                    if len(points) < 3:
                        continue
                    # LS points are % of image size
                    norm_pts = " ".join(f"{p[0]/100:.6f} {p[1]/100:.6f}" for p in points)
                    yolo_lines.append(f"{cls_id} {norm_pts}")

                # ── rectanglelabels (bbox) ─────────────────────────────────
                elif res_type == "rectanglelabels" or "rectanglelabels" in val:
                    labels_list = val.get("rectanglelabels", [])
                    if not labels_list:
                        continue
                    label = labels_list[0]
                    if label not in class_map:
                        class_map[label] = len(class_map)
                    cls_id = class_map[label]
                    x_pct = float(val.get("x", 0))
                    y_pct = float(val.get("y", 0))
                    w_pct = float(val.get("width", 0))
                    h_pct = float(val.get("height", 0))

                    if task == "segment":
                        # Convert bbox → 4-corner polygon for YOLO-seg
                        x1 = x_pct / 100.0;  y1 = y_pct / 100.0
                        x2 = (x_pct + w_pct) / 100.0; y2 = y1
                        x3 = x2;             y3 = (y_pct + h_pct) / 100.0
                        x4 = x1;             y4 = y3
                        yolo_lines.append(
                            f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                            f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                        )
                    else:
                        # YOLO detect: cx cy w h normalised
                        cx = (x_pct + w_pct / 2) / 100.0
                        cy = (y_pct + h_pct / 2) / 100.0
                        nw = w_pct / 100.0
                        nh = h_pct / 100.0
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if yolo_lines:
            label_path = os.path.join(labels_dir, fname.replace(".jpg", ".txt"))
            with open(label_path, "w") as lf:
                lf.write("\n".join(yolo_lines) + "\n")

    if not class_map:
        raise RuntimeError("No labelled bounding boxes found in JSON export")

    names_list = [k for k, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    print(f"[train] YOLO dataset ({task}): {len(json_data)} tasks, classes={names_list}")

    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as yf:
        yf.write(f"path: {out_dir}\n")
        yf.write(f"train: images\n")
        yf.write(f"val: images\n")
        yf.write(f"nc: {len(names_list)}\n")
        yf.write(f"names: {json.dumps(names_list)}\n")
    return yaml_path


def prepare_segmentation_masks_from_json(json_data, out_dir, ls_url, ls_token):
    """Convert LS JSON (polygonlabels) → image+mask pairs for segmentation training."""
    import base64 as _b64
    import requests as _req
    from PIL import Image as _Image, ImageDraw as _Draw
    import io as _io
    ls_url = ls_url.rstrip("/")
    images_dir = os.path.join(out_dir, "images")
    masks_dir  = os.path.join(out_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)
    class_map = {}  # label -> class id (1-based; 0 = background)
    pairs = []
    for task in json_data:
        image_url = task.get("data", {}).get("image", "")
        if not image_url: continue
        task_id = task.get("id", "unknown")
        try:
            if image_url.startswith("data:"):
                _, data = image_url.split(",", 1)
                img_bytes = _b64.b64decode(data)
            else:
                url = re.sub(r"https?://[^/]+", ls_url, image_url)
                r = _req.get(url, headers={"Authorization": f"Token {ls_token}"}, timeout=60)
                r.raise_for_status()
                img_bytes = r.content
            pil_img = _Image.open(_io.BytesIO(img_bytes)).convert("RGB")
            img_w, img_h = pil_img.size
        except Exception as e:
            print(f"[warn] Cannot fetch image task {task_id}: {e}")
            continue
        mask = _Image.new("L", (img_w, img_h), 0)
        draw = _Draw.Draw(mask)
        has_mask = False
        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                val   = res.get("value", {})
                pts   = val.get("points", [])
                lbls  = val.get("polygonlabels", [])
                if not lbls or not pts: continue
                label = lbls[0]
                if label not in class_map:
                    class_map[label] = len(class_map) + 1
                cls_id = class_map[label]
                pixel_pts = [(p[0] * img_w / 100.0, p[1] * img_h / 100.0) for p in pts]
                draw.polygon(pixel_pts, fill=cls_id)
                has_mask = True
        if not has_mask:
            print(f"[warn] Task {task_id}: no polygon annotations, skipping")
            continue
        fname = f"task_{task_id}"
        img_path  = os.path.join(images_dir, fname + ".jpg")
        mask_path = os.path.join(masks_dir,  fname + ".png")
        pil_img.save(img_path, "JPEG")
        mask.save(mask_path, "PNG")
        pairs.append((img_path, mask_path))
    if not pairs:
        raise RuntimeError(
            "No polygon mask annotations found. "
            "Segmentation training requires polygon labels in Label Studio. "
            "Use Detection training type for bounding box annotations."
        )
    names_list = [k for k, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    print(f"[train] Segmentation: {len(pairs)} image-mask pairs, classes={names_list}")
    return pairs, names_list


def train_pytorch_segmentation(pairs, class_names, model_name, epochs, imgsz, batch, lr, optimizer,
                               job_id, w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Train UNet / UNet++ / DeepLabV3+ using segmentation_models_pytorch."""
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image as _Image
    import numpy as _np
    nc     = len(class_names) + 1  # +1 for background (masks use 1-based class ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}, classes: {nc}, model: {model_name}")
    # Parse model_name: can be "encoder-arch" (e.g. "resnet34-unet")
    # OR just an arch name (e.g. "Unet", "unetplusplus", "deeplabv3+") → use default encoder
    _ARCH_KEYWORDS = ("unet", "deeplabv3", "fpn", "pspnet", "pan", "manet", "linknet")
    mn_lower = model_name.lower().replace("+", "plus").replace(" ", "")
    _is_arch_only = any(mn_lower.startswith(k) or mn_lower.endswith(k) for k in _ARCH_KEYWORDS)
    if _is_arch_only:
        encoder = "resnet34"
        arch    = mn_lower
    else:
        parts   = model_name.split("-", 1)
        encoder = parts[0] if parts else "resnet34"
        arch    = parts[1].lower() if len(parts) > 1 else "unet"
    if "unetplusplus" in arch or "unet++" in arch:
        model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=nc)
    elif "deeplabv3" in arch:
        model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=nc)
    else:
        model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=nc)
    model = model.to(device)
    tf_img = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_msk = transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.NEAREST)
    class SegDS(Dataset):
        def __init__(self, p): self.p = p
        def __len__(self): return len(self.p)
        def __getitem__(self, i):
            img = _Image.open(self.p[i][0]).convert("RGB")
            msk = _Image.open(self.p[i][1])
            return tf_img(img), torch.from_numpy(_np.array(tf_msk(msk))).long()
    n       = len(pairs)
    n_val   = max(1, int(n * 0.2))
    random.shuffle(pairs)
    train_dl = DataLoader(SegDS(pairs[n_val:]),  batch_size=batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(SegDS(pairs[:n_val]),  batch_size=batch, shuffle=False, num_workers=2)
    ce_fn   = nn.CrossEntropyLoss()
    dice_fn = smp.losses.DiceLoss(mode="multiclass")
    opt_map = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    opt     = opt_map.get(optimizer.lower(), torch.optim.AdamW)(model.parameters(), lr=lr)
    best_dice = 0.0
    best_path = f"/tmp/best_{job_id}.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for imgs, msks in train_dl:
            imgs, msks = imgs.to(device), msks.to(device)
            preds = model(imgs)
            loss = 0.5 * ce_fn(preds, msks) + 0.5 * dice_fn(preds, msks)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item()
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for imgs, msks in val_dl:
                imgs, msks = imgs.to(device), msks.to(device)
                pred_cls = model(imgs).argmax(1)
                for c in range(nc):
                    tp = ((pred_cls == c) & (msks == c)).sum().item()
                    fp = ((pred_cls == c) & (msks != c)).sum().item()
                    fn = ((pred_cls != c) & (msks == c)).sum().item()
                    if tp + fp + fn > 0:
                        dice_scores.append(2 * tp / (2 * tp + fp + fn))
        val_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
        print(f"[train] Epoch {epoch}/{epochs} — loss={t_loss/len(train_dl):.4f}  val_dice={val_dice:.4f}")
        if val_dice >= best_dice:
            best_dice = val_dice
            torch.save({"epoch": epoch, "class_names": class_names, "model_state_dict": model.state_dict()}, best_path)
    print(f"[train] Best val_dice: {best_dice:.4f}")
    upload_to_minio(best_path, w_bucket, w_key, minio_url, minio_access, minio_secret)


def train_pytorch_classification(img_dir, model_name, epochs, imgsz, batch, lr, optimizer, job_id, w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Train a TIMM classification model."""
    import torch
    import torch.nn as nn
    import timm
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    classes = sorted([
        d for d in os.listdir(img_dir)
        if os.path.isdir(os.path.join(img_dir, d))
        and any(f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(os.path.join(img_dir, d)))
    ])
    if not classes:
        raise RuntimeError(f"No image class folders found in {img_dir}")
    nc = len(classes)
    print(f"[train] Classes ({nc}): {classes}")

    class ImgDS(Dataset):
        def __init__(self, samples, tf):
            self.samples, self.tf = samples, tf
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            p, l = self.samples[i]
            return self.tf(Image.open(p).convert("RGB")), l

    samples = []
    for ci, cls in enumerate(classes):
        for f in os.listdir(os.path.join(img_dir, cls)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(img_dir, cls, f), ci))
    if not samples:
        raise RuntimeError("No image files found after organising by class")

    random.shuffle(samples)
    n_val = max(1, int(0.2 * len(samples)))
    # Some models (e.g. efficientvit_m5) have a FIXED input size — they
    # will hard-fail with feature-map mismatches if we feed the wrong
    # resolution. Always honour the model's default in that case.
    timm_name = model_name.replace("-", "_")
    # ── Model name alias map ──────────────────────────────────────────────────
    # The catalog uses torchvision-style names for some models (e.g.
    # `regnet_y_400mf`), but the rest of this function talks to TIMM,
    # which uses different names (e.g. `regnety_004`). Without this map
    # TIMM raises "Unknown model" and the user sees a confusing error
    # with no hint about the right name to use. Keys are catalog names;
    # values are the TIMM equivalent.
    _TIMM_ALIAS = {
        "regnet_y_400mf":   "regnety_004",
        "regnet_y_800mf":   "regnety_008",
        "regnet_y_1_6gf":   "regnety_016",
        "regnet_y_3_2gf":   "regnety_032",
        "regnet_y_8gf":     "regnety_080",
        "regnet_y_16gf":    "regnety_160",
        "regnet_y_32gf":    "regnety_320",
        "regnet_z_500mf":   "regnetz_005",
        # EfficientNet — TIMM ships the original TF-pretrained weights under
        # `tf_efficientnet_b{0..7}`; the plain `efficientnet_b4` only has
        # noisy-students weights. We default to the TF version (the
        # smaller, faster, better-known checkpoint).
        "efficientnet-b0":  "tf_efficientnet_b0",
        "efficientnet-b1":  "tf_efficientnet_b1",
        "efficientnet-b2":  "tf_efficientnet_b2",
        "efficientnet-b3":  "tf_efficientnet_b3",
        "efficientnet-b4":  "tf_efficientnet_b4",
        "efficientnet-b5":  "tf_efficientnet_b5",
        "efficientnet-b6":  "tf_efficientnet_b6",
        "efficientnet-b7":  "tf_efficientnet_b7",
    }
    if timm_name in _TIMM_ALIAS:
        _alias_target = _TIMM_ALIAS[timm_name]
        print(f"[train] aliasing {timm_name} → {_alias_target} for TIMM", flush=True)
        timm_name = _alias_target
    _model_default_input = None
    _model_fixed = False
    try:
        import timm as _timm
        _cfg = _timm.get_pretrained_cfg(timm_name)
        if _cfg is not None:
            _in = getattr(_cfg, "input_size", None)
            if isinstance(_in, (list, tuple)) and len(_in) >= 3:
                _model_default_input = _in[-1]  # H or W
            _model_fixed = bool(getattr(_cfg, "fixed_input_size", False))
    except Exception:
        pass
    if _model_fixed and _model_default_input:
        if imgsz != _model_default_input:
            print(f"[train] {timm_name} requires input_size={_model_default_input} (fixed) — overriding imgsz={imgsz}")
            imgsz = _model_default_input
    elif _model_default_input and abs(_model_default_input - imgsz) > 32:
        print(f"[train] imgsz={imgsz} doesn't match {timm_name} default {_model_default_input} — using {_model_default_input}")
        imgsz = _model_default_input
    tf_tr = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_vl = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dl = DataLoader(ImgDS(samples[:-n_val], tf_tr), batch_size=batch, shuffle=True, drop_last=True)
    val_dl   = DataLoader(ImgDS(samples[-n_val:], tf_vl), batch_size=batch, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device} | model: {timm_name} | imgsz={imgsz}")
    # ── torchxrayvision branch ──────────────────────────────────────────────
    # `densenet121-res224-all` and other TorchXRayVision medical models are
    # NOT in the TIMM registry — they're a separate library with a different
    # loader API (xrv.models.DenseNet(weights="densenet121-res224-all")).
    # TIMM's create_model() raises a confusing "not a valid model" error if
    # we route them through here, so detect and dispatch instead.
    # Note: xrv's weights param uses the original dashed name, not the
    # underscored `timm_name` we computed above.
    _is_xrv_model = model_name.lower().startswith("densenet") and "res224" in model_name.lower()
    if _is_xrv_model:
        import torchxrayvision as xrv
        # xrv expects a single-channel 224x224 input with specific mean/std
        # normalization. We override the user's imgsz since the weights are
        # trained on 224. Also the model outputs (1, num_classes_xrv) for the
        # 18 X-ray pathologies, not the user's label count — we wrap it with
        # a fresh linear head to map to our `nc` classes.
        if imgsz != 224:
            print(f"[train] {timm_name} is fixed at 224x224 (xrv weights) — overriding imgsz={imgsz}")
            imgsz = 224
        xrv_model = xrv.models.DenseNet(weights=model_name)
        # The xrv DenseNet.forward() runs an internal `op_norm()` step that
        # multiplies the features by a buffer sized to the model's
        # pathology count (18 for densenet121-res224-all). Simply swapping
        # `self.classifier` to a different output size makes op_norm
        # broadcast a (1, 18) tensor against a (B, nc) feature map → crash
        # with "tensor a (3) must match tensor b (18)". The clean fix is
        # to bypass xrv's full forward() entirely and use only its feature
        # extractor (the conv stack) + our own linear classifier.
        class _XRVEncoderClassifier(nn.Module):
            def __init__(self, xrv_base: nn.Module, num_classes: int):
                super().__init__()
                # Pull out the conv backbone (everything up to the
                # adaptive pool + flatten) and discard the pathology head.
                self.features = xrv_base.features
                feat_dim = self._infer_feature_dim()
                self.classifier = nn.Linear(feat_dim, num_classes)
            def _infer_feature_dim(self) -> int:
                # xrv.features ends in adaptive_avg_pool2d((7,7)) NOT (1,1),
                # so the output is (1, C, 7, 7). We must run the pool
                # ourselves to get the per-image feature size (C), not
                # C*7*7 = 50176.
                with torch.no_grad():
                    self.features.eval()
                    dummy = torch.zeros(1, 1, imgsz, imgsz)
                    out = self.features(dummy)
                    if out.dim() > 2:
                        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
                    return int(out.flatten(1).shape[-1])
            def forward(self, x):
                f = self.features(x)
                if f.dim() > 2:
                    f = nn.functional.adaptive_avg_pool2d(f, (1, 1)).view(f.size(0), -1)
                return self.classifier(f)
        model = _XRVEncoderClassifier(xrv_model, nc).to(device)
        # Use a single-channel transform (xrv default) instead of 3-channel
        tf_tr_xrv = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # xrv's default mean/std
        ])
        tf_vl_xrv = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_dl = DataLoader(ImgDS(samples[:-n_val], tf_tr_xrv), batch_size=batch, shuffle=True, drop_last=True)
        val_dl   = DataLoader(ImgDS(samples[-n_val:], tf_vl_xrv), batch_size=batch, drop_last=False)
    elif (
        # ── DINOv2-style encoder models ─────────────────────────────────────
        # DINOv2 (and HF models that use DINOv2 backbones — MahmoodLab/UNI,
        # microsoft/rad-dino) ship as VISION ENCODERS: they output
        # (batch, embed_dim) features, NOT (batch, num_classes) logits.
        # `timm.create_model("vit_base_patch14_dinov2", num_classes=nc)`
        # works for downstream linear-probe because TIMM has a registered
        # classification head, BUT it ignores the self-supervised
        # pretraining (uses the head, not the encoder features). For
        # linear-probe the cleanest approach is:
        #   1. Load the model with num_classes=0 → no head, just features
        #   2. Pool the output (DINOv2 ViT-B/14 uses [CLS]-style tokens)
        #   3. Add a fresh nn.Linear(embed_dim, num_classes)
        # This works the same for HF models that share the DINOv2
        # architecture (rad-dino, UNI) when called via timm.create_model
        # with the hf-hub:org/repo prefix stripped.
        (timm_name.startswith("vit_base_patch14_dinov2")
         or timm_name == "vit_small_patch14_dinov2"
         or timm_name in ("vit_small_patch14_reg4_dinov2",
                          "vit_base_patch14_reg4_dinov2"))
    ):
        print(f"[train] loading {timm_name} as DINOv2-style encoder (no head) + custom classifier")
        # num_classes=0 makes timm drop the classification head and
        # return the pooled feature vector.
        encoder = timm.create_model(timm_name, pretrained=True, num_classes=0)
        embed_dim = encoder.num_features
        class _DinoV2LinearProbe(nn.Module):
            def __init__(self, enc: nn.Module, embed_dim: int, num_classes: int):
                super().__init__()
                self.encoder = enc
                self.classifier = nn.Linear(embed_dim, num_classes)
            def forward(self, x):
                feats = self.encoder(x)
                return self.classifier(feats)
        model = _DinoV2LinearProbe(encoder, embed_dim, nc).to(device)
    else:
        model = timm.create_model(timm_name, pretrained=True, num_classes=nc).to(device)

    crit = nn.CrossEntropyLoss()
    opt_map = {
        "adamw": torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2),
        "adam":  torch.optim.Adam(model.parameters(), lr=lr),
        "sgd":   torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4),
    }
    opt = opt_map.get(optimizer.lower(), opt_map["adamw"])

    best_acc, best_path = -1.0, f"/tmp/best_{job_id}.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), lbls)
            loss.backward(); opt.step()
            total_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                _, pred = model(imgs).max(1)
                correct += (pred == lbls).sum().item()
                total += lbls.size(0)
        val_acc = correct / max(total, 1)
        print(f"[train] Epoch {epoch}/{epochs}: loss={total_loss/max(len(train_dl),1):.4f}  val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch, "model_name": model_name, "num_classes": nc,
                "classes": classes, "val_acc": val_acc,
                "model_state_dict": model.state_dict(),
            }, best_path)
    print(f"[train] Best val_acc: {best_acc:.4f}")
    upload_to_minio(best_path, w_bucket, w_key, minio_url, minio_access, minio_secret)


def train_monai(data_dir, model_name, training_type, epochs, imgsz, batch, lr,
                job_id, w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Fine-tune MONAI models for medical image classification or segmentation."""
    import zipfile as _zf, tempfile as _tmp
    import torch
    from pathlib import Path

    print(f"[monai] Starting MONAI training — model={model_name} type={training_type}")

    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "monai>=1.3", "nibabel", "boto3", "Pillow"], capture_output=True)
    import monai  # noqa
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
        Resized, RandFlipd, RandRotate90d, ToTensord,
    )
    from monai.networks.nets import DenseNet121, UNet
    from monai.data import Dataset, DataLoader
    from monai.losses import DiceLoss

    img_extensions = (".jpg", ".jpeg", ".png", ".nii", ".nii.gz", ".dcm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[monai] Using device: {device}")

    if training_type == "classification":
        # ── Classification — expect class subdirectories ──────────────────────
        class_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
        if not class_dirs:
            raise RuntimeError("No class subdirectories found. Expected: data/class_name/image.png")
        class_names = [d.name for d in class_dirs]
        print(f"[monai] Classes: {class_names}")

        samples = []
        for label_idx, d in enumerate(class_dirs):
            for img_path in d.iterdir():
                if img_path.suffix.lower() in img_extensions:
                    samples.append({"image": str(img_path), "label": label_idx})

        if not samples:
            raise RuntimeError("No images found in class subdirectories")

        train_transforms = Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(imgsz, imgsz)),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image"], prob=0.5),
            ToTensord(keys=["image"]),
        ])

        val_size = max(1, int(len(samples) * 0.1))
        train_ds = Dataset(data=samples[val_size:], transform=train_transforms)
        val_ds   = Dataset(data=samples[:val_size],  transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)

        # Detect input channels from first sample
        first = train_ds[0]["image"]
        in_ch = first.shape[0] if hasattr(first, "shape") else 1
        print(f"[monai] Detected in_channels={in_ch}")

        model = DenseNet121(spatial_dims=2, in_channels=in_ch, out_channels=len(class_names)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_data in train_loader:
                imgs   = batch_data["image"].to(device)
                labels = torch.tensor(batch_data["label"], dtype=torch.long).to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"[monai] Epoch {epoch+1}/{epochs} loss={epoch_loss/max(1,len(train_loader)):.4f}")
            print("PIPELINE_STEP:training")

        tmpdir2 = _tmp.mkdtemp()
        weights_path = os.path.join(tmpdir2, "monai_cls_weights.pt")
        torch.save({"model_state": model.state_dict(), "classes": class_names}, weights_path)

    else:
        # ── Segmentation — expect images/ and masks/ subdirs ──────────────────
        images_dir = os.path.join(data_dir, "images")
        masks_dir  = os.path.join(data_dir, "masks")
        if not os.path.isdir(images_dir):
            raise RuntimeError("MONAI segmentation expects data/images/ and data/masks/ directories")

        img_files = sorted([f for f in Path(images_dir).iterdir()
                             if f.suffix.lower() in img_extensions])
        msk_files = sorted([f for f in Path(masks_dir).iterdir()
                             if f.suffix.lower() in img_extensions]) if os.path.isdir(masks_dir) else []
        if not img_files:
            raise RuntimeError("No image files in images/ directory")

        samples = [{"image": str(i), "mask": str(m)}
                   for i, m in zip(img_files, msk_files)] if msk_files else \
                  [{"image": str(i), "mask": str(i)} for i in img_files]  # fallback: use image as mask

        seg_transforms = Compose([
            LoadImaged(keys=["image", "mask"], image_only=True),
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image", "mask"], spatial_size=(imgsz, imgsz)),
            ToTensord(keys=["image", "mask"]),
        ])
        train_ds     = Dataset(data=samples, transform=seg_transforms)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

        # Detect input channels from first sample
        first_seg = train_ds[0]["image"]
        in_ch_seg = first_seg.shape[0] if hasattr(first_seg, "shape") else 1
        print(f"[monai] Detected in_channels={in_ch_seg}")

        model = UNet(
            spatial_dims=2, in_channels=in_ch_seg, out_channels=2,
            channels=(16, 32, 64, 128), strides=(2, 2, 2),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_data in train_loader:
                imgs  = batch_data["image"].to(device)
                masks = batch_data["mask"].to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = loss_fn(out, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"[monai] Epoch {epoch+1}/{epochs} loss={epoch_loss/max(1,len(train_loader)):.4f}")
            print("PIPELINE_STEP:training")

        tmpdir2 = _tmp.mkdtemp()
        weights_path = os.path.join(tmpdir2, "monai_seg_weights.pt")
        torch.save({"model_state": model.state_dict()}, weights_path)

    # ── Upload ──────────────────────────────────────────────────────────────────
    print("PIPELINE_STEP:saving")
    zip_path = weights_path.replace(".pt", ".zip")
    with _zf.ZipFile(zip_path, "w") as z:
        z.write(weights_path, os.path.basename(weights_path))
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)
    print(f"WEIGHTS_UPLOADED: {w_bucket}/{w_key}")
    print("[monai] Done!")


def train_medsam(data_dir, model_name, epochs, imgsz, batch, lr,
                 job_id, w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Fine-tune MedSAM (SAM adapted for medical image segmentation)."""
    import zipfile as _zf, tempfile as _tmp
    import torch
    from pathlib import Path

    print(f"[medsam] Starting MedSAM fine-tuning — model={model_name}")

    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segment-anything", "monai>=1.3", "nibabel", "boto3", "Pillow",
                    "git+https://github.com/bowang-lab/MedSAM.git"], capture_output=True)

    img_extensions = (".jpg", ".jpeg", ".png", ".nii", ".nii.gz")
    images_dir = os.path.join(data_dir, "images")
    masks_dir  = os.path.join(data_dir, "masks")
    if not os.path.isdir(images_dir):
        raise RuntimeError("MedSAM training expects data/images/ and data/masks/ directories")

    img_files = sorted([f for f in Path(images_dir).iterdir()
                        if f.suffix.lower() in img_extensions])
    msk_files = sorted([f for f in Path(masks_dir).iterdir()
                        if f.suffix.lower() in img_extensions]) if os.path.isdir(masks_dir) else []

    if not img_files:
        raise RuntimeError("No images found in images/ directory")

    print(f"[medsam] Found {len(img_files)} images, {len(msk_files)} masks")

    # Download MedSAM checkpoint
    import urllib.request
    ckpt_dir  = _tmp.mkdtemp()
    ckpt_path = os.path.join(ckpt_dir, "medsam_vit_b.pth")
    ckpt_url  = "https://huggingface.co/bowang/medsam/resolve/main/medsam_vit_b.pth"
    print("[medsam] Downloading MedSAM checkpoint (~360MB)...")
    try:
        urllib.request.urlretrieve(ckpt_url, ckpt_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download MedSAM checkpoint: {e}")

    from segment_anything import sam_model_registry
    from torch.nn.functional import threshold, normalize
    import numpy as np
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[medsam] Using device: {device}")

    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    sam.to(device)

    # Fine-tune image encoder + mask decoder (freeze prompt encoder)
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    seg_loss = torch.nn.BCEWithLogitsLoss()

    def load_img_mask(img_path, msk_path, size):
        img = np.array(Image.open(img_path).convert("RGB").resize((size, size)))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if msk_path and os.path.exists(msk_path):
            msk = np.array(Image.open(msk_path).convert("L").resize((size, size)))
            msk_t = torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        else:
            msk_t = torch.zeros(1, 1, size // 4, size // 4)
        return img_t.to(device), msk_t.to(device)

    for epoch in range(epochs):
        sam.train()
        epoch_loss = 0
        for i, img_path in enumerate(img_files):
            msk_path = str(msk_files[i]) if i < len(msk_files) else None
            img_t, msk_t = load_img_mask(str(img_path), msk_path, imgsz)
            optimizer.zero_grad()
            img_emb = sam.image_encoder(img_t)
            # Use center-point prompt
            cx, cy = imgsz // 2, imgsz // 2
            pt = torch.tensor([[[cx, cy]]], dtype=torch.float32).to(device)
            pl = torch.tensor([[1]], dtype=torch.int).to(device)
            sparse_emb, dense_emb = sam.prompt_encoder(points=(pt, pl), boxes=None, masks=None)
            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            loss = seg_loss(low_res_masks, msk_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[medsam] Epoch {epoch+1}/{epochs} loss={epoch_loss/max(1,len(img_files)):.4f}")
        print(f"PIPELINE_STEP:training")

    print(f"PIPELINE_STEP:saving")
    tmpdir2 = _tmp.mkdtemp()
    weights_path = os.path.join(tmpdir2, "medsam_finetuned.pth")
    torch.save(sam.state_dict(), weights_path)
    zip_path = weights_path.replace(".pth", ".zip")
    with _zf.ZipFile(zip_path, "w") as z:
        z.write(weights_path, os.path.basename(weights_path))
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)
    print(f"WEIGHTS_UPLOADED: {w_bucket}/{w_key}")
    print("[medsam] Done!")


def train_nnunet(data_dir, model_name, epochs, imgsz,
                 job_id, w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Run nnU-Net auto-configure segmentation pipeline."""
    import zipfile as _zf, tempfile as _tmp, subprocess, sys
    from pathlib import Path

    print(f"[nnunet] Starting nnU-Net training — model={model_name}")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "nnunetv2", "boto3", "nibabel"], capture_output=True)

    import nnunetv2
    # nnU-Net expects nnUNet_raw / nnUNet_preprocessed / nnUNet_results env vars
    base_dir = _tmp.mkdtemp(prefix="nnunet_")
    raw_dir  = os.path.join(base_dir, "nnUNet_raw")
    pre_dir  = os.path.join(base_dir, "nnUNet_preprocessed")
    res_dir  = os.path.join(base_dir, "nnUNet_results")
    for d in [raw_dir, pre_dir, res_dir]:
        os.makedirs(d, exist_ok=True)
    os.environ["nnUNet_raw"]           = raw_dir
    os.environ["nnUNet_preprocessed"] = pre_dir
    os.environ["nnUNet_results"]       = res_dir

    # ── Convert dataset to nnU-Net Dataset001 format ────────────────────────────
    ds_dir = os.path.join(raw_dir, "Dataset001_MedImage")
    img_tr = os.path.join(ds_dir, "imagesTr")
    lbl_tr = os.path.join(ds_dir, "labelsTr")
    for d in [img_tr, lbl_tr]:
        os.makedirs(d, exist_ok=True)

    import nibabel as nib
    import numpy as np
    from PIL import Image

    images_src = os.path.join(data_dir, "images")
    masks_src  = os.path.join(data_dir, "masks")
    if not os.path.isdir(images_src):
        raise RuntimeError("nnU-Net expects data/images/ and data/masks/ directories")

    img_files = sorted([f for f in Path(images_src).iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".nii", ".nii.gz")])
    msk_files = sorted([f for f in Path(masks_src).iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".nii", ".nii.gz")]) \
                if os.path.isdir(masks_src) else []

    if not img_files:
        raise RuntimeError("No images found in images/")

    print(f"[nnunet] Converting {len(img_files)} images to nnU-Net format...")
    for idx, img_path in enumerate(img_files):
        case = f"case_{idx:04d}"
        # Convert to NIfTI
        if str(img_path).endswith((".nii", ".nii.gz")):
            import shutil
            shutil.copy2(str(img_path), os.path.join(img_tr, f"{case}_0000.nii.gz"))
        else:
            arr = np.array(Image.open(str(img_path)).convert("L"))
            arr = arr[np.newaxis, np.newaxis, ...]  # (1, H, W) → 3D for nnunet
            nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)),
                     os.path.join(img_tr, f"{case}_0000.nii.gz"))
        if idx < len(msk_files):
            m_path = str(msk_files[idx])
            if m_path.endswith((".nii", ".nii.gz")):
                import shutil
                shutil.copy2(m_path, os.path.join(lbl_tr, f"{case}.nii.gz"))
            else:
                arr = np.array(Image.open(m_path).convert("L"))
                arr = (arr > 127).astype(np.uint8)[np.newaxis, ...]
                nib.save(nib.Nifti1Image(arr, np.eye(4)),
                         os.path.join(lbl_tr, f"{case}.nii.gz"))

    # Write dataset.json
    import json as _json
    _json.dump({
        "name": "MedImage", "description": "Auto-generated by H-Forge",
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "foreground": 1},
        "numTraining": len(img_files), "file_ending": ".nii.gz",
    }, open(os.path.join(ds_dir, "dataset.json"), "w"))

    print("[nnunet] Preprocessing...")
    print(f"PIPELINE_STEP:export")
    subprocess.run([sys.executable, "-m", "nnunetv2.experiment_planning.plan_and_preprocess",
                    "-d", "1", "--verify_dataset_integrity"], check=True)

    print("[nnunet] Training (2d config)...")
    print(f"PIPELINE_STEP:training")
    subprocess.run([sys.executable, "-m", "nnunetv2.run.run_training",
                    "1", "2d", "0",
                    "--num_epochs", str(epochs)], check=True)

    print(f"PIPELINE_STEP:saving")
    # Pack results
    tmpdir2 = _tmp.mkdtemp()
    zip_path = os.path.join(tmpdir2, "nnunet_results.zip")
    with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(res_dir):
            for f in files:
                fp = os.path.join(root, f)
                z.write(fp, os.path.relpath(fp, res_dir))
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)
    print(f"WEIGHTS_UPLOADED: {w_bucket}/{w_key}")
    print("[nnunet] Done!")


def train_anomalib_anomaly(img_dir, model_name, imgsz, job_id,
                            w_bucket, w_key, minio_url, minio_access, minio_secret):
    """PaDiM-style anomaly detection using pure torch/torchvision (no anomalib dep)."""
    import zipfile as _zf
    import torch, torchvision
    import numpy as np
    from torchvision import transforms, models
    from PIL import Image
    import glob as _glob

    print(f"[train] Anomaly detection (PaDiM-style) — model: {model_name}  images: {img_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")

    # Collect all images from img_dir recursively
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    img_files = []
    for ext in exts:
        img_files += _glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
    img_files = sorted(set(img_files))
    if not img_files:
        raise RuntimeError(f"No images found under {img_dir}")
    print(f"[train] Found {len(img_files)} normal images")

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load pretrained backbone (wide_resnet50_2 for PaDiM, else resnet50)
    arch = model_name.lower()
    if "wide" in arch or "padim" in arch:
        backbone = models.wide_resnet50_2(weights="DEFAULT")
    elif "patchcore" in arch:
        backbone = models.wide_resnet50_2(weights="DEFAULT")
    else:
        backbone = models.resnet50(weights="DEFAULT")
    backbone = backbone.to(device).eval()

    # Extract global average-pooled features using the layer before fc
    features_list = []
    hook_output = {}
    def _hook(module, inp, out):
        hook_output["feat"] = out.detach()
    hook_handle = backbone.avgpool.register_forward_hook(_hook)

    print("[train] Extracting features from normal images ...")
    with torch.no_grad():
        for img_path in img_files:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            tensor = preprocess(img).unsqueeze(0).to(device)
            backbone(tensor)
            feat = hook_output["feat"].squeeze(-1).squeeze(-1).cpu().numpy()  # (1, C)
            features_list.append(feat)
    hook_handle.remove()

    if not features_list:
        raise RuntimeError("Feature extraction produced no results")

    features = np.vstack(features_list)  # (N, C)
    print(f"[train] Features shape: {features.shape}")

    # Fit gaussian: compute mean and covariance
    mean = features.mean(axis=0)  # (C,)
    cov = np.cov(features.T) + np.eye(features.shape[1]) * 1e-4  # regularise
    cov_inv = np.linalg.pinv(cov)

    # Compute per-sample mahalanobis distance for threshold estimation
    diffs = features - mean  # (N, C)
    dists = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs), 0))
    threshold = float(np.percentile(dists, 95))  # 95th percentile as anomaly threshold
    print(f"[train] Threshold (95th pct): {threshold:.4f}  mean_dist: {dists.mean():.4f}")

    # Save model state as .pt
    save_dir = f"/tmp/anomaly_{job_id}"
    os.makedirs(save_dir, exist_ok=True)
    pt_path = os.path.join(save_dir, "model.pt")
    torch.save({
        "arch": arch,
        "imgsz": imgsz,
        "mean": torch.tensor(mean, dtype=torch.float32),
        "cov_inv": torch.tensor(cov_inv, dtype=torch.float32),
        "threshold": threshold,
        "num_train_images": len(features_list),
    }, pt_path)
    print(f"[train] Saved model state to {pt_path}")

    # Zip and upload
    zip_path = f"/tmp/anomaly_{job_id}.zip"
    with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED) as zf:
        zf.write(pt_path, "model.pt")
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)
    print(f"[train] WEIGHTS_UPLOADED: s3://{w_bucket}/{w_key}")


def train_self_supervised(img_dir, model_name, epochs, imgsz, batch, lr, job_id,
                           w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Self-supervised contrastive pre-training using SimCLR-style approach (pure torch/torchvision)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image as _Image
    import glob as _glob, zipfile as _zf

    print(f"[train] Self-supervised ({model_name}) — images: {img_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")

    # Collect images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    img_files = []
    for ext in exts:
        img_files += _glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
    img_files = sorted(set(img_files))
    if not img_files:
        raise RuntimeError(f"No images found under {img_dir}")
    print(f"[train] Found {len(img_files)} images")

    # Augmentation pair transforms (SimCLR-style)
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    aug = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class ContrastiveDS(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = _Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), self.transform(img)  # two augmented views

    loader = DataLoader(ContrastiveDS(img_files, aug), batch_size=min(batch, len(img_files)),
                        shuffle=True, num_workers=2, drop_last=False)

    # Build encoder backbone
    arch = model_name.lower().replace("-", "").replace("_", "")
    if "resnet50" in arch:
        backbone = models.resnet50(weights=None)
        feat_dim = 2048
    elif "resnet101" in arch:
        backbone = models.resnet101(weights=None)
        feat_dim = 2048
    else:  # default resnet18
        backbone = models.resnet18(weights=None)
        feat_dim = 512
    # Remove final classifier
    backbone.fc = nn.Identity()
    # Projection head (SimCLR paper: 2-layer MLP)
    proj_dim = 128
    projector = nn.Sequential(
        nn.Linear(feat_dim, 512), nn.ReLU(inplace=True),
        nn.Linear(512, proj_dim),
    )

    class SimCLR(nn.Module):
        def __init__(self, enc, proj):
            super().__init__()
            self.encoder   = enc
            self.projector = proj
        def forward(self, x):
            h = self.encoder(x)
            z = F.normalize(self.projector(h), dim=1)
            return z

    model_ssl = SimCLR(backbone, projector).to(device)
    optimizer_ssl = torch.optim.AdamW(model_ssl.parameters(), lr=lr, weight_decay=1e-4)

    # NT-Xent (InfoNCE) loss
    def nt_xent(z1, z2, temp=0.5):
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)           # (2N, D)
        sim = torch.mm(z, z.T) / temp             # (2N, 2N)
        # mask self-similarity
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(device)
        return F.cross_entropy(sim, labels)

    print(f"[train] Starting SimCLR training — {epochs} epoch(s)")
    best_loss = float("inf")
    save_dir  = f"/tmp/ssl_{job_id}"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model_ssl.train()
        total_loss = 0.0
        steps = 0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = model_ssl(v1), model_ssl(v2)
            loss = nt_xent(z1, z2)
            optimizer_ssl.zero_grad()
            loss.backward()
            optimizer_ssl.step()
            total_loss += loss.item()
            steps += 1
        avg = total_loss / max(steps, 1)
        print(f"[train] Epoch {epoch}/{epochs} — loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({
                "epoch": epoch,
                "encoder_state": backbone.state_dict(),
                "model_state":   model_ssl.state_dict(),
                "loss": best_loss,
                "arch": arch,
                "feat_dim": feat_dim,
                "proj_dim": proj_dim,
                "imgsz": imgsz,
            }, os.path.join(save_dir, "best.pt"))
    print(f"[train] Best loss: {best_loss:.4f}")
    # Zip and upload
    zip_path = f"/tmp/ssl_{job_id}.zip"
    with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED) as zf:
        for fp in Path(save_dir).rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(save_dir))
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)
    print(f"[train] WEIGHTS_UPLOADED: s3://{w_bucket}/{w_key}")


def _move_batch_to_device(batch, device):
    """Recursively move every tensor inside a batch dict/list/BatchFeature to ``device``.

    HuggingFace detection pipelines return a BatchFeature like
    ``{"pixel_values": Tensor, "labels": [{"boxes": Tensor, "class_labels": Tensor}, ...]}``
    where ``labels`` is a list of per-image dicts. BatchFeature's own
    ``.to()`` is shallow — it moves the top-level Tensor values
    (``pixel_values``) but does NOT descend into the nested list-of-dicts
    in ``labels``. The Hungarian matcher then crashes at
    ``torch.cdist(out_bbox, target_bbox, p=1)`` with
    "Expected all tensors to be on the same device, but found at least
    two devices, cuda:0 and cpu!".

    Implementation:
    - Detect BatchFeature (duck-typed: has ``data`` dict attribute) and
      unwrap to that dict, so we recursively walk the same way.
    - Duck-type ``Tensor`` as anything with a ``.to(device)`` method.
    - Recurse into dicts, lists, tuples.
    - Leaves everything else (strs, ints, None) untouched.

    Using duck-typing for the tensor check also lets the helper work
    before ``torch`` is imported at module top level — calling
    ``isinstance(batch, torch.Tensor)`` from inside the _VISION_TRAIN_SCRIPT
    string would NameError because the helper runs before any
    ``import torch`` statement executes.
    """
    # Unwrap BatchFeature: it has a `.data` dict but its own shallow .to()
    if hasattr(batch, "data") and isinstance(getattr(batch, "data", None), dict) \
            and not isinstance(batch, dict):
        return _move_batch_to_device(dict(batch.data), device)
    if hasattr(batch, "to") and callable(batch.to):
        # Real Tensor / numpy / etc. Try to move; fall back to original on error.
        try:
            return batch.to(device)
        except Exception:
            return batch
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_move_batch_to_device(v, device) for v in batch]
        return type(batch)(moved) if isinstance(batch, tuple) else moved
    return batch


def train_hf_vision(data_dir, model_name, training_type, epochs, batch, lr, optimizer,
                    job_id, w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Fine-tune a HuggingFace vision model (DETR detection or ViT classification)."""
    import json, os, glob, time
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoImageProcessor

    # ── Collect images + YOLO-format labels ──────────────────────────────────
    img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")
    all_images = []
    for ext in img_extensions:
        all_images += glob.glob(os.path.join(data_dir, "**", ext), recursive=True)
    if not all_images:
        # Log directory contents to help diagnose missing images
        listing = []
        for root, dirs, files in os.walk(data_dir):
            for f in files[:5]:
                listing.append(os.path.relpath(os.path.join(root, f), data_dir))
        raise RuntimeError(f"No images found in {data_dir}. Contents: {listing[:20] or '(empty)'}")

    # Build class list from all label files
    class_set = set()
    for img_path in all_images:
        lbl_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(lbl_path):
            base = os.path.basename(os.path.splitext(img_path)[0])
            lbl_path = os.path.join(data_dir, "labels", base + ".txt")
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_set.add(int(parts[0]))
    # Try yaml for class names
    yaml_files = glob.glob(os.path.join(data_dir, "*.yaml")) + glob.glob(os.path.join(data_dir, "**", "*.yaml"), recursive=True)
    class_names = {}
    for yf in yaml_files:
        try:
            import yaml
            with open(yf) as f:
                d = yaml.safe_load(f)
            if isinstance(d.get("names"), list):
                class_names = {i: n for i, n in enumerate(d["names"])}
                break
            elif isinstance(d.get("names"), dict):
                class_names = {int(k): v for k, v in d["names"].items()}
                break
        except Exception:
            pass
    if not class_names:
        class_names = {i: f"class_{i}" for i in sorted(class_set)}
    num_classes = len(class_names)
    id2label = {i: class_names[i] for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}
    print(f"[train] Classes: {id2label}")
    print(f"[train] Images : {len(all_images)}")

    split = max(1, int(len(all_images) * 0.9))
    train_imgs, val_imgs = all_images[:split], all_images[split:] or all_images[:1]

    # ── Validate HF access BEFORE doing dataset prep (fail fast) ──────────
    # Gated models (e.g. MahmoodLab/UNI, microsoft/rad-dino) return 401/403
    # with an opaque "Access to model ... is restricted" message that's
    # hard to debug in the worker log. Two checks up front:
    #   1. If no HF_TOKEN at all → tell user to set one
    #   2. If gated AND has token, probe a real file download — many
    #      gated models (UNIs, rad-dino) require EXPLICIT APPROVAL on
    #      the HF model page before any token works. A 403 here means
    #      the user needs to go to the model page and click "Agree and
    #      access repository", then retry.
    _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    _hf_token_present = bool(_hf_token)
    _is_gated = False
    _gated_kind = None  # "auto" / "manual" / "false"
    try:
        from huggingface_hub import HfApi
        _api = HfApi(token=_hf_token)
        _model_info = _api.model_info(model_name)
        _g = getattr(_model_info, "gated", False)
        _gated_kind = _g if isinstance(_g, str) else ("true" if _g else "false")
        _is_gated = bool(_g) and _gated_kind != "false"
    except Exception:
        pass  # probe itself failed — let the actual download report the error

    if _is_gated and not _hf_token_present:
        raise RuntimeError(
            f"Model '{model_name}' is a GATED HuggingFace model — it requires "
            f"a HuggingFace token.\n"
            f"  → In this UI: Profile → HuggingFace Token → paste your token (from "
            f"https://huggingface.co/settings/tokens) → Save.\n"
            f"  → Or set HF_TOKEN in the medimage-api container's environment."
        )

    if _is_gated and _hf_token_present and _gated_kind in ("manual", "auto", "true"):
        # Many gated repos require the user to explicitly request
        # access on the HF model page BEFORE their token works. Probe
        # by trying to fetch a tiny metadata file — if the HF API
        # returns 403/401 the user hasn't been approved yet.
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=model_name, filename="config.json",
                token=_hf_token, cache_dir=None,
            )
        except Exception as _probe_err:
            _err_str = str(_probe_err).lower()
            if any(t in _err_str for t in ("403", "401", "gated", "restricted", "authorized list")):
                raise RuntimeError(
                    f"Model '{model_name}' is GATED and your HF token is not in the "
                    f"authorized list.\n"
                    f"  → Go to https://huggingface.co/{model_name} in a browser\n"
                    f"  → Click the 'Agree and access repository' button (you may need "
                    f"to log in with the same HF account that owns the token set in "
                    f"your Profile)\n"
                    f"  → Wait ~1 min for HF to propagate the approval\n"
                    f"  → Retry the test-all run\n"
                    f"\nUnderlying error: {_probe_err}"
                ) from _probe_err

    processor = AutoImageProcessor.from_pretrained(
        model_name, trust_remote_code=True, token=_hf_token or True
    )

    if training_type == "detection":
        from transformers import AutoModelForObjectDetection

        class YoloDetectDataset(Dataset):
            def __init__(self, paths):
                self.paths = paths
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                img_path = self.paths[idx]
                image = Image.open(img_path).convert("RGB")
                lbl_path = os.path.splitext(img_path)[0] + ".txt"
                if not os.path.exists(lbl_path):
                    base = os.path.basename(os.path.splitext(img_path)[0])
                    lbl_path = os.path.join(data_dir, "labels", base + ".txt")
                boxes, labels = [], []
                if os.path.exists(lbl_path):
                    with open(lbl_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                boxes.append([cx, cy, w, h])
                                labels.append(cls_id)
                # COCO format expected by DETR/YOLOS ImageProcessor:
                #   bbox = [x_top_left, y_top_left, width, height]  in ABSOLUTE pixels
                #   category_id = int class index
                #   area = w * h
                #   iscrowd = 0
                # The image processor's do_convert_annotations=True (default)
                # then re-projects these to relative [cx, cy, w, h] in [0, 1] and
                # builds the final {"class_labels": LongTensor, "boxes": FloatTensor}
                # entry that YolosForObjectDetection.forward expects.
                img_w, img_h = image.size
                annotations = {"image_id": idx, "annotations": [
                    {
                        "bbox": [cx * img_w - w * img_w / 2,
                                 cy * img_h - h * img_h / 2,
                                 w * img_w, h * img_h],
                        "category_id": l,
                        "area": (w * img_w) * (h * img_h),
                        "iscrowd": 0,
                    }
                    for (cx, cy, w, h), l in zip(boxes, labels)
                ]}
                return image, annotations

        def collate_fn(batch):
            images, anns = zip(*batch)
            encodings = processor(images=list(images), annotations=list(anns), return_tensors="pt")
            return encodings

        # Validate that the model supports fine-tuning via AutoModelForObjectDetection
        # Zero-shot / open-vocabulary detectors (GroundingDINO, OWL-ViT, etc.) are NOT supported
        _UNSUPPORTED_DET = ("grounding_dino", "owlvit", "owlv2", "sam", "gdino")
        try:
            from transformers import AutoConfig
            _cfg_type = type(AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, token=_hf_token or True
            )).__name__.lower()
            if any(u in _cfg_type for u in _UNSUPPORTED_DET):
                raise RuntimeError(
                    f"Model '{model_name}' ({_cfg_type}) is a zero-shot / open-vocabulary detector "
                    f"and cannot be fine-tuned with the HuggingFace detection pipeline.\n"
                    f"Supported architectures: DETR, Deformable-DETR, RT-DETR, YOLOS, DETA.\n"
                    f"Try: facebook/detr-resnet-50  or  hustvl/yolos-tiny"
                )
        except RuntimeError:
            raise
        except Exception:
            pass  # if config fetch fails, proceed and let from_pretrained raise

        model = AutoModelForObjectDetection.from_pretrained(
            model_name, num_labels=num_classes,
            id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
            trust_remote_code=True, token=_hf_token or True,
        )

    elif training_type == "classification":
        from transformers import AutoModelForImageClassification

        class SimpleClassifyDataset(Dataset):
            def __init__(self, paths):
                self.paths = paths
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                img_path = self.paths[idx]
                image = Image.open(img_path).convert("RGB")
                lbl_path = os.path.splitext(img_path)[0] + ".txt"
                label = 0
                if os.path.exists(lbl_path):
                    with open(lbl_path) as f:
                        line = f.readline().strip().split()
                        if line:
                            label = int(line[0])
                return image, label

        def collate_fn(batch):
            images, labels = zip(*batch)
            encodings = processor(images=list(images), return_tensors="pt")
            encodings["labels"] = torch.tensor(list(labels), dtype=torch.long)
            return encodings

        model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=num_classes,
            id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
            trust_remote_code=True, token=_hf_token or True,
        )
    else:
        raise RuntimeError(f"HuggingFace training not supported for type={training_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer_fn = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if training_type == "detection":
        train_ds = YoloDetectDataset(train_imgs)
        val_ds   = YoloDetectDataset(val_imgs)
    else:
        train_ds = SimpleClassifyDataset(train_imgs)
        val_ds   = SimpleClassifyDataset(val_imgs)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=2)

    best_loss = float("inf")
    best_path = f"/tmp/best_{job_id}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_enc in train_dl:
            batch_enc = _move_batch_to_device(batch_enc, device)
            optimizer_fn.zero_grad()
            outputs = model(**batch_enc)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_fn.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_dl), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_enc in val_dl:
                batch_enc = _move_batch_to_device(batch_enc, device)
                outputs = model(**batch_enc)
                val_loss += outputs.loss.item()
        avg_val = val_loss / max(len(val_dl), 1)

        print(f"[train] Epoch [{epoch:>3}/{epochs}]  loss={avg_loss:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model.config.to_dict(),
                "id2label": id2label,
                "label2id": label2id,
                "training_type": training_type,
                "model_name": model_name,
                "val_loss": avg_val,
            }, best_path)

    print(f"[train] Best val_loss: {best_loss:.4f}")
    upload_to_minio(best_path, w_bucket, w_key, minio_url, minio_access, minio_secret)
    # Emit the full s3:// URI, not just the key. The outer reconciler
    # parses WEIGHTS_UPLOADED lines and verifies the object exists in
    # MinIO; if the line only has the key, the parser has to guess
    # the bucket (and the previous version of that guesser was buggy).
    print(f"[train] WEIGHTS_UPLOADED: s3://{w_bucket}/{w_key}")


def train_ultralytics(yaml_path, data_dir, model_name, training_type, epochs, imgsz, batch, lr, optimizer, job_id, w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Train detection / segmentation / YOLO-cls model with ultralytics."""
    import sys, types as _types
    # Patch add_integration_callbacks to no-op BEFORE importing YOLO.
    # Ultralytics iterates through all integrations (clearml, mlflow, comet, …)
    # each of which imports pandas — which crashes on this cluster due to numpy
    # ABI mismatch (pandas compiled for numpy 2.x, cluster has numpy 1.x).
    try:
        import ultralytics.utils.callbacks.base as _ucb
        _ucb.add_integration_callbacks = lambda *a, **kw: None
    except Exception:
        pass
    # Also stub out any already-registered integration loaders via SETTINGS
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"clearml": False, "comet": False, "hub": False,
                         "mlflow": False, "neptune": False, "raytune": False,
                         "tensorboard": False, "wandb": False})
    except Exception:
        pass
    # Stub the most common offenders so import-time crashes can't happen
    for _m in ("clearml", "mlflow", "comet_ml", "neptune", "wandb"):
        if _m not in sys.modules:
            sys.modules[_m] = _types.ModuleType(_m)
    from ultralytics import YOLO
    print(f"[train] Loading model: {model_name}")
    # Resolve non-standard .pt filenames to a HF Hub repo if the file isn't
    # available locally. Ultralytics YOLO() will try to fetch `yolov8s.pt` etc.
    # from its own model index, but community weights (road-damage-yolov8s.pt,
    # rtdetr-l.pt etc.) need to be downloaded explicitly. Format: "hf_repo:filename"
    # or just a HF repo path; we default to looking up the same basename.
    _model_name = model_name
    if not _model_name.endswith(".pt"):
        pass  # not a .pt — assume Ultralytics knows it
    else:
        import os as _os
        if not _os.path.isfile(_model_name):
            # Try a few well-known HF repos for the same basename.
            _basename = _os.path.basename(_model_name)
            _candidates = []
            if "road-damage" in _basename or "rdd" in _basename.lower():
                _candidates.append(("oracl4/RoadDamageDetection", _basename))
                _candidates.append(("oracl4/RoadDamageDetection", "best.pt"))
                _candidates.append(("ozair23/yolov8-road-damage-detector", "best.pt"))
            if "rtdetr" in _basename.lower():
                _candidates.append(("Intel/RT-DETR", _basename))
            for _repo, _fn in _candidates:
                try:
                    from huggingface_hub import hf_hub_download
                    print(f"[train] weights {model_name!r} not found locally; trying {_repo}::{_fn} ...")
                    _local = hf_hub_download(repo_id=_repo, filename=_fn, cache_dir=None)
                    print(f"[train] Downloaded weights: {_local}")
                    _model_name = _local
                    break
                except Exception as _dl_err:
                    print(f"[train] HF download failed: {_dl_err}")
                    continue
    model = YOLO(_model_name)
    print(f"[train] Training started ...")
    results = model.train(
        data=yaml_path,
        epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr,
        optimizer=optimizer, project="/tmp/yolo_runs", name=job_id,
        exist_ok=True, verbose=True,
    )
    # ultralytics >= 8.x: model.train() returns the Results object on
    # success, but on certain validation/edge cases it can return None.
    # We always fall back to the on-disk save dir + best.pt path so the
    # training pipeline can continue even when results is None.
    save_dir = getattr(results, "save_dir", None) if results is not None else None
    if save_dir is None:
        save_dir = f"/tmp/yolo_runs/{job_id}"
        print(f"[train] results.save_dir is None; falling back to {save_dir}")
    print(f"[train] Training done. Save dir: {save_dir}")
    best_pt = Path(save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(save_dir) / "weights" / "last.pt"
    if not best_pt.exists():
        print("[train] ERROR: No weights file found after training!")
        sys.exit(1)
    upload_to_minio(best_pt, w_bucket, w_key, minio_url, minio_access, minio_secret)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import requests, boto3
    from botocore.config import Config

    engine        = os.environ.get("ENGINE", "Ultralytics")
    training_type = os.environ.get("TRAINING_TYPE", "detection")
    model_name    = os.environ.get("MODEL_NAME", "yolov8n.pt")
    epochs        = int(os.environ.get("EPOCHS", "10"))
    imgsz         = int(os.environ.get("IMGSZ", "640"))
    batch         = int(os.environ.get("BATCH_SIZE", "16"))
    lr            = float(os.environ.get("LR", "0.001"))
    optimizer     = os.environ.get("OPTIMIZER", "AdamW")
    job_id        = os.environ.get("JOB_ID", "unknown")
    project_id    = int(os.environ.get("PROJECT_ID", "0") or 0)
    dataset_url   = os.environ.get("DATASET_URL", "")
    w_bucket      = os.environ.get("WEIGHTS_BUCKET", "medimage-weights")
    w_key         = os.environ.get("WEIGHTS_KEY", f"{job_id}/best.pt")
    minio_url     = os.environ.get("MINIO_URL", "http://minio:9000")
    minio_access  = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret  = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    ls_url        = os.environ.get("LS_PUBLIC_URL", "http://localhost:8080")
    ls_token      = os.environ.get("LS_TOKEN", "")
    hf_token      = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    # LLM-specific extras
    text_dataset  = os.environ.get("TEXT_DATASET", "")
    lora_rank     = int(os.environ.get("LORA_RANK", "16"))
    quantization  = os.environ.get("QUANTIZATION", "4bit")
    max_seq_len   = int(os.environ.get("MAX_SEQ_LEN", "2048"))
    chat_template = os.environ.get("CHAT_TEMPLATE", "chatml")
    grad_accum    = int(os.environ.get("GRAD_ACCUM", "4"))

    print(f"[train] Job {job_id} — engine={engine} type={training_type} model={model_name} epochs={epochs}")

    # ── Engines that do NOT need an image dataset zip ─────────────────────────
    # VLM / LLM are handled by vlm_llm_train.py (separate module so changes
    # to the deep-learning path don't affect them and vice versa). This
    # _VISION_TRAIN_SCRIPT path only handles the image-dataset engines:
    # PyTorch, TorchVision, TIMM, MONAI, MedSAM, nnU-Net, Anomalib,
    # Ultralytics (YOLO), and the HuggingFace vision path. The dispatcher
    # in main.py:_run_on_ray_cluster routes VLM/LLM jobs to the
    # vlm_llm_train.run() entry point instead of exec'ing this script.

    # 3-HF. HuggingFace vision model fine-tuning (DETR detection, ViT classification)
    if engine == "HuggingFace" and training_type in ("detection", "classification", "segmentation"):
        tmpdir = tempfile.mkdtemp(prefix="hftrain_")
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir, exist_ok=True)
        ls_tasks_json = None  # may be populated from JSON export

        if dataset_url:
            import requests as _req, zipfile as _zf
            zip_path = os.path.join(tmpdir, "dataset.zip")
            print("[train] Downloading dataset ...")
            r = _req.get(dataset_url, stream=True, timeout=600)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            with _zf.ZipFile(zip_path) as z: z.extractall(data_dir)
            # Check for JSON export (annotations.json) — download images from MinIO
            ann_json_path = os.path.join(data_dir, "annotations.json")
            if os.path.exists(ann_json_path):
                with open(ann_json_path) as _jf:
                    ls_tasks_json = json.load(_jf)

        # If JSON export, download images from MinIO using S3 paths in tasks
        if ls_tasks_json:
            import boto3 as _b3
            from botocore.config import Config as _Cfg
            img_dir = os.path.join(data_dir, "images", "train")
            lbl_dir = os.path.join(data_dir, "labels", "train")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            s3 = _b3.client(
                "s3",
                endpoint_url=minio_url,
                aws_access_key_id=minio_access,
                aws_secret_access_key=minio_secret,
                config=_Cfg(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
            downloaded = 0
            for idx, task in enumerate(ls_tasks_json):
                img_url = task.get("data", {}).get("image", "")
                if not img_url:
                    continue
                task_id = task.get("id", idx)
                if img_url.startswith("data:"):
                    fname_base = f"task_{task_id}"
                else:
                    fname_base = os.path.splitext(os.path.basename(img_url.split("?")[0]))[0] or f"task_{task_id}"
                img_dest = os.path.join(img_dir, fname_base + ".jpg")
                try:
                    if img_url.startswith("data:"):
                        # Base64 data URI — decode directly
                        import base64 as _b64
                        header, b64data = img_url.split(",", 1)
                        ext = ".png" if "png" in header else ".jpg"
                        img_dest = os.path.join(img_dir, fname_base + ext)
                        with open(img_dest, "wb") as _f:
                            _f.write(_b64.b64decode(b64data))
                    elif img_url.startswith("s3://"):
                        # s3://bucket/key
                        parts = img_url[5:].split("/", 1)
                        s3.download_file(parts[0], parts[1], img_dest)
                    else:
                        # Relative LS path like /data/upload/1/xxx.jpg
                        # or full http URL — always resolve via LS public URL with auth
                        if img_url.startswith("/") or not img_url.startswith("http"):
                            full_url = ls_url.rstrip("/") + img_url
                        else:
                            full_url = img_url
                        r2 = _req.get(
                            full_url,
                            headers={"Authorization": f"Token {ls_token}"},
                            timeout=60,
                        )
                        r2.raise_for_status()
                        with open(img_dest, "wb") as _f: _f.write(r2.content)
                    downloaded += 1
                except Exception as _de:
                    print(f"[train] Warning: could not download image for task {task.get('id')}: {_de}")
                    continue
                # Write YOLO-format label from LS annotations
                anns = task.get("annotations", [])
                if anns:
                    results = anns[0].get("result", [])
                    with open(os.path.join(lbl_dir, fname_base + ".txt"), "w") as _lf:
                        for res in results:
                            v = res.get("value", {})
                            if res.get("type") == "rectanglelabels":
                                x_c = (v["x"] + v["width"] / 2) / 100
                                y_c = (v["y"] + v["height"] / 2) / 100
                                w   = v["width"] / 100
                                h   = v["height"] / 100
                                _lf.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                            elif res.get("type") == "choices":
                                _lf.write("0\n")
            print(f"[train] Downloaded {downloaded}/{len(ls_tasks_json)} images from LS/MinIO")
            if downloaded == 0:
                raise RuntimeError(
                    f"Failed to download any images. LS_PUBLIC_URL={ls_url}, "
                    f"first task image URL: {ls_tasks_json[0].get('data', {}).get('image', 'N/A') if ls_tasks_json else 'N/A'}"
                )

        train_hf_vision(
            data_dir, model_name, training_type, epochs, batch, lr, optimizer,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )
        print("[train] Complete!")
        return

    # 3-SAM. SAM is zero-shot — no training
    if engine in ("Meta SAM", "Meta SAM2"):
        raise RuntimeError(
            "SAM / SAM2 is a zero-shot foundation model and does not require training. "
            "Use it directly for inference or auto pre-labeling in Label Studio."
        )

    # 3-SKIP. Export-only engines
    if engine in ("TF-Lite", "ONNX", "NVIDIA", "Intel", "Apple"):
        raise RuntimeError(
            f"Engine '{engine}' is for model export/conversion, not training. "
            "First train a model, then use the export feature."
        )

    # ── Download dataset zip ──────────────────────────────────────────────────
    # 1. Download dataset zip
    tmpdir   = tempfile.mkdtemp(prefix="ytrain_")
    zip_path = os.path.join(tmpdir, "dataset.zip")
    data_dir = os.path.join(tmpdir, "data")
    print("[train] Downloading dataset ...")
    try:
        r = requests.get(dataset_url, stream=True, timeout=600)
        r.raise_for_status()
    except Exception as _dl_err:
        # The presigned URL's host is part of the S3v4 signature, so the
        # host the worker hits MUST be the one the API signed against.
        # The API's _resolve_minio_url_for_ray() picks that host from
        # MINIO_PUBLIC_URL / MINIO_HOST_IP / RAY_URL. If the worker can't
        # reach the chosen host, surface a clear actionable message
        # rather than just the raw socket error.
        from urllib.parse import urlparse as _urlparse
        try:
            _u = _urlparse(dataset_url)
            _bad_host = f"{_u.hostname}:{_u.port}" if _u.hostname else dataset_url
        except Exception:
            _bad_host = dataset_url
        raise RuntimeError(
            f"Failed to download dataset from {_bad_host}: {_dl_err}.\n"
            f"  The Ray worker tried to reach MinIO at the host the API "
            f"signed the URL against ({_bad_host}). That host is either "
            f"DNS-unresolvable, refusing the connection, or firewalled.\n"
            f"  Fix: on the medimage-api container, set MINIO_HOST_IP "
            f"to the IP of the host running docker-compose "
            f"(or MINIO_PUBLIC_URL=http://that-host:9000) and restart "
            f"the API. The host must be reachable from the Ray cluster "
            f"on port 9000."
        )
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"[train] Dataset downloaded ({os.path.getsize(zip_path)//1024} KB)")

    # 2. Extract
    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    print(f"[train] Extracted to {data_dir}")

    # ── Helper: find LS JSON in extracted zip ─────────────────────────────────
    def _find_ls_json():
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(root, fname)) as jf:
                            d = json.load(jf)
                        if isinstance(d, list):
                            return d
                    except Exception:
                        pass
        return None

    # 3a. Classification — all PyTorch/TIMM/TorchVision engines
    CLASSIFICATION_ENGINES = {"PyTorch", "PyTorch+TIMM", "TorchVision", "TIMM"}
    if training_type == "classification" and engine in CLASSIFICATION_ENGINES:
        json_data = _find_ls_json()
        if not json_data:
            raise RuntimeError("No JSON annotation file found in dataset zip")
        print(f"[train] {len(json_data)} tasks in JSON export")

        img_dir = os.path.join(tmpdir, "cls_images")
        os.makedirs(img_dir, exist_ok=True)
        n_downloaded = download_images_from_ls(json_data, img_dir, ls_url, ls_token)
        if n_downloaded == 0:
            raise RuntimeError(
                f"No images were downloaded from Label Studio ({ls_url}). "
                "Check LS_PUBLIC_URL and LS_TOKEN environment variables."
            )
        print(f"[train] {n_downloaded} images downloaded and organised by class")
        train_pytorch_classification(
            img_dir, model_name, epochs, imgsz, batch, lr, optimizer,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )

    # 3b. Segmentation Models PyTorch / TorchVision segmentation path (UNet, UNet++, DeepLabV3+)
    elif engine in ("Segmentation Models PyTorch", "TorchVision") and training_type == "segmentation":
        json_data = _find_ls_json()
        if not json_data:
            raise RuntimeError("No JSON annotation file found in dataset zip")
        print(f"[train] {len(json_data)} tasks in JSON export")
        seg_dir = os.path.join(tmpdir, "seg_data")
        os.makedirs(seg_dir, exist_ok=True)
        pairs, class_names = prepare_segmentation_masks_from_json(json_data, seg_dir, ls_url, ls_token)
        train_pytorch_segmentation(
            pairs, class_names, model_name, epochs, imgsz, batch, lr, optimizer,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )

    # 3c. Anomalib anomaly detection (PaDiM, PatchCore)
    elif engine == "Anomalib":
        json_data = _find_ls_json()
        img_dir = os.path.join(tmpdir, "anom_images")
        os.makedirs(img_dir, exist_ok=True)
        if json_data:
            # Download images from LS (ignore labels — all treated as normal)
            import base64 as _b64
            import requests as _rq2
            for task in json_data:
                image_url = task.get("data", {}).get("image", "")
                if not image_url: continue
                tid = task.get("id", "x")
                dst = os.path.join(img_dir, f"task_{tid}.jpg")
                try:
                    if image_url.startswith("data:"):
                        _, data = image_url.split(",", 1)
                        img_b = _b64.b64decode(data)
                    else:
                        url = re.sub(r"https?://[^/]+", ls_url, image_url)
                        resp = _rq2.get(url, headers={"Authorization": f"Token {ls_token}"}, timeout=60)
                        resp.raise_for_status()
                        img_b = resp.content
                    with open(dst, "wb") as f: f.write(img_b)
                except Exception as e:
                    print(f"[warn] Cannot fetch task {tid}: {e}")
        else:
            # Copy any images found in data_dir
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        import shutil as _sh2
                        _sh2.copy2(os.path.join(root, f), img_dir)
        n_imgs = len([x for x in os.listdir(img_dir) if x.endswith(".jpg")])
        print(f"[train] Anomaly detection: {n_imgs} normal images")
        if n_imgs == 0:
            raise RuntimeError("No images available for anomaly detection training")
        train_anomalib_anomaly(
            img_dir, model_name, imgsz, job_id,
            w_bucket, w_key, minio_url, minio_access, minio_secret,
        )

    # 3e. MONAI medical imaging (classification or segmentation)
    elif engine == "MONAI":
        # Auto-prepare images/ + masks/ from LS JSON annotations if not already present
        med_data_dir = os.path.join(tmpdir, "monai_data")
        if training_type == "segmentation" and not os.path.isdir(os.path.join(data_dir, "images")):
            json_data = _find_ls_json()
            if json_data:
                print("[monai] Auto-converting LS annotations → images/ + masks/")
                prepare_segmentation_masks_from_json(json_data, med_data_dir, ls_url, ls_token)
            else:
                med_data_dir = data_dir  # fallback: use raw extracted zip
        elif training_type == "classification" and not os.path.isdir(os.path.join(data_dir, "images")):
            json_data = _find_ls_json()
            if json_data:
                print("[monai] Auto-converting LS annotations → class subdirs")
                med_data_dir = os.path.join(tmpdir, "monai_cls")
                download_images_from_ls(json_data, med_data_dir, ls_url, ls_token)
            else:
                med_data_dir = data_dir
        else:
            med_data_dir = data_dir
        train_monai(
            med_data_dir, model_name, training_type, epochs, imgsz, batch, lr,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )

    # 3f. MedSAM fine-tuning (medical segmentation)
    elif engine == "MedSAM":
        med_data_dir = os.path.join(tmpdir, "medsam_data")
        if not os.path.isdir(os.path.join(data_dir, "images")):
            json_data = _find_ls_json()
            if json_data:
                print("[medsam] Auto-converting LS annotations → images/ + masks/")
                prepare_segmentation_masks_from_json(json_data, med_data_dir, ls_url, ls_token)
            else:
                med_data_dir = data_dir
        else:
            med_data_dir = data_dir
        train_medsam(
            med_data_dir, model_name, epochs, imgsz, batch, lr,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )

    # 3g. nnU-Net auto-configure pipeline
    elif engine == "nnU-Net":
        med_data_dir = os.path.join(tmpdir, "nnunet_data")
        if not os.path.isdir(os.path.join(data_dir, "images")):
            json_data = _find_ls_json()
            if json_data:
                print("[nnunet] Auto-converting LS annotations → images/ + masks/")
                prepare_segmentation_masks_from_json(json_data, med_data_dir, ls_url, ls_token)
            else:
                med_data_dir = data_dir
        else:
            med_data_dir = data_dir
        train_nnunet(
            med_data_dir, model_name, epochs, imgsz,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )

    # 3d. Self-supervised contrastive pre-training (SimCLR-style)
    elif training_type == "self-supervised" and engine not in ("Anomalib",):
        json_data = _find_ls_json()
        ssl_img_dir = os.path.join(tmpdir, "ssl_images")
        os.makedirs(ssl_img_dir, exist_ok=True)
        if json_data:
            import base64 as _b64ssl
            import requests as _rqssl
            for task in json_data:
                image_url = task.get("data", {}).get("image", "")
                if not image_url: continue
                tid = task.get("id", "x")
                dst = os.path.join(ssl_img_dir, f"task_{tid}.jpg")
                try:
                    if image_url.startswith("data:"):
                        _, data = image_url.split(",", 1)
                        img_b = _b64ssl.b64decode(data)
                    else:
                        url = re.sub(r"https?://[^/]+", ls_url, image_url)
                        resp = _rqssl.get(url, headers={"Authorization": f"Token {ls_token}"}, timeout=60)
                        resp.raise_for_status()
                        img_b = resp.content
                    with open(dst, "wb") as f: f.write(img_b)
                except Exception as e:
                    print(f"[warn] Cannot fetch task {tid}: {e}")
        else:
            import shutil as _sh3
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        _sh3.copy2(os.path.join(root, f), ssl_img_dir)
        train_self_supervised(
            ssl_img_dir, model_name, epochs, imgsz, batch, lr,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )

    # 3e. Ultralytics YOLO path (detection, instance segmentation, ultralytics-cls)
    else:
        yaml_path = None
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith((".yaml", ".yml")):
                    yaml_path = os.path.join(root, f)
                    break
            if yaml_path:
                break

        if not yaml_path:
            images_dir  = os.path.join(data_dir, "images")
            labels_dir  = os.path.join(data_dir, "labels")
            classes_txt = os.path.join(data_dir, "classes.txt")
            if not os.path.exists(images_dir):
                # Check if we have a Label Studio JSON export (images embedded as base64)
                json_data = _find_ls_json()
                if json_data:
                    print(f"[train] Found LS JSON export ({len(json_data)} tasks)")
                    yolo_task = "segment" if training_type == "segmentation" else "detect"
                    yaml_path = prepare_yolo_dataset_from_json(json_data, data_dir, ls_url, ls_token, task=yolo_task)
                else:
                    for root, dirs, files in os.walk(data_dir):
                        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                            images_dir = root; break
            if not yaml_path:
                if os.path.exists(classes_txt):
                    with open(classes_txt) as f:
                        classes = [l.strip() for l in f if l.strip()]
                else:
                    class_ids = set()
                    if os.path.exists(labels_dir):
                        for root, dirs, files in os.walk(labels_dir):
                            for lf in files:
                                if lf.endswith(".txt"):
                                    with open(os.path.join(root, lf)) as fh:
                                        for line in fh:
                                            parts = line.strip().split()
                                            if parts:
                                                try: class_ids.add(int(parts[0]))
                                                except ValueError: pass
                    classes = [f"class_{i}" for i in sorted(class_ids)] or ["object"]
                train_imgs = os.path.join(images_dir, "train")
                val_imgs   = os.path.join(images_dir, "val")
                train_path = train_imgs if os.path.exists(train_imgs) else images_dir
                val_path   = val_imgs   if os.path.exists(val_imgs)   else train_path
                yaml_content = (
                    f"path: {data_dir}\ntrain: {train_path}\nval: {val_path}\n"
                    f"nc: {len(classes)}\nnames: {json.dumps(classes)}\n"
                )
                yaml_path = os.path.join(data_dir, "dataset.yaml")
                with open(yaml_path, "w") as f:
                    f.write(yaml_content)
                print(f"[train] Created dataset.yaml — {len(classes)} classes: {classes}")
        else:
            import yaml as _yaml
            with open(yaml_path) as f:
                y = _yaml.safe_load(f)
            y["path"] = data_dir
            for k in ("train", "val", "test"):
                if k in y and y[k] and not os.path.isabs(str(y[k])):
                    y[k] = os.path.join(data_dir, str(y[k]))
            with open(yaml_path, "w") as f:
                _yaml.dump(y, f)
            print(f"[train] Using existing dataset.yaml: {yaml_path}")

        train_ultralytics(
            yaml_path, data_dir, model_name, training_type,
            epochs, imgsz, batch, lr, optimizer,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )

    print("[train] Complete!")


main()
'''


def _export_ls_to_minio(project_id: int, job_id: str, preferred_fmt: str = "YOLO") -> str | None:
    """Export Label Studio YOLO dataset, upload zip to MinIO, return presigned download URL."""
    try:
        import requests as _req
        import boto3 as _boto3
        from botocore.config import Config as _BConfig

        # Check project exists and has labeled tasks
        proj_resp = _req.get(
            f"{LS_API_URL}/api/projects/{project_id}/",
            headers={"Authorization": f"Token {LS_TOKEN}"},
            timeout=15,
        )
        if proj_resp.status_code != 200:
            raise RuntimeError(
                f"Label Studio project {project_id} not found (HTTP {proj_resp.status_code})"
            )
        proj_data   = proj_resp.json()
        task_count  = proj_data.get("task_number", 0)
        labeled     = proj_data.get("num_tasks_with_annotations", 0)
        proj_title  = proj_data.get("title", str(project_id))

        if task_count == 0:
            raise RuntimeError(
                f"Dataset '{proj_title}' (project {project_id}) has no tasks — "
                "please upload images first"
            )
        if labeled == 0:
            raise RuntimeError(
                f"Dataset '{proj_title}' (project {project_id}) has {task_count} tasks "
                "but none are labeled — please annotate your data first"
            )

        _append_log(job_id, f"[export] Project '{proj_title}': {task_count} tasks, {labeled} labeled")

        # Try YOLO export first, fall back to JSON. If the YOLO zip is
        # suspiciously small (< 16 KB) it almost certainly has no images
        # (LS YOLO export plugin doesn't extract data:image/jpeg;base64,...
        # URIs to files) — fall back to JSON and convert below.
        export_url = f"{LS_API_URL}/api/projects/{project_id}/export"
        fmt_order = [preferred_fmt] + [f for f in ("YOLO", "JSON") if f != preferred_fmt]
        export_data = None
        fmt = None
        for _try_fmt in fmt_order:
            resp = _req.get(
                export_url,
                params={"exportType": _try_fmt},
                headers={"Authorization": f"Token {LS_TOKEN}"},
                timeout=300,
                stream=True,
            )
            if resp.status_code != 200:
                _append_log(job_id, f"[export] {_try_fmt} export failed (HTTP {resp.status_code})")
                continue
            _body = resp.content
            _append_log(job_id, f"[export] Exported as {_try_fmt} ({len(_body)//1024} KB)")
            # Inspect the zip — if it has no images/, we need the JSON path
            # to extract data URI images ourselves.
            _has_images = False
            try:
                import zipfile as _zfchk
                with _zfchk.ZipFile(__import__("io").BytesIO(_body)) as _zf:
                    for _n in _zf.namelist():
                        if _n.startswith("images/") and not _n.endswith("/"):
                            _has_images = True
                            break
            except Exception:
                _has_images = True  # not a zip (e.g. JSON), keep going
            if _try_fmt == "YOLO" and not _has_images:
                _append_log(job_id, f"[export] {_try_fmt} zip has no images/ — falling back to JSON")
                continue
            export_data = _body
            fmt = _try_fmt
            break
        if export_data is None or fmt is None:
            raise RuntimeError(
                f"Label Studio export failed for project {project_id} "
                f"(no format produced a zip with images) — last status: {resp.status_code}"
            )

        bucket = "medimage-datasets"
        key    = f"{job_id}/dataset.zip"

        # If JSON export, build a YOLO-format zip with images extracted
        # from the base64 data URIs in the JSON. Label Studio's YOLO
        # export plugin only writes image files for tasks whose `image`
        # field is a http(s):// URL — tasks with data:image/jpeg;base64,...
        # URIs (synthetic test projects, projects created via the API
        # with inline images) end up with a 2KB zip that has dataset.yaml
        # but no images/, so the training script crashes with "No images
        # found". Converting the JSON export to a proper YOLO zip
        # ourselves bypasses that plugin bug entirely.
        if fmt == "JSON":
            import io as _io, zipfile as _zipfile, json as _json
            json_data = _json.loads(export_data)
            # Pre-decode every data-URI image into a dict so the shared
            # prepare_yolo_dataset_from_json() helper can reuse the bytes
            # without round-tripping back through the Label Studio server.
            # URL-only images are skipped here and fetched by the helper
            # via ls_url/ls_token at the usual endpoint.
            import base64 as _b64
            _image_bytes_map: dict[str, bytes] = {}
            for _task in json_data:
                _img_url = _task.get("data", {}).get("image", "")
                if not _img_url or not _img_url.startswith("data:"):
                    continue
                _stem = f"task_{_task.get('id', len(_image_bytes_map))}.jpg"
                try:
                    _, _b64data = _img_url.split(",", 1)
                    _image_bytes_map[_stem] = _b64.b64decode(_b64data)
                except Exception:
                    pass

            # Delegate to the module-level helper. For YOLO detection we
            # use the standard bbox format (cx cy w h); for segmentation
            # we use polygon points — both honour rectangle→polygon or
            # polygon→polygon natively.
            _yolo_task = "segment" if training_type_for_export == "segmentation" else "detect"
            try:
                export_data = _convert_ls_json_to_yolo_zip(
                    json_data, ls_url, ls_token,
                    task=_yolo_task, image_bytes_map=_image_bytes_map,
                )
            except RuntimeError as _prep_err:
                raise RuntimeError(
                    f"Failed to convert LS JSON → YOLO {_yolo_task} format: {_prep_err}"
                )
            # Re-bundle the raw LS JSON for engines that look for any
            # *.json list-of-tasks manifest (_find_ls_json helper in
            # train_hf_vision / train_pytorch_segmentation).
            import io as _io2, zipfile as _zf2
            _tmp_buf = _io2.BytesIO(export_data)
            _out_buf = _io2.BytesIO()
            with _zf2.ZipFile(_tmp_buf, "r") as _src_zf, _zf2.ZipFile(_out_buf, "w", _zf2.ZIP_DEFLATED) as _out_zf:
                for _item in _src_zf.namelist():
                    _out_zf.writestr(_item, _src_zf.read(_item))
                _out_zf.writestr("annotations.json", json.dumps(json_data, ensure_ascii=False))
            export_data = _out_buf.getvalue()
            _n_imgs = sum(1 for _ in _image_bytes_map)
            # Count labels in zip
            _n_lbls = 0
            with _zf2.ZipFile(_io2.BytesIO(export_data), "r") as _cnt_zf:
                for _n in _cnt_zf.namelist():
                    if _n.startswith("labels/") and _n.endswith(".txt"):
                        _n_lbls += 1
            _classes_seen: list[str] = []
            try:
                with _zf2.ZipFile(_io2.BytesIO(export_data), "r") as _cfg_zf:
                    if "dataset.yaml" in _cfg_zf.namelist():
                        import yaml as _yl
                        _yd = _yl.safe_load(_cfg_zf.read("dataset.yaml"))
                        if isinstance(_yd, dict) and "names" in _yd:
                            _classes_seen = list(_yd["names"])
            except Exception:
                pass
            _append_log(job_id, f"[export] JSON→YOLO ({_yolo_task}) conversion: {_n_imgs} images, {_n_lbls} labels, {len(_classes_seen)} classes")
            _append_log(job_id, f"[export] Wrapped converted YOLO into zip ({len(export_data)//1024} KB)")
            fmt = "YOLO"  # downstream code already handles YOLO zip

        # Upload to MinIO (internal URL)
        s3 = _boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_BConfig(signature_version="s3v4"),
        )
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key=key, Body=export_data)

        # Generate presigned URL using a host the consumer (the Ray worker)
        # can actually resolve. The S3v4 signature includes the Host header
        # as a signed component, so the host used here MUST match the host
        # the worker will hit — otherwise MinIO rejects the request with
        # SignatureDoesNotMatch. MINIO_PUBLIC_URL is the *internal* docker
        # service name (http://minio:9000) by default, which the worker
        # cannot resolve, so we go through _resolve_minio_url_for_ray().
        ray_minio_url = _resolve_minio_url_for_ray()
        s3_pub = _boto3.client(
            "s3",
            endpoint_url=ray_minio_url,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_BConfig(signature_version="s3v4"),
        )
        url = s3_pub.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=86400,
        )
        _append_log(job_id, f"[export] Dataset staged in MinIO (presigned URL generated)")
        return url

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Dataset export error: {e}") from e


def _run_training_remote(script_b64: str, env_vars: dict) -> dict:
    """Placeholder — actual implementation defined inline inside _run_on_ray_cluster
    to avoid cloudpickle serializing it as a reference to the 'main' module."""
    raise NotImplementedError


def _run_on_ray_cluster(job: dict) -> None:
    """Export dataset, submit real YOLO training task to Ray cluster, poll until done."""
    import os as _os
    _os.environ["RAY_ADDRESS"] = ""
    _os.environ["RAY_ENABLE_AUTO_CONNECT"] = "0"
    import ray

    _os.environ["RAY_IGNORE_VERSION_MISMATCH"] = "1"

    # Serialize Ray session access to prevent concurrent init/shutdown conflicts
    if not hasattr(_run_on_ray_cluster, "_lock"):
        _run_on_ray_cluster._lock = _threading.Lock()

    job_id        = job["id"]
    cluster       = job.get("cluster", "ray")
    project_id    = int(job.get("project_id") or 0)
    training_type = job.get("training_type", "detection")

    # Resolve Ray URL (http://host:8265 → ray://host:10001)
    if cluster == "modal":
        modal_ray_url = _modal_state.get("ray_url")
        if not modal_ray_url:
            raise RuntimeError("Modal cluster is not running or ray_url is unavailable")
        http_url = modal_ray_url
    else:
        http_url = RAY_URL

    host      = http_url.split("://", 1)[-1].split(":")[0]
    ray_addr  = f"ray://{host}:10001"
    _set_step(job_id, "connect")
    _append_log(job_id, f"[cluster] Connecting to Ray at {ray_addr}")

    # Export dataset from Label Studio. Skip only when the engine+training
    # type doesn't need image data at all (LLM with text_dataset, self-supervised
    # paths that fetch from a different source, etc.). VLM is special: it can
    # use EITHER a JSONL text_dataset OR the project's image+annotation zip
    # — so we only skip the export when the job has a text_dataset attached.
    engine = job.get("engine", "Ultralytics")
    training_type_for_export = job.get("training_type", "detection")
    _job_text_dataset = job.get("text_dataset", "") or ""
    _is_pure_llm = (engine == "Unsloth" and training_type_for_export == "llm-text")
    _is_vlm_with_text = (training_type_for_export == "vlm-finetune" and bool(_job_text_dataset))
    if _is_pure_llm or _is_vlm_with_text:
        dataset_url = ""
        _append_log(job_id, "[cluster] Using text_dataset — skipping LS export")
    else:
        _set_step(job_id, "export")
        _append_log(job_id, f"[cluster] Exporting dataset from LS project {project_id} ...")
        # SMP segmentation, self-supervised, HuggingFace vision, and VLM-without-
        # text-dataset all need JSON so we can download images from MinIO.
        _needs_json = (
            engine in ("Segmentation Models PyTorch", "HuggingFace", "PyTorch", "PyTorch+TIMM", "TorchVision", "TIMM", "MONAI")
            or training_type_for_export == "self-supervised"
            or training_type_for_export == "vlm-finetune"
            or training_type_for_export == "segmentation"  # all segmentation needs JSON
            or training_type_for_export == "classification"  # classification needs LS JSON for image + label lookup
        )
        _export_fmt = "JSON" if _needs_json else "YOLO"
        dataset_url = _export_ls_to_minio(project_id, job_id, preferred_fmt=_export_fmt)
        _append_log(job_id, "[cluster] Dataset exported and staged in MinIO ✓")

    weights_bucket = "medimage-weights"
    weights_key    = f"{job_id}/best.pt"
    script_b64     = base64.b64encode(_VISION_TRAIN_SCRIPT.encode()).decode()
    # Resolve the submitter's saved HuggingFace token. If they haven't saved
    # one in their profile, fall back to whatever HF_TOKEN is set on the API
    # container's environment (operator-level default for shared infra).
    _job_user_id   = job.get("user_id", "") or ""
    _user_hf_tok   = _load_user_hf_token(_job_user_id)
    _hf_token      = _user_hf_tok or os.getenv("HF_TOKEN", "") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    if _hf_token:
        # Also export into the API process so any *synchronous* HF calls
        # in the worker (e.g. dataset tree fetches) inherit the token.
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    env_vars = {
        "ENGINE":          engine,
        "TRAINING_TYPE":   training_type,
        "MODEL_NAME":      job.get("model_name", "yolov8n.pt"),
        "EPOCHS":          str(job.get("epochs", 10)),
        "IMGSZ":           str(job.get("imgsz", 640)),
        "BATCH_SIZE":      str(job.get("batch_size", 16)),
        "LR":              str(job.get("learning_rate", 0.001)),
        "OPTIMIZER":       (job.get("optimizer") or "AdamW").capitalize(),
        "JOB_ID":          job_id,
        "PROJECT_ID":      str(project_id),
        "DATASET_URL":     dataset_url,
        "WEIGHTS_BUCKET":  weights_bucket,
        "WEIGHTS_KEY":     weights_key,
        "MINIO_URL":       _resolve_minio_url_for_ray(),
        "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
        "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
        "LS_PUBLIC_URL":   LS_PUBLIC_URL,
        "LS_TOKEN":        LS_TOKEN,
        "HF_TOKEN":        _hf_token,
        "HUGGING_FACE_HUB_TOKEN": _hf_token,
        # LLM-specific — resolve the text_dataset id to a real local path
        # so load_dataset() can read it. The training script first tries
        # HF hub by that name and falls back to local JSONL if it fails,
        # so passing the full path is the cleanest, most reliable option.
        "TEXT_DATASET":    _resolve_text_dataset_path(job.get("text_dataset", "")),
        "LORA_RANK":       str(job.get("lora_rank", 16)),
        "QUANTIZATION":    job.get("quantization", "4bit"),
        "MAX_SEQ_LEN":     str(job.get("max_seq_len", 2048)),
        "CHAT_TEMPLATE":   job.get("chat_template", "chatml"),
        "GRAD_ACCUM":      str(job.get("grad_accum", 4)),
    }

    # ── VLM / LLM dispatch ─────────────────────────────────────────────────
    # VLM (training_type=vlm-finetune) and LLM (engine=Unsloth) go through
    # vlm_llm_train.py — a separate module with its own train functions,
    # its own torch pre-flight, and its own triton/sklearn import-stub
    # installer. The Ray worker image bakes the VLM/LLM deps in once at
    # build time (transformers / peft / trl / bitsandbytes / accelerate /
    # datasets / unsloth / numpy<2.0), so this path doesn't pip-install
    # anything per-job. Keeps the VLM/LLM code isolated from the DL path
    # below so changes to one don't affect the other.
    #
    # main.py is the dispatcher only — it does NOT import vlm_llm_train.
    # The DL module is shipped to the worker via Ray's runtime_env
    # (py_modules), so workers get the source without baking it into the
    # image, and edits to vlm_llm_train.py are picked up on the next
    # task without redeploying main.py.
    _is_vlm_llm = (engine == "Unsloth") or (training_type == "vlm-finetune")
    if _is_vlm_llm:
        if not _VLM_LLM_MODULE_PATH:
            raise RuntimeError(
                "vlm_llm_train.py not found next to main.py — the VLM/LLM "
                "training module is required for this job. Make sure "
                "vlm_llm_train.py is in the backend/ directory."
            )

        # ── Submit via Ray Jobs REST API ──────────────────────────────────
        # Ray 2.55.0's client server has a bug where _runtime_env_agent_address
        # is None, making ray.init(address="ray://...") crash with:
        #   TypeError: object of type 'NoneType' has no len()
        # The Jobs API (dashboard REST) works fine and doesn't need ray.init().
        #
        # Shipping vlm_llm_train.py: runtime_env py_modules is validated on
        # the dashboard node (which can't see /app/), so we embed the module
        # as base64 in a wrapper script. The worker decodes, writes to /tmp,
        # imports, and runs. Pip packages are specified via runtime_env["pip"].
        import requests as _ray_req
        import base64 as _b64

        _ray_submit_url = f"{http_url}/api/jobs/"

        # Read vlm_llm_train.py and base64-encode for embedding in entrypoint
        with open(_VLM_LLM_MODULE_PATH, "rb") as _f:
            _module_b64 = _b64.b64encode(_f.read()).decode("ascii")

        # The entrypoint is a self-contained Python script that:
        #  1) Decodes vlm_llm_train.py from an inline base64 string
        #  2) Writes it to /tmp
        #  3) Imports and runs it, calling _vlm_setup_hook() then main()
        #
        # This avoids all working_dir / runtime_env shipping issues:
        #   - data: URIs aren't supported by the Jobs REST API
        #   - http:// URIs are rejected (only https:// is allowed)
        #   - s3:// URIs fail because env_vars aren't set during working_dir download
        # By embedding everything in the entrypoint command itself, we bypass
        # all of Ray's runtime_env validation and file-shipping mechanisms.
        _entrypoint_script = (
            "import base64,os,sys,tempfile;"
            "mod_b64=" + repr(_module_b64) + ";"
            "mod_path=os.path.join(tempfile.gettempdir(),'vlm_llm_train.py');"
            "open(mod_path,'wb').write(base64.b64decode(mod_b64));"
            "sys.path.insert(0,tempfile.gettempdir());"
            "from vlm_llm_train import _vlm_setup_hook,main;"
            "_vlm_setup_hook();"
            "main()"
        )
        _entrypoint = f"python3 -c {_entrypoint_script!r}"

        # Build clean runtime_env: env_vars only. All Python deps are
        # pre-installed on the Ray cluster image, so the runtime_env.pip
        # list is empty — telling Ray to skip its own pip resolver (which
        # is what was causing VLM/LLM jobs to sit PENDING while the head
        # node spent several minutes resolving and installing the same
        # packages we already have on disk). No working_dir / py_modules /
        # setup_hook — everything is embedded in the entrypoint command.
        _re_env = {
            "env_vars": {"RAY_ADDRESS": "", "RAY_RUNTIME_ENV_HOOK_DISABLED": "1", "RAY_ENABLE_AUTO_CONNECT": "0"},
            "pip": [],
        }
        if _VLM_LLM_RUNTIME_ENV and "env_vars" in _VLM_LLM_RUNTIME_ENV:
            for _ek, _ev in _VLM_LLM_RUNTIME_ENV["env_vars"].items():
                _re_env["env_vars"][_ek] = str(_ev)
        for _k, _v in env_vars.items():
            _re_env["env_vars"][_k] = str(_v)

        _num_gpus = int(job.get("num_gpus") or 1)
        _payload = {
            "entrypoint": _entrypoint,
            "runtime_env": _re_env,
        }
        # Pass GPU count via env var so the training script can use
        # it for ray.remote(num_gpus=...) or os.environ lookups.
        # NOTE: Do NOT use "entrypoint_num_gpus" — Ray 2.55.0 silently
        # PENDINGs jobs that request GPUs via that field (never allocates
        # the head).  GPU scheduling must happen inside the script via
        # ray.remote(num_gpus=...) or torch.cuda .
        _payload["runtime_env"]["env_vars"]["MEDIMAGE_NUM_GPUS"] = str(_num_gpus)

        _set_step(job_id, "submit")
        _append_log(job_id, f"[cluster] Submitting VLM/LLM job via Ray Jobs API ...")
        try:
            _resp = _ray_req.post(_ray_submit_url, json=_payload, timeout=30)
            _resp.raise_for_status()
            _job_data = _resp.json()
            _ray_job_id = _job_data.get("submission_id") or _job_data.get("job_id")
            _append_log(job_id, f"[cluster] Ray job submitted: {_ray_job_id}")
            with get_db() as _c:
                _c.execute("UPDATE jobs SET ray_submission_id=? WHERE id=?", (_ray_job_id, job_id))
                _c.commit()
        except Exception as _e:
            _append_log(job_id, f"[cluster] Ray job submission failed: {_e}")
            raise

        # Poll for job completion via the dashboard REST API
        _set_step(job_id, "training")
        _last_status = ""
        elapsed = 0
        poll_sec = 15
        while True:
            time.sleep(poll_sec)
            elapsed += poll_sec
            # If the user cancelled the job in the UI, the DB row will be
            # 'error' but Ray may still report PENDING (the autoscaler
            # never allocated a slot). Stop polling, signal stop to Ray,
            # and let the surrounding code surface the DB error.
            try:
                with get_db() as _c:
                    _db_row = _c.execute(
                        "SELECT status FROM jobs WHERE id=?", (job_id,)
                    ).fetchone()
                if _db_row and _db_row["status"] not in ("running", "queued"):
                    _append_log(job_id, f"[cluster] DB status is '{_db_row['status']}' — abandoning Ray poll")
                    try:
                        _ray_req.post(f"{http_url}/api/jobs/{_ray_job_id}/stop", timeout=5)
                    except Exception:
                        pass
                    _ray_status = "STOPPED"
                    break
            except Exception:
                pass
            try:
                _status_resp = _ray_req.get(f"{http_url}/api/jobs/{_ray_job_id}", timeout=10)
                _status_resp.raise_for_status()
                _ray_status = _status_resp.json().get("status", "UNKNOWN")
            except Exception as _e:
                _append_log(job_id, f"[cluster] Status poll failed: {_e}")
                _ray_status = _last_status

            if _ray_status != _last_status:
                _append_log(job_id, f"[cluster] Ray job status: {_ray_status}")
                _last_status = _ray_status

            if _ray_status in ("SUCCEEDED", "FAILED", "STOPPED"):
                break

            _append_log(job_id, f"[cluster] Training in progress ({elapsed // 60}m {elapsed % 60}s) ...")

        # Ray cluster: flush stdout/stderr of the finished job to the
        # dashboard before we try to read it back. Without this gap
        # the /logs endpoint returns an empty body even though the
        # job is reported as SUCCEEDED.
        if _ray_status == "SUCCEEDED":
            time.sleep(8)

        # Fetch logs and parse weights path from WEIGHTS_UPLOADED lines.
        # The Ray dashboard can take a few seconds after status flips
        # to SUCCEEDED before the job's stdout is queryable, so we
        # retry a few times before giving up.
        _weights_path = None
        _all_log_lines = []
        # Fallback: even if the dashboard log endpoint never returns
        # the WEIGHTS_UPLOADED line (cluster-side buffering, or the
        # log file was GC'd before the line made it through), the
        # upload_to_minio() call always writes to the standard
        # {WEIGHTS_BUCKET}/{WEIGHTS_KEY} pair, so we can construct the
        # path directly.
        _expected_weights = f"s3://{weights_bucket}/{weights_key}"
        for _log_attempt in range(15):
            try:
                _logs_resp = _ray_req.get(f"{http_url}/api/jobs/{_ray_job_id}/logs", timeout=30)
                # Parse JSON wrapper first — Ray returns {"logs": "..."}
                # so _logs_resp.text length is misleading (e.g. {"logs":""}
                # is 12 bytes but contains zero log lines).
                _logs_str = ""
                if _logs_resp.ok:
                    try:
                        _logs_str = _logs_resp.json().get("logs", "") or ""
                    except Exception:
                        _logs_str = _logs_resp.text if _logs_resp.text else ""
                _append_log(job_id, f"[cluster] log attempt {_log_attempt+1}/15: ok={_logs_resp.ok} json_len={len(_logs_str)}")
                if _logs_str.strip():
                    _all_log_lines = []
                    for _line in _logs_str.splitlines():
                        _line_s = _line.strip()
                        if _line_s:
                            _all_log_lines.append(_line_s)
                            _append_log(job_id, _line_s)
                            if "WEIGHTS_UPLOADED" in _line_s:
                                _payload = _line_s.split("WEIGHTS_UPLOADED", 1)[1].lstrip(":").strip()
                                if not _payload:
                                    continue
                                if _payload.startswith("s3://"):
                                    _weights_path = _payload
                                elif "/" in _payload and _payload.count("/") == 1:
                                    # 2-part path like "jobid/best.pt" — assume
                                    # the default medimage-weights bucket
                                    # because the inline _VISION_TRAIN_SCRIPT's
                                    # `print(f"WEIGHTS_UPLOADED:{w_key}")` (line
                                    # ~2907) only emits the key, not the full URI.
                                    _weights_path = f"s3://medimage-weights/{_payload}"
                                else:
                                    _weights_path = f"s3://medimage-weights/{_payload}"
                    if _weights_path:
                        break
                _append_log(job_id, f"[cluster] Log fetch attempt {_log_attempt+1}/15 empty, retrying...")
                time.sleep(3)
            except Exception as _e:
                _append_log(job_id, f"[cluster] Could not fetch logs (attempt {_log_attempt+1}): {_e}")
                time.sleep(3)

        if _ray_status != "SUCCEEDED":
            raise RuntimeError(f"Ray job {_ray_job_id} ended with status: {_ray_status}")

        # If we couldn't parse the WEIGHTS_UPLOADED line, fall back to
        # the bucket/key pair the worker was told to use. This is the
        # correct path in the common case because upload_to_minio()
        # always writes to exactly that S3 location.
        if not _weights_path:
            _append_log(job_id, f"[cluster] Falling back to expected weights path: {_expected_weights}")
            _weights_path = _expected_weights

        _set_step(job_id, "saving")
        if _weights_path:
            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET s3_weights_path = ? WHERE id = ?",
                    (_weights_path, job_id),
                )
                conn.commit()
        # Verify the weights actually exist in MinIO before declaring success.
        # Ray SUCCEEDED + missing object = "phantom" success (training script
        # returned without uploading). See _verify_weights_uploaded docstring.
        if _weights_path:
            _ok, _msg = _verify_weights_uploaded(_weights_path)
            _append_log(job_id, f"[cluster] weight check: {_msg}")
            if not _ok:
                raise RuntimeError(
                    f"Training script returned 0 but weights are missing in MinIO: {_msg}. "
                    f"This usually means the training script exited without uploading — "
                    f"check the captured stdout above for the actual failure."
                )
        _append_log(job_id, f"✓ Training complete — weights: {_weights_path or '(not found)'}")
        _set_step(job_id, "done")
        _set_status(job_id, "completed")
        return

    # ── Deep-learning dispatch ─────────────────────────────────────────────
    # pip packages depend on engine
    if engine in ("PyTorch", "PyTorch+TIMM", "TIMM"):
        # torchxrayvision needed for medical DenseNet ("densenet121-res224-all"
        # etc.) which TIMM doesn't ship. ~50MB wheel, only used by that branch.
        pip_pkgs = ["timm>=0.9", "torchxrayvision>=1.2", "boto3>=1.34", "requests", "Pillow"]
    elif engine == "TorchVision" and training_type == "classification":
        pip_pkgs = ["timm>=0.9", "boto3>=1.34", "requests", "Pillow"]
    elif engine in ("Segmentation Models PyTorch", "TorchVision"):
        pip_pkgs = ["segmentation-models-pytorch", "boto3>=1.34", "requests", "Pillow"]
    elif engine == "Anomalib":
        pip_pkgs = ["boto3>=1.34", "requests", "Pillow", "scipy"]
    elif engine == "MONAI":
        pip_pkgs = ["monai[all]>=1.3", "nibabel", "boto3>=1.34", "Pillow"]
    elif engine == "MedSAM":
        pip_pkgs = ["segment-anything", "monai>=1.3", "nibabel", "boto3>=1.34", "Pillow"]
    elif engine == "nnU-Net":
        pip_pkgs = ["nnunetv2", "nibabel", "boto3>=1.34", "Pillow"]
    elif engine == "HuggingFace":
        # numpy<2.0 keeps ABI compatibility with the cluster's pre-installed
        # pandas (built against numpy 1.x dtype size 96). Without the pin, the
        # transformers/datasets deps pull numpy 2.x (size 88) and break
        # pandas._libs.interval import — which crashes Trainer import via the
        # transformers -> sklearn -> pandas chain before training can even start.
        pip_pkgs = ["transformers>=4.40,<5.0", "peft>=0.10", "accelerate>=0.28",
                    "bitsandbytes>=0.43", "datasets>=2.18",
                    "boto3>=1.34", "Pillow", "numpy<2.0"]
    elif training_type == "self-supervised":
        pip_pkgs = ["boto3>=1.34", "requests", "Pillow"]  # torch/torchvision pre-installed
    else:
        # ultralytics is pre-installed on Ray cluster — do NOT reinstall (causes numpy binary conflict)
        # Pin numpy<2.0 — cluster has pandas compiled against numpy 1.x ABI
        pip_pkgs = ["boto3>=1.34", "pyyaml", "requests", "ultralytics",
                    "opencv-python-headless", "numpy<2.0"]

    # ── Submit deep-learning job via Ray Jobs API (avoids Ray Client port) ──
    # VLM/LLM jobs use the dispatch path above (line 2996+) — don't touch them.
    # Only Mask R-CNN, YOLO, MONAI, etc. land here, and they now go through
    # the dashboard REST API instead of ray.init(address="ray://...").
    import base64 as _b64lib, requests as _ray_req
    # All Python deps are now pre-installed on the Ray cluster image, so the
    # wrapper just sets env vars and exec()s the training script — no more
    # subprocess.run(pip install) calls (those were the cause of jobs
    # sitting PENDING for several minutes while the head node fought over
    # pip resolver locks). Opencv swap is kept as an idempotent safety net
    # in case a future pip install pulls opencv-python back in.
    _wrapper = (
        "import os, sys, subprocess, base64, time\n"
        f"_env = {env_vars!r}\n"
        "# Training script can arrive in 2 ways:\n"
        "#  (1) MEDIMAGE_SCRIPT_URL — wrapper downloads from API cache (used\n"
        "#      for big scripts > 60KB b64 that wouldn't fit in env vars).\n"
        "#  (2) MEDIMAGE_TRAIN_SCRIPT_B64_0, _1, ... chunks — back-compat\n"
        "#      path for small scripts.\n"
        "_chunks = []\n"
        "_idx = 0\n"
        "while True:\n"
        "    _part = os.environ.get(f'MEDIMAGE_TRAIN_SCRIPT_B64_{_idx}')\n"
        "    if _part is None:\n"
        "        break\n"
        "    _chunks.append(_part)\n"
        "    _idx += 1\n"
        "_b64 = ''.join(_chunks) if _chunks else os.environ.get('MEDIMAGE_TRAIN_SCRIPT_B64', '')\n"
        "print(f'[wrapper] python={sys.executable} pid={os.getpid()}', flush=True)\n"
        "print(f'[wrapper] env keys={len(_env)} script_chunks={len(_chunks)} script_b64_len={len(_b64)}', flush=True)\n"
        "print(f'[wrapper] JOB_ID={os.environ.get(\"JOB_ID\", \"?\")}', flush=True)\n"
        "# Idempotent opencv fix: if opencv-python got pulled back in (it needs\n"
        "# libGL.so.1, missing on the cluster), swap it for opencv-python-headless.\n"
        "# If headless is already there, this is a no-op (subprocess rc=0, fast).\n"
        "try:\n"
        "    import cv2  # noqa\n"
        "    print('[opencv-fix] cv2 import ok, version:', cv2.__version__, flush=True)\n"
        "except Exception as _e:\n"
        "    print(f'[opencv-fix] cv2 import failed ({_e!r}) — swapping opencv-python -> opencv-python-headless', flush=True)\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'opencv-python', 'opencv-contrib-python'], capture_output=True, text=True)\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall', '--no-deps', 'opencv-python-headless'], capture_output=True, text=True)\n"
        "    import cv2  # noqa\n"
        "    print(f'[opencv-fix] cv2 ok after swap: {cv2.__version__}', flush=True)\n"
        "for k, v in _env.items(): os.environ[k] = v\n"
        "# Resolve the training script. Two delivery modes are supported:\n"
        "#  (a) MinIO URL — API uploads the script to MinIO and passes the\n"
        "#      URL. Used when the script is too big to fit in env vars\n"
        "#      (Linux ARG_MAX is ~128KB, _VISION_TRAIN_SCRIPT is ~130KB\n"
        "#      b64). MinIO is used because the API itself isn't reachable\n"
        "#      from the Ray cluster's network but MinIO is — it already\n"
        "#      has the dataset zip + weights keys.\n"
        "#  (b) Inline env var(s) — used for short scripts. Kept for back-compat.\n"
        "_job_id = os.environ.get('JOB_ID', 'unknown')\n"
        "_script_url = os.environ.get('MEDIMAGE_SCRIPT_URL', '')\n"
        "if _script_url:\n"
        "    print(f'[wrapper] downloading training script from {_script_url}', flush=True)\n"
        "    import urllib.request\n"
        "    with urllib.request.urlopen(_script_url, timeout=60) as _r:\n"
        "        _script_src = _r.read().decode('utf-8')\n"
        "    print(f'[wrapper] downloaded {len(_script_src)} chars', flush=True)\n"
        "else:\n"
        "    _script_src = base64.b64decode(_b64).decode('utf-8')\n"
        "from io import StringIO\n"
        "_cap = StringIO()\n"
        "_so, _se = sys.stdout, sys.stderr\n"
        "sys.stdout = sys.stderr = _cap\n"
        "print('[wrapper] exec training script ...', flush=True)\n"
        "try:\n"
        "    exec(_script_src, {'__name__': '__main__'})\n"
        "except SystemExit as _e:\n"
        "    if _e.code not in (None, 0):\n"
        "        sys.stdout = _so; sys.stderr = _se\n"
        "        raise RuntimeError(f'Script exited {_e.code}\\n{_cap.getvalue()}')\n"
        "except BaseException as _e:\n"
        "    sys.stdout = _so; sys.stderr = _se\n"
        "    raise RuntimeError(f'Script crashed: {type(_e).__name__}: {_e}\\n{_cap.getvalue()}')\n"
        "finally:\n"
        "    sys.stdout, sys.stderr = _so, _se\n"
        "print(_cap.getvalue())\n"
    )
    _wrapper_b64 = _b64lib.b64encode(_wrapper.encode()).decode()
    # Upload the training script to MinIO and pass a presigned URL to the
    # wrapper. The base64 of the ~97KB _VISION_TRAIN_SCRIPT is ~130KB which
    # exceeds Linux's per-payload env var limit (~128KB = ARG_MAX/2). When
    # that limit is hit, Ray silently PENDINGS the job forever with no logs.
    # We can't GET the script from the API (the API isn't exposed to the
    # Ray cluster's network) but MinIO is — we already use it for the
    # dataset zip + weights. Bucket: "medimage-scripts" (auto-created).
    # The presigned URL is generated against MINIO_PUBLIC_URL so the
    # signature matches the hostname the Ray cluster uses to GET it.
    import tempfile as _tf
    _script_text = base64.b64decode(script_b64).decode("utf-8")
    _tmp = _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    _tmp.write(_script_text)
    _tmp.close()
    _script_bucket = "medimage-scripts"
    _script_key = f"{job_id}/train_script.py"
    _script_presigned_url = ""
    try:
        # Upload via the docker-internal endpoint (always reachable from
        # inside the API container).
        upload_to_minio(
            _tmp.name, _script_bucket, _script_key,
            MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
        )
        # Sign the presigned URL using the public endpoint so the signature
        # matches the host the Ray cluster will use to GET it.
        import boto3 as _boto
        from botocore.config import Config as _Cfg
        _s3_pub = _boto.client(
            "s3",
            endpoint_url=MINIO_PUBLIC_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_Cfg(signature_version="s3v4"),
        )
        _script_presigned_url = _s3_pub.generate_presigned_url(
            "get_object",
            Params={"Bucket": _script_bucket, "Key": _script_key},
            ExpiresIn=3600,
        )
    except Exception as _e:
        _append_log(job_id, f"[cluster] WARN: could not upload training script to MinIO: {_e}")
    finally:
        try:
            os.unlink(_tmp.name)
        except Exception:
            pass
    _env_for_job = dict(env_vars)
    # Don't pass the script in env vars — just point at the MinIO URL.
    _env_for_job.pop("MEDIMAGE_TRAIN_SCRIPT_B64", None)
    _env_for_job["MEDIMAGE_SCRIPT_URL"] = _script_presigned_url
    _env_for_job["MEDIMAGE_SCRIPT_BUCKET"] = _script_bucket
    _env_for_job["MEDIMAGE_SCRIPT_KEY"] = _script_key
    _num_gpus = int(job.get("num_gpus") or 1)
    _env_for_job["MEDIMAGE_NUM_GPUS"] = str(_num_gpus)
    _deploy_payload = {
        "entrypoint": f"bash -c 'echo {_wrapper_b64} | base64 -d > /tmp/dl_train.py && python3 /tmp/dl_train.py'",
        "runtime_env": {"env_vars": _env_for_job},
    }
    try:
        _resp = _ray_req.post(f"{http_url}/api/jobs/", json=_deploy_payload, timeout=30)
        _resp.raise_for_status()
        _ray_job_id = _resp.json().get("submission_id") or _resp.json().get("job_id")
        _append_log(job_id, f"[cluster] DL training job submitted: {_ray_job_id}")
        with get_db() as _c:
            _c.execute("UPDATE jobs SET ray_submission_id=? WHERE id=?", (_ray_job_id, job_id))
            _c.commit()
    except Exception as _e:
        _append_log(job_id, f"[cluster] DL job submission failed: {_e}")
        raise

    _set_step(job_id, "training")
    _last_status = ""
    elapsed = 0
    poll_sec = 15
    while True:
        time.sleep(poll_sec)
        elapsed += poll_sec
        # Check if the job was cancelled in the DB (DELETE endpoint sets
        # status to 'error' with the cancellation message). The Ray job
        # may still report PENDING/RUNNING because the autoscaler
        # hasn't reaped it yet, but the user wants the training thread
        # to stop polling and exit so the row doesn't keep "training in
        # progress" forever.
        try:
            with get_db() as _c:
                _db_row = _c.execute(
                    "SELECT status FROM jobs WHERE id=?", (job_id,)
                ).fetchone()
            if _db_row and _db_row["status"] not in ("running", "queued"):
                # DB no longer says the job is live — assume cancellation
                # or external error. Stop the Ray job, exit the loop, and
                # let the surrounding handler surface the DB error.
                _append_log(job_id, f"[cluster] DB status is '{_db_row['status']}' — abandoning Ray poll")
                try:
                    _req_stop = _ray_req.post(
                        f"{http_url}/api/jobs/{_ray_job_id}/stop", timeout=5
                    )
                    _append_log(job_id, f"[cluster] best-effort stop: {_req_stop.status_code}")
                except Exception:
                    pass
                _ray_status = "STOPPED"
                break
        except Exception:
            pass
        try:
            _status_resp = _ray_req.get(f"{http_url}/api/jobs/{_ray_job_id}", timeout=10)
            _status_resp.raise_for_status()
            _ray_status = _status_resp.json().get("status", "UNKNOWN")
        except Exception as _e:
            _append_log(job_id, f"[cluster] Status poll failed: {_e}")
            _ray_status = _last_status
        if _ray_status != _last_status:
            _append_log(job_id, f"[cluster] Ray job status: {_ray_status}")
            _last_status = _ray_status
        if _ray_status in ("SUCCEEDED", "FAILED", "STOPPED"):
            break
        _append_log(job_id, f"[cluster] Training in progress ({elapsed // 60}m {elapsed % 60}s) ...")

    if _ray_status == "SUCCEEDED":
        time.sleep(8)

    _weights_path = None
    _expected_weights = f"s3://{weights_bucket}/{weights_key}"
    for _log_attempt in range(15):
        try:
            _logs_resp = _ray_req.get(f"{http_url}/api/jobs/{_ray_job_id}/logs", timeout=30)
            # Parse JSON wrapper first — Ray returns {"logs": "..."}
            # so raw text length is misleading (e.g. {"logs":""} = 12 bytes but 0 lines).
            _logs_str = ""
            if _logs_resp.ok:
                try:
                    _logs_str = _logs_resp.json().get("logs", "") or ""
                except Exception:
                    _logs_str = _logs_resp.text if _logs_resp.text else ""
            _append_log(job_id, f"[cluster] log attempt {_log_attempt+1}/15: ok={_logs_resp.ok} json_len={len(_logs_str)}")
            if _logs_str.strip():
                for _line in _logs_str.splitlines():
                    _line_s = _line.strip()
                    if not _line_s: continue
                    if _line_s.startswith("Running entrypoint") or "Runtime env is" in _line_s or "Connecting to existing" in _line_s or "Connected to Ray" in _line_s or "Using address" in _line_s:
                        continue
                    _append_log(job_id, _line_s)
                    if "WEIGHTS_UPLOADED" in _line_s:
                        _payload = _line_s.split("WEIGHTS_UPLOADED", 1)[1].lstrip(":").strip()
                        if not _payload:
                            continue
                        if _payload.startswith("s3://"):
                            _weights_path = _payload
                        elif "/" in _payload and _payload.count("/") == 1:
                            _weights_path = f"s3://{_payload}"
                        else:
                            _weights_path = f"s3://medimage-weights/{_payload}"
                if _weights_path:
                    break
            _append_log(job_id, f"[cluster] Log fetch attempt {_log_attempt+1}/15 empty, retrying...")
            time.sleep(3)
        except Exception as _e:
            _append_log(job_id, f"[cluster] Could not fetch logs (attempt {_log_attempt+1}): {_e}")
            time.sleep(3)

    if _ray_status != "SUCCEEDED":
        raise RuntimeError(f"DL training job {_ray_job_id} ended with status: {_ray_status}")

    if not _weights_path:
        _append_log(job_id, f"[cluster] Falling back to expected weights path: {_expected_weights}")
        _weights_path = _expected_weights

    # Save weights path to DB
    if _weights_path:
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET s3_weights_path = ? WHERE id = ?",
                (_weights_path, job_id),
            )
            conn.commit()
    # Verify the weights actually exist in MinIO before declaring success.
    # See _verify_weights_uploaded docstring — Ray SUCCEEDED ≠ training done.
    if _weights_path:
        _ok, _msg = _verify_weights_uploaded(_weights_path)
        _append_log(job_id, f"[cluster] weight check: {_msg}")
        if not _ok:
            raise RuntimeError(
                f"Training script returned 0 but weights are missing in MinIO: {_msg}. "
                f"This usually means the training script exited without uploading — "
                f"check the captured stdout above for the actual failure."
            )
    _append_log(job_id, f"✓ Training complete — weights: {_weights_path or '(not found)'}")
    _set_step(job_id, "done")
    _set_status(job_id, "completed")
    return
    _pip_pkgs_for_worker = pip_pkgs
    def _run_training_inline(script_b64_arg: str, env_vars_arg: dict) -> dict:
        """Inline runner for the deep-learning path (YOLO, MONAI, nnU-Net,
        HuggingFace vision, etc.). Installs the DL-specific pip_pkgs,
        force-reinstalls numpy<2.0 as a safety net against stale 2.x
        wheels from a previous run, and execs _VISION_TRAIN_SCRIPT.

        VLM/LLM jobs go through a different code path (vlm_llm_train.run)
        which has its own torch pre-flight and import-stub installer.
        This function is only called when (engine, training_type) is in
        the deep-learning set — see the dispatch in _run_on_ray_cluster.
        """
        import os as _os, base64 as _b64, sys as _sys, subprocess as _sp
        from io import StringIO as _StringIO
        # Force numpy to a 1.x version FIRST. Without --force-reinstall, pip's
        # resolver can leave a numpy 2.x in place from a previous run (since
        # the other deps in pip_pkgs don't directly require numpy, pip sees
        # the existing 2.x as "already satisfied"). The pre-installed pandas
        # was built for numpy 1.x ABI (dtype size 96), so any later import of
        # pandas._libs will crash with "numpy.dtype size changed" until numpy
        # is back on 1.x. This explicit reinstall runs before the other
        # packages so its resolved version sticks.
    # Force-reinstall numpy<2.0 LAST so it overrides ultralytics's pull.
    # ultralytics requires opencv-python (not -headless) which needs libGL.so.1
    # on the cluster. After ultralytics installs opencv-python, we swap it
    # for opencv-python-headless to avoid the missing libGL dep.
    _sp.run(
        [_sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "numpy<2.0"],
        capture_output=True,
    )
    # Install required packages directly — avoids the virtualenv requirement
    if _pip_pkgs_for_worker:
        _sp.run(
            [_sys.executable, "-m", "pip", "install", "-q"] + _pip_pkgs_for_worker,
            capture_output=True,
        )
    # Replace full opencv-python with headless variant to skip libGL dep.
    # ultralytics pulls opencv-python (which needs libGL.so.1, missing on
    # the cluster). We must uninstall opencv-python AND force-reinstall
    # opencv-python-headless with --no-deps to ensure the headless .so
    # files actually overwrite the non-headless ones on next import.
    r_uninst = _sp.run(
        [_sys.executable, "-m", "pip", "uninstall", "-y", "-q",
         "opencv-python", "opencv-contrib-python"],
        capture_output=True, text=True,
    )
    print(f"[opencv-fix] uninstall rc={r_uninst.returncode}", flush=True)
    r_inst = _sp.run(
        [_sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         "--no-deps", "opencv-python-headless"],
        capture_output=True, text=True,
    )
    print(f"[opencv-fix] headless install rc={r_inst.returncode}", flush=True)
    # Verify cv2 import works without libGL dep
    r_check = _sp.run(
        [_sys.executable, "-c", "import cv2; print('cv2 ok:', cv2.__version__)", ],
        capture_output=True, text=True,
    )
    print(f"[opencv-fix] cv2 check: {r_check.stdout.strip()} {r_check.stderr.strip()}", flush=True)
    for k, v in env_vars_arg.items():
        _os.environ[k] = v
    # Force unbuffered stdout/stderr in the exec'd training script so
    # Ray's log endpoint (and the API's 15x log-fetch loop) see
    # crash output as it happens. Without this, a failing script
    # buffers everything until exit — by which time the log fetcher
    # has already given up and reported an empty "15/15 empty" log.
    _os.environ["PYTHONUNBUFFERED"] = "1"
    # Synchronous CUDA errors. Default is async, so a device-side
    # assert is reported at some later random API call and the
    # traceback points to a line that has nothing to do with the
    # real failure. CUDA_LAUNCH_BLOCKING=1 makes each kernel report
    # its own error inline, so the traceback matches the real crash
    # site (yolov8n, monai-unet, etc. all hit "Log fetch 15/15 empty"
    # because the async assert never made it back before the script
    # was killed by the next crash).
    _os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    captured = _StringIO()
    _orig_out, _orig_err = _sys.stdout, _sys.stderr
    _sys.stdout = _sys.stderr = captured
    try:
        exec(_b64.b64decode(script_b64_arg).decode(), {"__name__": "__main__"})  # noqa: S102
    except SystemExit as e:
        if e.code not in (None, 0):
            raise RuntimeError(f"Script exited code {e.code}\n{captured.getvalue()}")
    finally:
        _sys.stdout, _sys.stderr = _orig_out, _orig_err
    output = captured.getvalue()
    wp = next((ln.split(":", 1)[1].strip() for ln in output.splitlines() if ln.startswith("WEIGHTS_UPLOADED:")), None)
    return {"stdout": output, "weights_path": wp}

    # Keep Ray connection alive after training (don't shutdown — other threads may
    # be using it for serve deployments concurrently)
    with _run_on_ray_cluster._lock:
        ray.init(
            address=ray_addr,
            ignore_reinit_error=True,
            allow_multiple=True,
            runtime_env=_VLM_LLM_RUNTIME_ENV,
        )
        _append_log(job_id, f"[cluster] Connected ({len(ray.nodes())} node(s))")

        _set_step(job_id, "submit")
        remote_fn = ray.remote(_run_training_inline)
        future    = remote_fn.remote(script_b64, env_vars)
        _append_log(job_id, "[cluster] Training task submitted, waiting ...")

        elapsed  = 0
        poll_sec = 15
        training_step_set = False
        while True:
            time.sleep(poll_sec)
            elapsed += poll_sec
            # Honour user cancellation even when using ray.wait (the
            # future may stay PENDING forever if the cluster can't
            # allocate a slot).
            try:
                with get_db() as _c:
                    _db_row = _c.execute(
                        "SELECT status FROM jobs WHERE id=?", (job_id,)
                    ).fetchone()
                if _db_row and _db_row["status"] not in ("running", "queued"):
                    _append_log(job_id, f"[cluster] DB status is '{_db_row['status']}' — abandoning Ray wait")
                    try:
                        ray.cancel(future)
                    except Exception:
                        pass
                    break
            except Exception:
                pass
            ready, _ = ray.wait([future], timeout=0)
            if ready:
                break
            if not training_step_set:
                _set_step(job_id, "training")
                training_step_set = True
            _append_log(job_id, f"[cluster] Training in progress ({elapsed // 60}m {elapsed % 60}s elapsed) ...")

        result = ray.get(future)

    # Save full stdout to log
    stdout = result.get("stdout", "")
    for line in stdout.splitlines():
        if line.strip():
            _append_log(job_id, line)

    _set_step(job_id, "saving")
    weights_path = result.get("weights_path")
    if weights_path:
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET s3_weights_path = ? WHERE id = ?",
                (weights_path, job_id),
            )
            conn.commit()
    _append_log(job_id, f"✓ Training complete — weights: {weights_path or '(not found)'}")
    _set_step(job_id, "done")
    _set_status(job_id, "completed")


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


def _set_step(job_id: str, step: str):
    """Update the pipeline step for real-time UI tracking."""
    with get_db() as conn:
        conn.execute("UPDATE jobs SET pipeline_step = ? WHERE id = ?", (step, job_id))
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
    """Run model training on Ray/Modal cluster (vision, LLM, VLM — all types)."""
    job_id  = job["id"]
    cluster = job.get("cluster", "ray")

    try:
        # Mark running
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                (time.time(), job_id),
            )
            conn.commit()

        if cluster in ("ray", "modal"):
            _run_on_ray_cluster(job)
            return

        raise RuntimeError(
            "No cluster configured for this job. Training requires a Ray cluster. "
            "Set cluster=ray and ensure the Ray cluster is running."
        )

    except Exception as exc:
        _append_log(job_id, f"ERROR: {exc}")
        _set_status(job_id, "error", str(exc))


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.get("/api/train/cluster-status")
def get_train_cluster_status():
    """Return availability of Ray and Modal clusters for training."""
    import requests as _req

    # Check on-prem Ray cluster
    ray_ok   = False
    ray_info = ""
    try:
        r = _req.get(f"{RAY_URL}/api/cluster_status", timeout=4)
        ray_ok   = r.ok
        ray_info = "connected" if r.ok else f"HTTP {r.status_code}"
    except Exception as e:
        ray_info = str(e)

    modal_status  = _modal_state.get("status", "idle")
    modal_ray_url = _modal_state.get("ray_url")
    creds_saved   = _load_modal_creds() is not None

    return {
        "ray":   {"available": ray_ok,                    "url": RAY_URL,       "info": ray_info},
        "modal": {
            "available":  modal_status == "running",
            "status":     modal_status,
            "ray_url":    modal_ray_url,
            "creds_saved": creds_saved,
            "gpu_type":    _modal_state.get("gpu_type", "T4"),
            "num_workers": _modal_state.get("num_workers", 0),
        },
        "minio_for_ray":   _resolve_minio_url_for_ray(),
        "minio_for_ray_reachable": _ping_host_port(
            *_parse_host_port(_resolve_minio_url_for_ray())
        ),
    }


@app.get("/api/diag/minio")
def diag_minio():
    """
    Diagnostic endpoint the training script can hit when the dataset
    download fails. Probes each candidate MinIO host:port and reports
    reachability + the env vars that would fix it. Cheap (TCP connect only).
    """
    from urllib.parse import urlparse as _urlparse
    candidates: list[dict] = []
    for label, url in [
        ("MINIO_PUBLIC_URL",    MINIO_PUBLIC_URL),
        ("MINIO_URL",           MINIO_URL),
        ("_resolve_minio_url_for_ray()", _resolve_minio_url_for_ray()),
    ]:
        try:
            p = _urlparse(url)
            if p.hostname and p.port:
                candidates.append({
                    "label":     label,
                    "url":       url,
                    "host":      p.hostname,
                    "port":      p.port,
                    "reachable": _ping_host_port(p.hostname, p.port),
                })
        except Exception:
            pass
    return {
        "RAY_URL":          RAY_URL,
        "MINIO_HOST_IP":    MINIO_HOST_IP,
        "candidates":       candidates,
        "fix_hint": (
            "If the chosen host is unreachable from the Ray cluster, set "
            "MINIO_HOST_IP=<ip-of-the-host-running-docker-compose> (or "
            "MINIO_PUBLIC_URL=http://that-host:9000) on the medimage-api "
            "container and restart it."
        ),
    }


_ZERO_SHOT_MODELS = ("grounding-dino", "groundingdino", "owl-vit", "owlvit", "owlv2")

# Common model ID typo corrections
_MODEL_ID_FIXES = {
    "facebook/detr-resnet50": "facebook/detr-resnet-50",
    "facebook/detr-resnet101": "facebook/detr-resnet-101",
}


def _normalize_model_name(model_name: str) -> str:
    return _MODEL_ID_FIXES.get(model_name, model_name)


def _resolve_text_dataset_path(ds_id_or_path: str) -> str:
    """Resolve a text_dataset value (ds_id, basename, or full path) to an
    absolute local path the Ray worker can read. Falls back to the input
    unchanged so HF hub dataset names still work."""
    if not ds_id_or_path:
        return ""
    # Already a path
    if ds_id_or_path.startswith("/"):
        return ds_id_or_path
    # Try to look up by id in DB
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT path FROM text_datasets WHERE id = ?",
                (ds_id_or_path,),
            ).fetchone()
        if row and row["path"]:
            return row["path"]
    except Exception:
        pass
    # Treat as relative path under /data/text-datasets/
    candidate = f"/data/text-datasets/{ds_id_or_path}"
    if os.path.isfile(candidate):
        return candidate
    return ds_id_or_path


_EXPORT_ENGINES = {"TF-Lite", "ONNX", "NVIDIA", "Intel", "Apple"}


def _check_zero_shot(model_name: str, engine: str):
    """Raise HTTPException if engine/model cannot be used for training."""
    if engine in _EXPORT_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Engine '{engine}' is for model export/conversion, not training. "
                "First train a model with YOLO, HuggingFace, or PyTorch, then use the Export feature."
            ),
        )
    if engine == "HuggingFace" and any(z in model_name.lower() for z in _ZERO_SHOT_MODELS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' is a zero-shot / open-vocabulary detector and "
                "cannot be fine-tuned with the HuggingFace detection pipeline. "
                "Supported fine-tunable architectures: DETR, Deformable-DETR, RT-DETR, YOLOS, DETA. "
                "Try: facebook/detr-resnet-50  or  hustvl/yolos-tiny"
            ),
        )


@app.post("/api/train/{project_id}")
def submit_job(project_id: int, req: TrainRequest, request: Request):
    req.model_name = _normalize_model_name(req.model_name)
    _check_zero_shot(req.model_name, req.engine)
    job_id = str(uuid.uuid4())[:8]
    # Capture the submitter's Keycloak user_id so the worker can pick up
    # that user's saved HuggingFace token (for gated models like
    # MahmoodLab/UNI). Falls back to '' when auth is disabled.
    _user_payload = _extract_token_payload(request)
    submitter_user_id = _user_payload.get("sub", "") if _user_payload else ""
    is_llm = req.training_type in ("llm-text", "vlm-finetune")
    if is_llm:
        ds_label = req.text_dataset or f"dataset-{project_id}"
        name = f"{req.training_type.upper()} · {req.model_name} · {ds_label}"
    else:
        name = f"{req.training_type.upper()} · {req.model_name} · proj-{project_id}"

    dataset_label = req.text_dataset if (is_llm and req.text_dataset) else f"LS project {project_id}"

    # Validate cluster availability
    cluster = req.cluster or "ray"
    if cluster == "modal":
        if _modal_state.get("status") != "running":
            raise HTTPException(
                400,
                f"Modal cluster is not running (status: {_modal_state.get('status', 'idle')}). "
                "Please start the Modal cluster first.",
            )

    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id, name, project_id, dataset, training_type, model_name, engine,
                epochs, batch_size, learning_rate, optimizer, imgsz, notes,
                lora_rank, quantization, max_seq_len, chat_template, grad_accum, text_dataset,
                cluster, num_gpus, status, progress, log, created_at, user_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',0,'',?,?)""",
            (
                job_id, name, project_id, dataset_label,
                req.training_type, req.model_name, req.engine,
                req.epochs, req.batch_size, req.learning_rate,
                req.optimizer, req.imgsz, req.notes,
                req.lora_rank, req.quantization, req.max_seq_len,
                req.chat_template, req.grad_accum, req.text_dataset,
                cluster, req.num_gpus,
                time.time(),
                submitter_user_id,
            ),
        )
        conn.commit()

    with get_db() as conn:
        row = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    threading.Thread(target=run_training, args=(row,), daemon=True).start()

    return {"job_id": job_id, "message": f"Training job {job_id} queued for {req.model_name} on {cluster} cluster"}


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
            "s3_weights_path": d.get("s3_weights_path", ""),
            "modal_url":          d.get("modal_url", ""),
            "modal_api_key":      d.get("modal_api_key", ""),
            "ray_serve_url":      d.get("ray_serve_url", ""),
            "inference_provider": d.get("inference_provider", ""),
            "pipeline_step":      d.get("pipeline_step", ""),
        })
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return dict(row)


# ─── WebSocket for real-time job updates ─────────────────────────────────────
_ws_clients: list[WebSocket] = []
_ws_broadcast_task: asyncio.Task | None = None


async def _ws_broadcast_loop():
    """Push job updates to all connected WebSocket clients every 2 seconds."""
    while True:
        await asyncio.sleep(2)
        if not _ws_clients:
            continue
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT id, name, training_type, model_name, engine, status, "
                    "progress, pipeline_step, created_at, started_at, finished_at, "
                    "error, dataset, epochs, batch_size "
                    "FROM jobs WHERE hidden_in_jobs = 0 ORDER BY created_at DESC"
                ).fetchall()
            payload = []
            for r in rows:
                d = dict(r)
                d["model"] = d.pop("model_name", "")
                payload.append(d)
            msg = _json.dumps({"type": "jobs", "data": payload})
            disconnected = []
            for ws in _ws_clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                _ws_clients.remove(ws)
        except Exception:
            pass


@app.websocket("/ws/jobs")
async def ws_jobs(websocket: WebSocket):
    """WebSocket endpoint for real-time job status updates.

    Accepts an optional ``token`` query parameter for JWT auth when Keycloak
    is enabled. Clients should pass the same Bearer token as the REST API.
    When Keycloak is disabled, the connection is accepted without auth.
    """
    global _ws_broadcast_task
    if _KC_ENABLED and _jwks_client is not None:
        token = websocket.query_params.get("token", "")
        if not token:
            await websocket.close(code=4001, reason="Missing token")
            return
        try:
            await asyncio.to_thread(_sync_verify_token, token)
        except Exception:
            await websocket.close(code=4001, reason="Invalid token")
            return
    await websocket.accept()
    _ws_clients.append(websocket)
    if _ws_broadcast_task is None or _ws_broadcast_task.done():
        _ws_broadcast_task = asyncio.create_task(_ws_broadcast_loop())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ─── Bulk-run state: persisted in SQLite, broadcast over WebSocket ───────────
# All Test-All "Run all compatible" state lives in the bulk_runs and
# bulk_run_jobs tables (already in init_db). Frontend connects to
# /ws/testall, sends/receives the latest snapshot every 2s. This replaces
# the previous localStorage + 3s polling scheme.

_bulk_ws_clients: list[WebSocket] = []
_bulk_ws_task: asyncio.Task | None = None


def _load_bulk_snapshot() -> dict:
    """Build a single JSON snapshot of all bulk runs + their jobs.

    Only the latest 5 runs are returned to keep the payload small.
    The job status is enriched with the live status from the jobs
    table (when a job_id is set) so the TestAll UI reflects the
    actual backend state without polling.
    """
    with get_db() as conn:
        runs = conn.execute(
            "SELECT * FROM bulk_runs ORDER BY started_at DESC LIMIT 5"
        ).fetchall()
        if not runs:
            return {"runs": [], "jobs_by_run": {}}
        run_ids = [r["id"] for r in runs]
        placeholders = ",".join("?" for _ in run_ids)
        jobs = conn.execute(
            f"SELECT * FROM bulk_run_jobs WHERE bulk_run_id IN ({placeholders})",
            run_ids,
        ).fetchall()
        # Enrich with live job status so single-test submissions (which
        # only update bulk_run_jobs at submit time) get the running /
        # completed / error status as the backend updates it.
        # sqlite3.Row supports [] but not .get(); convert upfront.
        job_ids = [j["job_id"] for j in jobs if j["job_id"]]
        live_status: dict[str, dict] = {}
        if job_ids:
            jplace = ",".join("?" for _ in job_ids)
            for row in conn.execute(
                f"SELECT id, status, error, started_at, finished_at, pipeline_step FROM jobs WHERE id IN ({jplace})",
                job_ids,
            ).fetchall():
                live_status[row["id"]] = dict(row)
        jobs_by_run: dict[str, list] = {rid: [] for rid in run_ids}
        for j in jobs:
            d = dict(j)
            # Overlay the live job status onto the bulk row. The bulk
            # row stores the status at submit time, which stays at
            # "queued" forever unless the frontend (or a "Run all"
            # loop) explicitly updates it. The jobs table is the
            # source of truth for the actual training state.
            if d.get("job_id") and d["job_id"] in live_status:
                live = live_status[d["job_id"]]
                d["status"] = live.get("status") or d.get("status") or "queued"
                if live.get("error") and d["status"] == "error":
                    d["error"] = live["error"]
                if live.get("started_at"):
                    d["started_at"] = live["started_at"]
                if live.get("finished_at"):
                    d["finished_at"] = live["finished_at"]
            jobs_by_run[j["bulk_run_id"]].append(d)
        return {
            "runs": [dict(r) for r in runs],
            "jobs_by_run": jobs_by_run,
        }


async def _bulk_ws_broadcast_loop():
    """Push the latest bulk-run snapshot every 1s to subscribed clients.
    The Jobs page polls /api/jobs directly so it doesn't need this,
    but TestAllModels shows the bulk_run_jobs snapshot overlaid with the
    live jobs-table status — 1s keeps the displayed row status in
    sync with the actual job state without burning the database.
    """
    while True:
        await asyncio.sleep(1)
        if not _bulk_ws_clients:
            continue
        try:
            snap = _load_bulk_snapshot()
            msg = _json.dumps({"type": "bulk_snapshot", "data": snap})
            disconnected = []
            for ws in _bulk_ws_clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                _bulk_ws_clients.remove(ws)
        except Exception:
            pass


@app.websocket("/ws/testall")
async def ws_testall(websocket: WebSocket):
    """WebSocket endpoint for Test-All bulk run state.

    Server pushes the current bulk_runs + bulk_run_jobs snapshot every
    2s. Clients may also send a {"action":"create_run", "id":...} to
    register a new bulk run row, and use REST endpoints to update jobs.
    """
    global _bulk_ws_task
    if _KC_ENABLED and _jwks_client is not None:
        token = websocket.query_params.get("token", "")
        if not token:
            await websocket.close(code=4001, reason="Missing token")
            return
        try:
            await asyncio.to_thread(_sync_verify_token, token)
        except Exception:
            await websocket.close(code=4001, reason="Invalid token")
            return
    await websocket.accept()
    _bulk_ws_clients.append(websocket)
    if _bulk_ws_task is None or _bulk_ws_task.done():
        _bulk_ws_task = asyncio.create_task(_bulk_ws_broadcast_loop())
    # Send an initial snapshot immediately so the client doesn't wait 2s
    try:
        snap = _load_bulk_snapshot()
        await websocket.send_text(_json.dumps({"type": "bulk_snapshot", "data": snap}))
    except Exception:
        pass
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _bulk_ws_clients:
            _bulk_ws_clients.remove(websocket)


@app.get("/api/bulk-runs/latest")
def get_latest_bulk_run():
    """Return the latest bulk run + its jobs (used for initial mount)."""
    return _load_bulk_snapshot()


@app.post("/api/bulk-runs")
def create_bulk_run(payload: dict, request: Request):
    """Create a new bulk_runs row. Returns the row's id.

    Body: {id?: str, provider?: 'ray'|'modal', deploy_enabled?: bool, total?: int}
    The frontend supplies a stable id (usually the same as a previous
    run, so the row upserts instead of growing the table forever).
    """
    import uuid
    rid = payload.get("id") or f"bulk_{uuid.uuid4().hex[:12]}"
    user_id = _current_user_id(request)
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO bulk_runs(id, started_at, provider, deploy_enabled, total, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rid, time.time(),
             payload.get("provider", "ray"),
             1 if payload.get("deploy_enabled") else 0,
             int(payload.get("total", 0)),
             user_id),
        )
        conn.commit()
    return {"id": rid}


@app.patch("/api/bulk-runs/{run_id}")
def update_bulk_run(run_id: str, payload: dict):
    """Update a bulk_runs row (finished_at, stopped, counts, etc)."""
    fields, params = [], []
    for k in ("finished_at", "stopped", "total", "ok", "failed", "deployed_count"):
        if k in payload:
            fields.append(f"{k}=?")
            params.append(payload[k])
    if not fields:
        return {"updated": 0}
    params.append(run_id)
    with get_db() as conn:
        cur = conn.execute(f"UPDATE bulk_runs SET {', '.join(fields)} WHERE id=?", params)
        conn.commit()
    return {"updated": cur.rowcount}


@app.post("/api/bulk-runs/{run_id}/stop")
def stop_bulk_run(run_id: str):
    """Mark a bulk run as stopped on the server. The frontend's "Stop
    Bulk" button calls this so a stop request survives a page refresh /
    tab close / JS loop crash — the in-flight train job on the Ray
    cluster is not cancelled (that would need a separate call to the
    jobs endpoint) but the bulk_runs row is finalised so the next WS
    push reflects a finished, stopped state and the UI gets unstuck.
    Idempotent: safe to call when the run is already finished.
    """
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE bulk_runs SET stopped = 1, "
            "finished_at = COALESCE(finished_at, ?) "
            "WHERE id = ? AND finished_at IS NULL",
            (time.time(), run_id),
        )
        conn.commit()
    return {"updated": cur.rowcount, "run_id": run_id}


@app.delete("/api/bulk-runs")
def delete_bulk_runs(request: Request):
    """Wipe all bulk_runs (and their bulk_run_jobs) for the current
    user. The frontend's "Reset" button calls this so a reset survives
    a page refresh — without a server-side delete, the next WS push
    would just refill the snapshot with the same rows and the UI would
    flicker for 2s before looking identical to before the click.
    Per-user: each user only wipes their own rows. The dismissed-flag
    in user_prefs is intentionally NOT cleared — that preference is
    per-run and outlives the run's lifetime.
    """
    uid = _current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="authentication required")
    with get_db() as conn:
        # Delete child rows first (FK not enforced in SQLite by default
        # but be explicit so the orphan check is obvious).
        conn.execute(
            "DELETE FROM bulk_run_jobs WHERE bulk_run_id IN "
            "(SELECT id FROM bulk_runs WHERE user_id = ?)",
            (uid,),
        )
        cur = conn.execute("DELETE FROM bulk_runs WHERE user_id = ?", (uid,))
        conn.commit()
    return {"deleted": cur.rowcount}


@app.post("/api/bulk-runs/{run_id}/jobs")
def upsert_bulk_run_job(run_id: str, payload: dict):
    """Upsert a single bulk_run_jobs row (per-model state)."""
    row_key = payload.get("row_key")
    if not row_key:
        raise HTTPException(status_code=400, detail="row_key required")
    # Build column list and VALUES placeholders separately. The
    # "column = excluded.column" form on the UPDATE branch is built
    # by joining each column name with the SQLite "excluded" pseudo-
    # table alias — NEVER embed "?" twice (the previous version of
    # this query produced "status=?=excluded.status" which SQLite
    # parses as 14 placeholders but the params list only has 8).
    cols = ["status", "error", "elapsed_sec", "deployed", "deploy_url", "job_id"]
    placeholders = ", ".join("?" for _ in cols)
    update_set = ", ".join(f"{c}=excluded.{c}" for c in cols)
    params = [
        payload.get("status", "idle"),
        payload.get("error"),
        float(payload.get("elapsed_sec", 0) or 0),
        1 if payload.get("deployed") else 0,
        payload.get("deploy_url"),
        payload.get("job_id"),
        run_id,
        row_key,
    ]
    with get_db() as conn:
        conn.execute(
            f"INSERT INTO bulk_run_jobs({', '.join(cols)}, bulk_run_id, row_key) "
            f"VALUES ({placeholders}, ?, ?) "
            f"ON CONFLICT(bulk_run_id, row_key) DO UPDATE SET {update_set}",
            params,
        )
        conn.commit()
    return {"ok": True}


@app.get("/api/bulk-runs/{run_id}/jobs")
def list_bulk_run_jobs(run_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM bulk_run_jobs WHERE bulk_run_id = ?", (run_id,)
        ).fetchall()
    return {"jobs": [dict(r) for r in rows]}


@app.get("/api/jobs/{job_id}/download-weights")
def download_weights(job_id: str):
    """Generate a presigned download URL for the model weights stored in MinIO."""
    with get_db() as conn:
        row = conn.execute("SELECT s3_weights_path, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    path = row["s3_weights_path"] or ""
    if not path:
        raise HTTPException(status_code=404, detail="No weights found for this job")

    # path stored as "bucket/key" or just "key" inside the default weights bucket
    if "/" in path:
        bucket, key = path.split("/", 1)
    else:
        bucket = "weights"
        key = path

    try:
        s3_pub = _boto3.client(
            "s3",
            endpoint_url=MINIO_PUBLIC_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_BConfig(signature_version="s3v4"),
        )
        url = s3_pub.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        filename = key.split("/")[-1]
        return {"url": url, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {e}")


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


# ─── Model Deployment Endpoints ──────────────────────────────────────────────

class DeploymentRequest(BaseModel):
    s3_weights_path: str | None = None
    modal_url: str | None = None
    modal_api_key: str | None = None
    ray_serve_url: str | None = None
    inference_provider: str | None = None  # '' | 'modal' | 'ray'


@app.patch("/api/jobs/{job_id}/deployment")
def patch_deployment(job_id: str, req: DeploymentRequest):
    """Save deployment config for a model (Modal and/or Ray Serve)."""
    with get_db() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if req.s3_weights_path is not None:
            conn.execute("UPDATE jobs SET s3_weights_path = ? WHERE id = ?", (req.s3_weights_path, job_id))
        if req.modal_url is not None:
            conn.execute("UPDATE jobs SET modal_url = ? WHERE id = ?", (req.modal_url, job_id))
        if req.modal_api_key is not None:
            conn.execute("UPDATE jobs SET modal_api_key = ? WHERE id = ?", (req.modal_api_key, job_id))
        if req.ray_serve_url is not None:
            conn.execute("UPDATE jobs SET ray_serve_url = ? WHERE id = ?", (req.ray_serve_url, job_id))
        if req.inference_provider is not None:
            conn.execute("UPDATE jobs SET inference_provider = ? WHERE id = ?", (req.inference_provider, job_id))
        conn.commit()
    return {"ok": True}


async def _ping_endpoint(url: str, api_key: str) -> dict:
    """Ping a /health endpoint and return {online, status_code?, error?}"""
    # Modal per-model URLs end with the function name
    # Health endpoint is either the stored URL itself (if it ends with 'health')
    # or we swap 'inference' -> 'health'
    if "modal.run" in url:
        if url.rstrip("/").endswith("-health.modal.run") or "-health" in url.split("modal.run")[0].split("--")[-1]:
            health_url = url.rstrip("/")
        else:
            health_url = url.replace("inference", "health").rstrip("/")
    else:
        health_url = url.rstrip("/") + "/health"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(health_url, headers=headers)
        return {"online": resp.status_code < 500, "status_code": resp.status_code}
    except Exception as e:
        return {"online": False, "error": str(e)}


@app.get("/api/jobs/{job_id}/modal-status")
async def get_modal_status(job_id: str):
    """Ping the Modal endpoint to check if it's online."""
    with get_db() as conn:
        row = conn.execute("SELECT modal_url, modal_api_key FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not row["modal_url"]:
        raise HTTPException(status_code=404, detail="No Modal endpoint configured")
    return await _ping_endpoint(row["modal_url"], row["modal_api_key"] or "")


@app.get("/api/jobs/{job_id}/ray-status")
async def get_ray_status(job_id: str):
    """Check if the model is reachable on the Ray cluster.

    Two deployment mechanisms exist:
      1. Ray Serve app named `medimage-{job_id}` (the in-script _run_modal
         path uses this for direct model serving).
      2. Ray detached actor named `model-{job_id}` (the modal-model-deploy
         path uses this; the actor is created with lifetime="detached"
         so it survives the driver job exiting).

    The actor is what the user actually sees in the Ray dashboard
    ('Model actor "model-7bc8f748" is running on Ray cluster!'),
    so it's the source of truth for "is this model reachable?". We
    also check the Ray Serve app as a fallback.
    """
    with get_db() as conn:
        row = conn.execute("SELECT ray_serve_url, inference_provider FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not row["ray_serve_url"]:
        return {"online": False, "status": "no_url"}
    try:
        import os as _os
        _os.environ["RAY_ADDRESS"] = ""
        _os.environ["RAY_ENABLE_AUTO_CONNECT"] = "0"
        import ray as _ray
        from ray import serve as _serve
        from urllib.parse import urlparse as _up
        _host = _up(row["ray_serve_url"]).hostname or "100.68.53.118"
        _addr = f"ray://{_host}:10001"
        if not _ray.is_initialized():
            _ray.init(address=_addr, ignore_reinit_error=True, allow_multiple=True, log_to_driver=False)

        # 1) Detached Ray actor (model-{job_id}) — preferred, since
        #    that's what the modal-model-deploy path creates and what
        #    the user sees in the Ray dashboard.
        _actor_name = f"model-{job_id}"
        try:
            _actor = _ray.get_actor(_actor_name, namespace="default")
            # get_actor() raises if the actor doesn't exist OR is DEAD.
            # Successful return ⇒ the actor handle is alive, and since
            # detached actors are only cleaned up on explicit kill, an
            # existing handle means inference is reachable.
            return {"online": True, "status": "actor_alive", "kind": "ray_actor", "name": _actor_name}
        except Exception:
            pass

        # 2) Ray Serve app (medimage-{job_id}) — fallback.
        _status = _serve.status()
        _app_name = f"medimage-{job_id}"
        _info = _status.applications.get(_app_name)
        if _info is not None:
            _s = str(_info.status)
            _online = "RUNNING" in _s or "HEALTHY" in _s
            return {"online": _online, "status": _s, "kind": "ray_serve", "name": _app_name}

        return {"online": False, "status": "not_deployed"}
    except Exception as e:
        return {"online": False, "status": "error", "error": str(e)}


@app.delete("/api/jobs/{job_id}/deployment")
def clear_deployment(job_id: str):
    """Remove all deployment endpoint configuration."""
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET modal_url='', modal_api_key='', ray_serve_url='', inference_provider='' WHERE id = ?",
            (job_id,),
        )
        conn.commit()
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, from_view: str = "jobs"):
    with get_db() as conn:
        row = conn.execute(
            "SELECT status, inference_provider, ray_serve_url, modal_url FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        # Verify the deployment is still actually live before refusing
        # the delete. The DB column can be stale (Ray cluster restarted,
        # Pod killed, network blip) — in that case there's nothing to
        # protect, so we auto-clear the stale URL and proceed.
        provider = (row["inference_provider"] or "").strip()
        ray_url  = (row["ray_serve_url"] or "").strip()
        modal_url = (row["modal_url"] or "").strip()
        if provider == "ray" and ray_url:
            try:
                _ping = httpx.get(ray_url.rstrip("/") + "/health", timeout=4)
                if _ping.status_code < 500:
                    raise HTTPException(
                        status_code=409,
                        detail="Model is deployed on Ray Serve — undeploy first (Deploy tab → Undeploy).",
                    )
            except HTTPException:
                raise
            except Exception:
                # Endpoint unreachable → stale DB state. Clear it.
                conn.execute(
                    "UPDATE jobs SET ray_serve_url='', inference_provider='' WHERE id=?",
                    (job_id,),
                )
                ray_url = ""
                provider = ""
        if provider == "modal" and modal_url:
            _modal_health = modal_url.replace("inference", "health").rstrip("/")
            try:
                _ping = httpx.get(_modal_health, timeout=4)
                if _ping.status_code < 500:
                    raise HTTPException(
                        status_code=409,
                        detail="Model is deployed on Modal — undeploy first (Deploy tab → Stop).",
                    )
            except HTTPException:
                raise
            except Exception:
                conn.execute(
                    "UPDATE jobs SET modal_url='', modal_api_key='', inference_provider='' WHERE id=?",
                    (job_id,),
                )
                modal_url = ""
                provider = ""
        if from_view == "models":
            # Deleting from the Models page removes the model entity
            # entirely — hide it from both views.
            conn.execute(
                "UPDATE jobs SET hidden_in_models = 1, hidden_in_jobs = 1 WHERE id = ?",
                (job_id,),
            )
        else:
            # Cancel if still active, then hide from jobs view only.
            if row["status"] in ("queued", "running"):
                conn.execute("UPDATE jobs SET status = 'error', error = 'Cancelled by user' WHERE id = ?", (job_id,))
            conn.execute("UPDATE jobs SET hidden_in_jobs = 1 WHERE id = ?", (job_id,))
        conn.commit()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    """Reset a failed job to queued and resubmit it in place (same job_id)."""
    import re as _re
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    old = dict(row)
    old["model_name"] = _normalize_model_name(old.get("model_name", ""))
    _check_zero_shot(old.get("model_name", ""), old.get("engine", ""))

    # Preflight: confirm the Ray cluster can actually reach MinIO at the
    # host the API will sign the dataset URL against. If not, fail fast
    # with a clear actionable error instead of letting the user wait 5
    # minutes for a training run that dies at the first HTTP GET.
    _ray_minio_url = _resolve_minio_url_for_ray()
    _hp = _parse_host_port(_ray_minio_url)
    if not _ping_host_port(_hp[0], _hp[1], timeout=2.0):
        raise HTTPException(
            status_code=400,
            detail=(
                f"MinIO at {_ray_minio_url} is not reachable from this API "
                f"container. The Ray worker will fail to download the "
                f"dataset (Connection refused). "
                f"Fix: set MINIO_HOST_IP=<ip-of-the-host-running-docker-compose> "
                f"on the medimage-api container, or MINIO_PUBLIC_URL=http://that-host:9000, "
                f"then restart the API. See /api/diag/minio for details."
            ),
        )

    # Strip any legacy [Retry-N] prefix from the name (idempotent cleanup)
    clean_name = _re.sub(r'^(\[Retry-?\d*\]\s*)+', '', old["name"]).strip() or old["name"]

    with get_db() as conn:
        conn.execute(
            """UPDATE jobs SET
                name = ?,
                status = 'queued',
                progress = 0,
                log = '',
                pipeline_step = '',
                error = NULL,
                started_at = NULL,
                finished_at = NULL,
                updated_at = CURRENT_TIMESTAMP
              WHERE id = ?""",
            (clean_name, job_id),
        )
        conn.commit()

    with get_db() as conn:
        row = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    threading.Thread(target=run_training, args=(row,), daemon=True).start()
    return {"job_id": job_id}


# ─── Import pretrained model ─────────────────────────────────────────────────

class ImportModelRequest(BaseModel):
    name: str
    training_type: str = "classification"
    model_name: str
    engine: str = "PyTorch"
    source_type: str = "huggingface"   # huggingface | url | builtin
    source_url: str = ""
    notes: str = ""


def _run_import(job_id: str, req: ImportModelRequest, user_id: str = ""):
    """Register an imported / pretrained model. No weights download needed —
    the Ray Serve actor loads pretrained weights from MODEL_NAME at deploy time."""
    try:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (time.time(), job_id))
            conn.commit()

        _set_step(job_id, "validate")
        _append_log(job_id, f"Registering model: {req.model_name} ({req.engine} / {req.training_type})")
        _set_progress(job_id, 50)

        # Validate HuggingFace model exists (quick metadata fetch, no download).
        # Use the submitter's saved HF token so private/gated models verify.
        if req.source_type == "huggingface" and req.model_name:
            _hf_token = _load_user_hf_token(user_id) or os.getenv("HF_TOKEN", "") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
            _headers  = {"Authorization": f"Bearer {_hf_token}"} if _hf_token else {}
            try:
                import urllib.request as _ur
                _req2 = _ur.Request(
                    f"https://huggingface.co/api/models/{req.model_name}",
                    headers=_headers,
                )
                _ur.urlopen(_req2, timeout=10)  # noqa: S310 — validating HF model
                _append_log(job_id, f"HuggingFace model '{req.model_name}' verified ✓")
            except Exception as _ve:
                _append_log(job_id, f"Warning: could not verify HuggingFace model ({_ve}) — proceeding anyway")

        _set_step(job_id, "register")
        _set_progress(job_id, 100)
        _append_log(job_id, f"Model '{req.model_name}' registered. Pretrained weights will be loaded from {req.source_type} at deploy time.")
        _set_step(job_id, "done")
        _set_status(job_id, "completed")
    except Exception as exc:
        _append_log(job_id, f"ERROR: {exc}")
        _set_status(job_id, "error", str(exc))


@app.post("/api/models/import")
def import_model(req: ImportModelRequest, request: Request):
    job_id = str(uuid.uuid4())[:8]
    _user_payload = _extract_token_payload(request)
    user_id = _user_payload.get("sub", "") if _user_payload else ""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id, name, project_id, dataset, training_type, model_name, engine,
                epochs, batch_size, learning_rate, optimizer, imgsz, notes,
                status, progress, log, created_at, source, source_url, user_id)
               VALUES (?,?,0,'pretrained',?,?,?,0,0,0,'—',0,?,'queued',0,'',?,?,?,?)""",
            (
                job_id, req.name,
                req.training_type, req.model_name, req.engine,
                req.notes, time.time(),
                "imported", req.source_url, user_id,
            ),
        )
        conn.commit()

    threading.Thread(target=_run_import, args=(job_id, req, user_id), daemon=True).start()
    return {"job_id": job_id}


# ─── Settings ───────────────────────────────────────────────────────────────

_INFERENCE_KEYS = ("global_inference_provider", "global_modal_url", "global_modal_api_key", "global_ray_serve_url")

@app.get("/api/settings/inference")
def get_inference_settings():
    """Return the global default inference endpoint configuration."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT key, value FROM settings WHERE key IN (?,?,?,?)", _INFERENCE_KEYS
        ).fetchall()
    d = {r["key"]: r["value"] for r in rows}
    return {
        "provider":    d.get("global_inference_provider", ""),
        "modal_url":   d.get("global_modal_url", ""),
        "modal_api_key": d.get("global_modal_api_key", ""),
        "ray_serve_url": d.get("global_ray_serve_url", ""),
    }


@app.put("/api/settings/inference")
def save_inference_settings(body: dict):
    """Persist the global default inference endpoint configuration."""
    pairs = [
        ("global_inference_provider", body.get("provider", "")),
        ("global_modal_url",          body.get("modal_url", "")),
        ("global_modal_api_key",      body.get("modal_api_key", "")),
        ("global_ray_serve_url",      body.get("ray_serve_url", "")),
    ]
    with get_db() as conn:
        for key, value in pairs:
            conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, str(value)))
        conn.commit()
    return {"ok": True}


@app.get("/api/settings/inference/modal-status")
async def global_modal_status():
    # The Modal cluster has TWO ways of being "online": (1) the legacy
    # `global_modal_url` setting (an external per-model endpoint) and
    # (2) the Modal Ray cluster tracked in _modal_state. The latter
    # is what the Train popup actually starts; if it's running, the
    # Playground / Models page should be able to find it.
    if _modal_state.get("status") == "running" and _modal_state.get("ray_url"):
        return {
            "online":   True,
            "source":   "modal_cluster",
            "ray_url":  _modal_state["ray_url"],
        }
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='global_modal_url'").fetchone()
    url = row["value"] if row else ""
    if not url:
        raise HTTPException(status_code=404, detail="No global Modal URL configured")
    with get_db() as conn:
        row2 = conn.execute("SELECT value FROM settings WHERE key='global_modal_api_key'").fetchone()
    key = row2["value"] if row2 else ""
    res = await _ping_endpoint(url, key)
    res["source"] = "global_setting"
    return res


@app.get("/api/settings/inference/ray-status")
async def global_ray_status(url: str = ""):
    # Source of truth, in priority order:
    #   1. Modal Ray cluster is up (_modal_state) — the Train popup's
    #      Start Cluster flow lives here.
    #   2. The on-prem RAY_URL dashboard's /api/cluster_status responds
    #      successfully. This catches the case where the user has a
    #      working on-prem Ray cluster (like the user here: model
    #      7bc8f748 is deployed on http://100.68.53.118:8000) but
    #      never set the `global_ray_serve_url` setting.
    #   3. The legacy `global_ray_serve_url` setting.
    if _modal_state.get("status") == "running" and _modal_state.get("ray_url"):
        return {
            "online":  True,
            "source":  "modal_cluster",
            "ray_url": _modal_state["ray_url"],
        }
    # Try the on-prem Ray dashboard directly
    try:
        r = httpx.get(f"{RAY_URL}/api/cluster_status", timeout=4)
        if r.is_success:
            data = r.json() or {}
            if data.get("result") is True:
                return {
                    "online":  True,
                    "source":  "ray_dashboard",
                    "ray_url": RAY_URL,
                }
    except Exception:
        pass
    if not url:
        with get_db() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='global_ray_serve_url'").fetchone()
        url = (row["value"] if row else "") or ""
    if not url:
        return {"online": False, "source": "none"}
    res = await _ping_endpoint(url, "")
    res["source"] = "global_setting"
    return res


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


# ─── Per-user preferences (replaces localStorage) ────────────────────────────
# All small per-user UI state (filter selections, dashboard URLs, MinIO
# credentials, API keys, etc.) lives here. The frontend has a small
# useUserPrefs() hook that mirrors this in memory so reads are sync.
@app.get("/api/user-prefs")
def list_user_prefs(request: Request):
    """Return all of the current user's prefs as {key: value, ...}."""
    uid = _current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    with get_db() as conn:
        rows = conn.execute("SELECT key, value FROM user_prefs WHERE user_id=?", (uid,)).fetchall()
    return {"prefs": {r["key"]: r["value"] for r in rows}}


@app.get("/api/user-prefs/{key}")
def get_user_pref(key: str, request: Request):
    uid = _current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    with get_db() as conn:
        row = conn.execute("SELECT value FROM user_prefs WHERE user_id=? AND key=?", (uid, key)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return {"key": key, "value": row["value"]}


@app.put("/api/user-prefs/{key}")
def put_user_pref(key: str, body: dict, request: Request):
    """Upsert a single preference. Use {"value": "..."} or {"value": null} to delete."""
    uid = _current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    value = body.get("value", "")
    if value is None:
        with get_db() as conn:
            conn.execute("DELETE FROM user_prefs WHERE user_id=? AND key=?", (uid, key))
            conn.commit()
        return {"ok": True, "deleted": True}
    with get_db() as conn:
        conn.execute(
            "INSERT INTO user_prefs(user_id, key, value, updated_at) VALUES(?,?,?,strftime('%s','now')) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (uid, key, str(value)),
        )
        conn.commit()
    return {"ok": True}


# ─── Modal.com Cloud Deploy ──────────────────────────────────────────────────

import re as _re
import threading as _threading

_modal_state: dict = {
    "status": "idle",    # idle | deploying | running | stopping | error
    "ray_url": None,
    "logs": [],
    "proc": None,
    "token_id": "",
    "token_secret": "",
    "gpu_type": "T4",
    "num_workers": 0,
}


class ModalStartRequest(BaseModel):
    # token_id / token_secret are optional in the request: the endpoint will
    # fall back to whatever is saved in the DB (modal_credentials table) when
    # they are missing or empty, so the UI can fire "Start" without having
    # the secret in memory.
    token_id:     str = ""
    token_secret: str = ""
    gpu_type:     str = "T4"
    num_workers:  int = 1


def _modal_script(req: ModalStartRequest) -> str:
    gpu_spec = f'gpu="{req.gpu_type}"' if req.gpu_type != "cpu" else "gpu=None"
    return textwrap.dedent(f"""
        import modal, subprocess, time, socket

        app = modal.App("medimage-ray")

        # Shared store so workers can find the head node IP
        head_ip_store = modal.Dict.from_name("medimage-ray-head-ip", create_if_missing=True)

        ray_image = (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install("ray[serve]>=2.30", "fastapi", "uvicorn",
                         "python-multipart", "pillow")
            # VLM/LLM training deps — baked in once at image build time.
            # The Ray worker (which runs vlm_llm_train.py) needs these
            # available without per-job pip install. Loose pins (>=) so
            # pip's resolver picks the latest compatible versions, which
            # become the "version that exists" for every job that runs on
            # this image. numpy<2.0 is kept pinned to preserve pandas
            # 1.x ABI on this cluster (pandas._libs.interval breaks with
            # numpy 2.x dtype size 88 vs 96).
            #
            # Torch is NOT baked here — the cluster's pre-installed torch
            # is what ultralytics has been using successfully on the DL
            # path. The VLM torch pre-flight in vlm_llm_train.py still
            # force-reinstalls cu wheels if the pre-installed torch
            # can't see a GPU.
            #
            # vlm_llm_train.py is NOT baked into the image — it's shipped
            # per-task via Ray's runtime_env["py_modules"] from the API
            # container. Edits to the DL module are picked up on the
            # next job without rebuilding this image.
            .pip_install(
                "transformers<5.0", "peft", "trl", "bitsandbytes",
                "accelerate", "datasets", "numpy<2.0",
                "Pillow", "boto3", "requests",
            )
        )

        # Head node — min_containers=0 so 'modal app stop' actually tears
        # the box down. With min_containers=1 Modal would respawn a
        # replacement head as soon as the current one dies, leaving the
        # user staring at "stopped" while a new container is starting.
        @app.function(image=ray_image, {gpu_spec}, timeout=86400, memory=16384, min_containers=0)
        @modal.web_server(8265, startup_timeout=300)
        def ray_head():
            head_ip = socket.gethostbyname(socket.gethostname())
            head_ip_store["ip"] = head_ip
            subprocess.Popen([
                "ray", "start", "--head",
                "--dashboard-host=0.0.0.0", "--dashboard-port=8265",
                "--port=6379",
            ])
            time.sleep(86400)

        # Worker node — connects to head via the shared Dict.
        # Spawned separately via `modal run --detach` for each worker.
        @app.function(image=ray_image, {gpu_spec}, timeout=86400, memory=16384, min_containers=0)
        def ray_worker():
            for _ in range(60):
                ip = head_ip_store.get("ip", "")
                if ip:
                    break
                time.sleep(2)
            subprocess.run(["ray", "start", f"--address={{ip}}:6379", "--block"])
    """)


_ANSI_ESCAPE = _re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def _strip_ansi(s: str) -> str:
    return _ANSI_ESCAPE.sub('', s)


def _get_modal_web_url(env: dict) -> str | None:
    """Fallback: query modal deploy output for web endpoint URL."""
    try:
        r = subprocess.run(
            ["python3", "-m", "modal", "app", "list"],
            env=env, capture_output=True, text=True, timeout=20,
        )
        # Look for URLs in app list output
        m = _re.search(r'https://[^\s]+\.modal\.run', _strip_ansi(r.stdout + r.stderr))
        if m:
            return m.group(0)
    except Exception:
        pass
    return None


def _run_modal(req: ModalStartRequest, script_path: str):
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = req.token_id
    env["MODAL_TOKEN_SECRET"] = req.token_secret
    try:
        # Use `modal deploy` so the cluster persists independently (not tied to this process).
        proc = subprocess.Popen(
            ["python3", "-m", "modal", "deploy", script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _modal_state["proc"] = proc
        url_found = False
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if line:
                clean = _strip_ansi(line)
                _modal_state["logs"].append(clean)
                m = _re.search(r'https://[^\s]+\.modal\.run[^\s]*', clean)
                if m:
                    _modal_state["ray_url"] = m.group(0)
                    url_found = True
        exit_code = proc.wait()
        if exit_code == 0 and not url_found:
            # Fallback: query modal app list for the web endpoint URL
            _modal_state["logs"].append("Querying Modal for web endpoint URL...")
            url = _get_modal_web_url(env)
            if url:
                _modal_state["ray_url"] = url
                url_found = True
                _modal_state["logs"].append(f"Web endpoint: {url}")
        if exit_code == 0 and url_found:
            _modal_state["status"] = "running"
        elif exit_code == 0:
            _modal_state["status"] = "running"  # cluster up even if URL unknown
            _modal_state["logs"].append(
                "Deploy succeeded — check modal.com/apps for the web endpoint URL"
            )
        else:
            _modal_state["status"] = "error"
            _modal_state["logs"].append(f"modal deploy failed (exit {exit_code})")
            return
        # Spawn worker containers (separate from head)
        num_w = _modal_state.get("num_workers", 0)
        if exit_code == 0 and num_w > 0:
            _modal_state["logs"].append(f"Spawning {num_w} worker(s)...")
            for i in range(num_w):
                subprocess.Popen(
                    ["python3", "-m", "modal", "run", "--detach",
                     script_path, "::ray_worker"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            _modal_state["logs"].append(f"Workers spawning in background.")
    except Exception as exc:
        _modal_state["status"] = "error"
        _modal_state["logs"].append(f"Fatal: {exc}")


@app.post("/api/modal/start")
def modal_start(req: ModalStartRequest):
    if _modal_state["status"] in ("deploying", "running"):
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Already running — stop first")
    # Allow the caller to omit creds if they are already saved in the DB
    # (so the UI can just say "Start" without re-asking for the secret).
    if not req.token_id or not req.token_secret:
        saved = _load_modal_creds()
        if not saved:
            raise HTTPException(
                status_code=400,
                detail="Modal credentials not provided and none saved — go to Modal Config to set them",
            )
        req = req.model_copy(update={
            "token_id":     saved["token_id"],
            "token_secret": saved["token_secret"],
        })
    _modal_state.update(status="deploying", ray_url=None, logs=[], proc=None,
                        token_id=req.token_id, token_secret=req.token_secret,
                        gpu_type=req.gpu_type, num_workers=req.num_workers)
    script_path = "/tmp/medimage_modal_ray.py"
    with open(script_path, "w") as f:
        f.write(_modal_script(req))
    threading.Thread(target=_run_modal, args=(req, script_path), daemon=True).start()
    return {"status": "deploying"}


@app.post("/api/modal/stop")
def modal_stop():
    if _modal_state["status"] == "idle":
        return {"status": "idle"}
    _modal_state["status"] = "stopping"
    _modal_state["logs"].append("Sending stop signal to Modal cloud...")

    token_id = _modal_state.get("token_id", "")
    token_secret = _modal_state.get("token_secret", "")

    def _do_stop():
        # Kill any lingering local deploy process first
        proc = _modal_state.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        if not (token_id and token_secret):
            _modal_state.update(status="idle", proc=None, ray_url=None)
            _modal_state["logs"].append("No credentials — cleared local state.")
            return

        env = os.environ.copy()
        env["MODAL_TOKEN_ID"] = token_id
        env["MODAL_TOKEN_SECRET"] = token_secret

        # Step 1: send `modal app stop`. With min_containers=0 in the
        # script, the running container drains within ~20s. The CLI
        # asks for interactive confirmation by default — pass --yes
        # to skip it (we have no TTY here, otherwise it exits 1
        # with "no interactive terminal detected").
        _modal_state["logs"].append("→ modal app stop --yes medimage-ray")
        try:
            r = subprocess.run(
                ["python3", "-m", "modal", "app", "stop", "--yes", "medimage-ray"],
                env=env, timeout=120, capture_output=True, text=True,
            )
            tail = (r.stdout + r.stderr).strip()
            if r.returncode == 0:
                _modal_state["logs"].append("✓ stop command accepted (exit 0)")
                if tail:
                    _modal_state["logs"].append(f"  output: {tail[-300:]}")
            else:
                _modal_state["logs"].append(f"⚠ stop exited {r.returncode}: {tail[-300:]}")
        except subprocess.TimeoutExpired:
            _modal_state["logs"].append("⚠ stop timed out after 120s")
        except Exception as exc:
            _modal_state["logs"].append(f"⚠ stop error: {exc}")

        # Step 2: poll `modal app list` for the app to actually leave
        # the *running* state. With min_containers=0 this is ~20s.
        # Note: `modal app list` keeps the app in the list with state
        # 'stopped' after a successful stop — it doesn't disappear.
        # We accept any of {stopped, stopping} as terminal, only
        # {deployed, running, active, initialized, ...} as "still alive".
        import time as _time
        _modal_state["logs"].append("→ polling `modal app list` for medimage-ray to leave the running state…")
        ALIVE_STATES = ("deployed", "running", "active", "initialized", "pending", "scheduled")
        deadline = _time.time() + 90
        while _time.time() < deadline:
            try:
                cl = subprocess.run(
                    ["python3", "-m", "modal", "app", "list"],
                    env=env, timeout=15, capture_output=True, text=True,
                )
                out = (cl.stdout or "") + (cl.stderr or "")
                still_alive = False
                state = ""
                for line in out.splitlines():
                    if "medimage-ray" in line.lower():
                        line_lower = line.lower()
                        for kw in ALIVE_STATES:
                            if kw in line_lower:
                                still_alive = True
                                state = kw
                                break
                        if not still_alive:
                            for kw in ("stopped", "stopping", "terminated", "disconnected"):
                                if kw in line.lower():
                                    state = kw
                                    break
                if not still_alive:
                    _modal_state["logs"].append(
                        f"✓ medimage-ray state is '{state or 'gone'}' — cluster is down"
                    )
                    _modal_state.update(status="idle", proc=None, ray_url=None, num_workers=0, gpu_type="T4")
                    return
                _modal_state["logs"].append(f"  still {state} — waiting…")
            except Exception as e:
                _modal_state["logs"].append(f"  poll error: {e}")
            _time.sleep(6)

        _modal_state["logs"].append(
            "⚠ Timed out after 90s waiting for the app to drop from the list. "
            "It may still be draining on modal.com — refresh the page and check /apps."
        )
        # Still flip to idle so the user isn't permanently stuck in "stopping"
        _modal_state.update(status="idle", proc=None, ray_url=None, num_workers=0, gpu_type="T4")

    threading.Thread(target=_do_stop, daemon=True).start()
    return {"status": "stopping"}


@app.get("/api/modal/status")
def modal_status_ep():
    return {
        "status": _modal_state["status"],
        "ray_url": _modal_state["ray_url"],
        "logs": _modal_state["logs"][-30:],
        "num_workers": _modal_state["num_workers"],
        "gpu_type":    _modal_state.get("gpu_type", "T4"),
    }


@app.post("/api/modal/scale")
def modal_scale(delta: int = Body(..., embed=True)):
    if _modal_state["status"] != "running":
        raise HTTPException(status_code=400, detail="Cluster not running — start it first")

    token_id = _modal_state.get("token_id", "")
    token_secret = _modal_state.get("token_secret", "")
    if not (token_id and token_secret):
        raise HTTPException(status_code=400, detail="No Modal credentials stored")

    new_workers = max(0, _modal_state["num_workers"] + delta)
    _modal_state["num_workers"] = new_workers

    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = token_id
    env["MODAL_TOKEN_SECRET"] = token_secret
    script_path = "/tmp/medimage_modal_ray.py"

    def _do_scale():
        if delta > 0:
            # Spawn new worker containers
            _modal_state["logs"].append(f"Spawning {delta} new worker(s)...")
            for _ in range(delta):
                subprocess.Popen(
                    ["python3", "-m", "modal", "run", "--detach",
                     script_path, "::ray_worker"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            _modal_state["logs"].append(
                f"Workers spawning \u2014 total workers: {new_workers}"
            )
        elif delta < 0:
            # Stop |delta| worker containers (skip head container)
            try:
                r = subprocess.run(
                    ["python3", "-m", "modal", "container", "list"],
                    env=env, capture_output=True, text=True, timeout=20,
                )
                lines = _strip_ansi(r.stdout + r.stderr).splitlines()
                # Worker containers show "ray_worker" in their entry
                worker_cids = [
                    l.split()[0] for l in lines
                    if "ray_worker" in l and l.split()
                ]
                to_stop = worker_cids[:abs(delta)]
                for cid in to_stop:
                    subprocess.run(
                        ["python3", "-m", "modal", "container", "stop", cid],
                        env=env, timeout=15, capture_output=True,
                    )
                    _modal_state["logs"].append(f"Stopped worker {cid[:12]}")
                if not to_stop:
                    _modal_state["logs"].append("No worker containers found to stop")
            except Exception as exc:
                _modal_state["logs"].append(f"Scale down error: {exc}")

    threading.Thread(target=_do_scale, daemon=True).start()
    return {"num_workers": new_workers}


@app.post("/api/modal/ray-stop")
def modal_ray_stop_ep():
    """Send `ray stop` to all running containers of the medimage-ray Modal app."""
    token_id = _modal_state.get("token_id", "")
    token_secret = _modal_state.get("token_secret", "")
    if not (token_id and token_secret):
        raise HTTPException(status_code=400, detail="No Modal credentials stored")

    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = token_id
    env["MODAL_TOKEN_SECRET"] = token_secret

    # 1. List running containers
    try:
        r = subprocess.run(
            ["python3", "-m", "modal", "container", "list"],
            env=env, capture_output=True, text=True, timeout=20,
        )
        output = _strip_ansi(r.stdout + r.stderr)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"container list failed: {exc}")

    # Parse container IDs that belong to medimage-ray
    container_ids = []
    for line in output.splitlines():
        # Modal container list columns: container_id  app_name  ...
        parts = line.split()
        if len(parts) >= 2 and "medimage-ray" in line:
            container_ids.append(parts[0])

    if not container_ids:
        return {"message": "No running containers found for medimage-ray", "containers": []}

    results = []
    for cid in container_ids:
        try:
            r2 = subprocess.run(
                ["python3", "-m", "modal", "container", "exec", cid, "ray", "stop"],
                env=env, capture_output=True, text=True, timeout=30,
            )
            out = _strip_ansi(r2.stdout + r2.stderr).strip()
            results.append({"container": cid, "exit_code": r2.returncode, "output": out})
            _modal_state["logs"].append(f"ray stop → {cid[:12]}: {out[:120]}")
        except subprocess.TimeoutExpired:
            results.append({"container": cid, "exit_code": -1, "output": "timed out"})
        except Exception as exc:
            results.append({"container": cid, "exit_code": -1, "output": str(exc)})

    return {"containers": results}


@app.get("/api/modal/nodes")
async def modal_nodes_ep():
    """Proxy Ray Dashboard /nodes endpoint from the Modal Ray cluster."""
    ray_url = _modal_state.get("ray_url")
    if not ray_url:
        raise HTTPException(status_code=404, detail="No Modal Ray cluster URL — start the cluster first")
    # Strip trailing slash; ray dashboard serves /nodes?view=summary
    base = ray_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10, verify=False) as client:
            r = await client.get(f"{base}/nodes?view=summary")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Modal Ray Dashboard: {e}")


# ─── Modal Inference Deploy (per-model) ───────────────────────────────────────

# Per-model deploy state (same pattern as Ray)
_modal_deploy_states: dict[str, dict] = {}

def _get_modal_deploy_state(job_id: str) -> dict:
    if job_id not in _modal_deploy_states:
        _modal_deploy_states[job_id] = {"status": "idle", "url": None, "logs": []}
    return _modal_deploy_states[job_id]


def _modal_model_script(
    job_id: str, training_type: str, engine: str, model_name: str,
    num_classes: int, class_names: list, gpu_type: str,
    minio_url: str, minio_access: str, minio_secret: str,
    weights_bucket: str, weights_key: str,
    volume_name: str = None,
) -> str:
    gpu_spec = f'gpu="{gpu_type}"' if gpu_type != "cpu" else "gpu=None"
    app_name = f"medimage-{job_id}"
    # Build pip list based on engine / training type
    pip_pkgs = ["fastapi", "uvicorn", "python-multipart", "pillow", "boto3", "numpy<2.0"]
    if training_type in ("detection", "segmentation") and engine == "Ultralytics":
        pip_pkgs += ["ultralytics"]
    elif engine == "MONAI":
        pip_pkgs += ["torch", "torchvision", "monai"]
    elif engine == "HuggingFace":
        pip_pkgs += ["torch", "torchvision", "transformers", "accelerate"]
    elif engine == "Segmentation Models PyTorch":
        pip_pkgs += ["torch", "torchvision", "segmentation-models-pytorch"]
    elif training_type in ("llm-text", "vlm-finetune"):
        pip_pkgs += ["torch", "transformers", "accelerate", "peft"]
    else:
        pip_pkgs += ["torch", "torchvision", "timm"]
    pip_str = repr(pip_pkgs)
    cls_json = json.dumps(class_names)

    # Volume mount or _bake_weights (run_function)
    if volume_name:
        vol_line = f"vol = modal.Volume.from_name({repr(volume_name)})"
        # single braces — this is a plain string, NOT an f-string
        volumes_spec = ', volumes={"/weights": vol}'
        run_bake = ""
    else:
        vol_line = ""
        volumes_spec = ""
        run_bake = "\n            .run_function(_bake_weights)"

    return textwrap.dedent(f"""
        import modal, os, io, zipfile, json, time
        from pathlib import Path
        from fastapi import Request

        TRAINING_TYPE  = {repr(training_type)}
        ENGINE         = {repr(engine)}
        MODEL_NAME     = {repr(model_name)}
        MODEL_ID       = {repr(job_id)}
        MINIO_URL      = {repr(minio_url)}
        MINIO_ACCESS   = {repr(minio_access)}
        MINIO_SECRET   = {repr(minio_secret)}
        WEIGHTS_BUCKET = {repr(weights_bucket)}
        WEIGHTS_KEY    = {repr(weights_key)}
        NUM_CLASSES    = {num_classes}
        CLASS_NAMES    = {cls_json}

        app = modal.App({repr(app_name)})
        {vol_line}

        WEIGHTS_DIR = "/weights"

        def _bake_weights():
            # Download weights from MinIO into the image at build time.
            # Values are inlined at script generation time (not read from globals).
            import boto3, zipfile, os
            from botocore.config import Config
            from pathlib import Path
            _minio_url    = {repr(minio_url)}
            _minio_access = {repr(minio_access)}
            _minio_secret = {repr(minio_secret)}
            _bucket       = {repr(weights_bucket)}
            _key          = {repr(weights_key)}
            if not _key:
                return
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            s3 = boto3.client("s3", endpoint_url=_minio_url,
                              aws_access_key_id=_minio_access, aws_secret_access_key=_minio_secret,
                              region_name="us-east-1", config=Config(s3={{"addressing_style": "path"}}))
            raw = Path(WEIGHTS_DIR) / "raw_weights"
            try:
                s3.download_file(_bucket, _key, str(raw))
                print(f"[modal-build] Downloaded weights: {{_key}} ({{raw.stat().st_size}} bytes)")
            except Exception as e:
                print(f"[modal-build] Could not download weights: {{e}}")
                return
            with open(raw, "rb") as f:
                magic = f.read(4)
            if magic[:2] == b"PK":
                out = Path(WEIGHTS_DIR) / "extracted"
                out.mkdir(exist_ok=True)
                with zipfile.ZipFile(raw) as z:
                    z.extractall(out)
                raw.unlink()
                print(f"[modal-build] Extracted zip to {{out}}")
            else:
                raw.rename(Path(WEIGHTS_DIR) / "model.pt")
                print(f"[modal-build] Saved model.pt")

        image = (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("libgl1-mesa-glx", "libglib2.0-0")
            .pip_install(*{pip_str}){run_bake}
        )

        def _download_weights():
            import zipfile
            from pathlib import Path
            if not WEIGHTS_KEY:
                return None
            extracted = Path(WEIGHTS_DIR) / "extracted"
            if extracted.exists():
                return extracted
            pt = Path(WEIGHTS_DIR) / "model.pt"
            if pt.exists():
                with open(pt, "rb") as _f:
                    _magic = _f.read(4)
                if _magic[:2] == b"PK":
                    extracted.mkdir(exist_ok=True)
                    with zipfile.ZipFile(pt) as _z:
                        _z.extractall(extracted)
                    return extracted
                return pt
            return None

        _model = None
        _tokenizer = None
        _hf_processor = None
        _hf_id2label = None
        _smp_engine = False

        def _load(wp):
            global _model, _tokenizer, _hf_processor, _hf_id2label, _smp_engine
            tt, eng = TRAINING_TYPE, ENGINE
            labels = CLASS_NAMES or [f"class_{{i}}" for i in range(NUM_CLASSES)]
            if eng == "HuggingFace" and tt in ("detection", "classification", "segmentation"):
                import torch
                from transformers import AutoImageProcessor
                id2label = {{i: n for i, n in enumerate(labels)}} if labels else {{}}
                num_cls = NUM_CLASSES
                _hf_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
                if wp is not None:
                    pt_path = wp if (hasattr(wp, "is_file") and wp.is_file()) else (wp / "model.pt" if hasattr(wp, "__truediv__") else wp)
                    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    id2label = ckpt.get("id2label", {{int(k): v for k, v in ckpt.get("config", {{}}).get("id2label", {{}}).items()}}) or id2label
                    num_cls = len(id2label) or NUM_CLASSES
                _hf_id2label = id2label
                label2id = {{v: k for k, v in id2label.items()}}
                if tt == "detection":
                    from transformers import AutoModelForObjectDetection
                    m = AutoModelForObjectDetection.from_pretrained(MODEL_NAME, num_labels=num_cls, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
                elif tt == "classification":
                    from transformers import AutoModelForImageClassification
                    m = AutoModelForImageClassification.from_pretrained(MODEL_NAME, num_labels=num_cls, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
                else:
                    from transformers import AutoModelForSemanticSegmentation
                    m = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME, num_labels=num_cls, ignore_mismatched_sizes=True)
                if wp is not None:
                    m.load_state_dict(ckpt["model_state_dict"], strict=False)
                _model = m.eval()
                return
            if tt in ("detection", "segmentation") and eng == "Ultralytics":
                from ultralytics import YOLO
                _model = YOLO(str(wp) if wp is not None else (MODEL_NAME or "yolov8n.pt"))
            elif tt == "classification" and eng != "MONAI":
                import torch, timm
                arch = MODEL_NAME.replace("-", "_")
                if wp is None:
                    _model = timm.create_model(arch, pretrained=True, num_classes=NUM_CLASSES).eval()
                else:
                    ckpt = torch.load(str(wp), map_location="cpu", weights_only=False)
                    if isinstance(ckpt, torch.nn.Module):
                        _model = ckpt.eval(); return
                    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    m = timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES)
                    m.load_state_dict(state, strict=False)
                    _model = m.eval()
            elif tt == "segmentation" and eng == "Segmentation Models PyTorch":
                import torch, segmentation_models_pytorch as smp
                if wp is None:
                    _model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=NUM_CLASSES).eval()
                    _smp_engine = True
                else:
                    ckpt_path = wp if wp.is_file() else (wp / "model.pt")
                    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                    arch = ckpt.get("arch", "unet")
                    enc  = ckpt.get("encoder", "resnet34")
                    nc   = ckpt.get("num_classes", NUM_CLASSES)
                    m = getattr(smp, arch.capitalize())(encoder_name=enc, in_channels=3, classes=nc)
                    m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                    _model = m.eval(); _smp_engine = True
            elif eng == "MONAI" and tt == "classification":
                import torch
                from monai.networks.nets import DenseNet121
                in_ch, n_cls = 1, NUM_CLASSES
                if wp is not None:
                    pt_path = wp if (hasattr(wp, "is_file") and wp.is_file()) else (wp / "model.pt" if hasattr(wp, "__truediv__") else wp)
                    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    for k, v in state.items():
                        if "conv" in k and len(v.shape) == 4: in_ch = v.shape[1]; break
                    for k, v in state.items():
                        if "out" in k and len(v.shape) == 2: n_cls = v.shape[0]
                    m = DenseNet121(spatial_dims=2, in_channels=in_ch, out_channels=n_cls)
                    m.load_state_dict(state, strict=False)
                    _model = m.eval()
                else:
                    _model = DenseNet121(spatial_dims=2, in_channels=in_ch, out_channels=NUM_CLASSES).eval()
            elif eng == "MONAI" and tt == "segmentation":
                import torch
                from monai.networks.nets import UNet
                in_ch = 1
                if wp is not None:
                    pt_path = wp if (hasattr(wp, "is_file") and wp.is_file()) else (wp / "model.pt" if hasattr(wp, "__truediv__") else wp)
                    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    for k, v in state.items():
                        if "conv" in k and len(v.shape) == 4: in_ch = v.shape[1]; break
                    m = UNet(spatial_dims=2, in_channels=in_ch, out_channels=NUM_CLASSES, channels=(16,32,64,128,256), strides=(2,2,2,2))
                    m.load_state_dict(state, strict=False)
                    _model = m.eval()
                else:
                    _model = UNet(spatial_dims=2, in_channels=in_ch, out_channels=NUM_CLASSES, channels=(16,32,64,128,256), strides=(2,2,2,2)).eval()
            elif tt in ("llm-text", "vlm-finetune"):
                import torch
                if tt == "vlm-finetune":
                    from transformers import AutoProcessor, AutoModelForVision2Seq
                    from peft import PeftModel
                    _hf_processor = AutoProcessor.from_pretrained(MODEL_NAME)
                    _model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
                else:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    from peft import PeftModel
                    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                    _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
                if wp is not None:
                    _model = PeftModel.from_pretrained(_model, str(wp))
            else:
                import torch
                if wp is not None:
                    pt = wp if (wp.is_file() if hasattr(wp, "is_file") else True) else next(wp.glob("*.pt"), None)
                    if pt:
                        _model = torch.load(str(pt), map_location="cpu", weights_only=False)

        @app.cls(image=image, {gpu_spec}, timeout=600, min_containers=1, scaledown_window=300{volumes_spec})
        class ModelService:
            @modal.enter()
            def startup(self):
                global _model
                print(f"[modal] Loading model {{MODEL_ID}} ({{TRAINING_TYPE}}/{{ENGINE}})…")
                wp = _download_weights()
                _load(wp)
                print(f"[modal] Model loaded: {{_model is not None}}")

            @modal.fastapi_endpoint(method="GET")
            def health(self):
                return {{"status": "ok", "model_id": MODEL_ID, "model_loaded": _model is not None}}

            @modal.fastapi_endpoint(method="POST")
            async def inference(self, request: Request):
                import io as _io
                body = await request.form()
                tt = TRAINING_TYPE
                conf = float(body.get("conf_threshold", 0.5))
                prompt = str(body.get("prompt", ""))
                system_prompt = str(body.get("system_prompt", ""))
                image_bytes = None
                if "image" in body:
                    _img_file = body["image"]
                    image_bytes = await _img_file.read()

                if _model is None:
                    return {{"error": "Model not loaded"}}
                labels = CLASS_NAMES or [f"class_{{i}}" for i in range(NUM_CLASSES)]
                try:
                    if _hf_processor is not None:
                        from PIL import Image as _Img
                        import torch
                        img = _Img.open(_io.BytesIO(image_bytes)).convert("RGB")
                        inputs = _hf_processor(images=img, return_tensors="pt")
                        with torch.no_grad():
                            outputs = _model(**inputs)
                        if tt == "detection":
                            target_sizes = torch.tensor([img.size[::-1]])
                            res = _hf_processor.post_process_object_detection(outputs, threshold=conf, target_sizes=target_sizes)[0]
                            dets = [{{"class_name": (_hf_id2label or {{}}).get(label.item(), f"class_{{label.item()}}"), "confidence": round(float(score), 4), "bbox": [x1, y1, x2-x1, y2-y1]}} for score, label, (x1,y1,x2,y2) in zip(res["scores"], res["labels"], res["boxes"].tolist())]
                            return {{"type": "detection", "detections": dets}}
                        elif tt == "classification":
                            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
                            preds = sorted([{{"label": (_hf_id2label or {{}}).get(i, f"class_{{i}}"), "confidence": p}} for i, p in enumerate(probs)], key=lambda x: -x["confidence"])
                            return {{"type": "classification", "predictions": preds[:5], "top_label": preds[0]["label"]}}
                        elif tt == "segmentation":
                            import torch.nn.functional as F
                            logits = outputs.logits
                            up = F.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
                            mask = up.argmax(dim=1)[0]
                            total_px = mask.numel()
                            id2l = _hf_id2label or {{}}
                            segs = [{{"class_name": id2l.get(ci, f"class_{{ci}}"), "class_id": ci, "pixel_count": int((mask==ci).sum()), "coverage": round(int((mask==ci).sum())/total_px, 4)}} for ci in mask.unique().tolist()]
                            return {{"type": "segmentation", "segments": sorted(segs, key=lambda x: -x["pixel_count"])}}
                    if tt in ("detection", "segmentation") and hasattr(_model, "predict"):
                        from PIL import Image as _Img
                        img = _Img.open(_io.BytesIO(image_bytes)).convert("RGB")
                        results = _model.predict(img, conf=conf, verbose=False)
                        if tt == "detection":
                            dets = []
                            for r in results:
                                iw, ih = img.size
                                for box in r.boxes:
                                    ci = int(box.cls[0])
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    dets.append({{"class_name": labels[ci] if ci < len(labels) else f"class_{{ci}}", "confidence": float(box.conf[0]), "bbox": [x1/iw, y1/ih, x2/iw, y2/ih]}})
                            return {{"type": "detection", "detections": dets}}
                        else:
                            masks = []
                            for r in results:
                                if r.masks is None: continue
                                for i, _ in enumerate(r.masks):
                                    ci = int(r.boxes.cls[i])
                                    masks.append({{"class_name": labels[ci] if ci < len(labels) else f"class_{{ci}}", "confidence": float(r.boxes.conf[i])}})
                            return {{"type": "segmentation", "masks": masks}}
                    if tt == "segmentation" and _smp_engine:
                        import torch, torchvision.transforms as T
                        from PIL import Image as _Img
                        img = _Img.open(_io.BytesIO(image_bytes)).convert("RGB")
                        tfm = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                        with torch.no_grad():
                            logits = _model(tfm(img).unsqueeze(0))
                        mask = logits.argmax(dim=1)[0]
                        total_px = mask.numel()
                        segs = [{{"class_name": labels[ci] if ci < len(labels) else f"class_{{ci}}", "class_id": ci, "pixel_count": int((mask==ci).sum()), "coverage": round(int((mask==ci).sum())/total_px, 4)}} for ci in mask.unique().tolist()]
                        return {{"type": "segmentation", "segments": sorted(segs, key=lambda x: -x["pixel_count"])}}
                    if tt == "classification":
                        import torch, torchvision.transforms as T
                        from PIL import Image as _Img
                        img = _Img.open(_io.BytesIO(image_bytes))
                        if ENGINE == "MONAI":
                            import numpy as np
                            in_ch = 1
                            for p in _model.parameters():
                                if len(p.shape) == 4: in_ch = p.shape[1]; break
                            img_arr = np.array(img.convert("L") if in_ch == 1 else img.convert("RGB"), dtype=np.float32)
                            img_t = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0) if in_ch == 1 else torch.tensor(img_arr.transpose(2,0,1)).unsqueeze(0)
                            img_t = torch.nn.functional.interpolate(img_t, size=(224,224), mode="bilinear", align_corners=False)
                            img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8)
                            with torch.no_grad():
                                probs = torch.softmax(_model(img_t), dim=1)[0].tolist()
                        else:
                            tfm = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                            with torch.no_grad():
                                probs = torch.softmax(_model(tfm(img.convert("RGB")).unsqueeze(0)), dim=1)[0].tolist()
                        preds = sorted([{{"label": labels[i] if i < len(labels) else f"class_{{i}}", "confidence": p}} for i, p in enumerate(probs)], key=lambda x: -x["confidence"])
                        return {{"type": "classification", "predictions": preds, "top_label": preds[0]["label"]}}
                    if tt in ("llm-text", "vlm-finetune"):
                        import torch
                        full_prompt = (system_prompt + "\\n" + prompt).strip() if system_prompt else prompt
                        inputs = _tokenizer(full_prompt, return_tensors="pt").to(_model.device)
                        t0 = time.time()
                        with torch.no_grad():
                            out = _model.generate(**inputs, max_new_tokens=512, do_sample=False)
                        resp = _tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                        return {{"type": tt, "response": resp, "tokens_generated": out.shape[1]-inputs.input_ids.shape[1], "inference_time_ms": int((time.time()-t0)*1000)}}
                    return {{"error": f"No inference handler for engine={{ENGINE}}, type={{tt}}"}}
                except Exception as exc:
                    return {{"error": str(exc)}}
    """)


def _run_modal_model_deploy(job_id: str, token_id: str, token_secret: str, gpu_type: str,
                             training_type: str, engine: str, model_name: str,
                             num_classes: int, class_names: list,
                             minio_url: str, minio_access: str, minio_secret: str,
                             weights_bucket: str, weights_key: str) -> None:
    state = _get_modal_deploy_state(job_id)
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = token_id
    env["MODAL_TOKEN_SECRET"] = token_secret

    # Download weights from MinIO (API server is in Docker network → can reach internal MinIO)
    # then push to Modal Volume so Modal cloud can access them
    volume_name = None
    if weights_key:
        state["logs"].append("[modal] Downloading weights from MinIO…")
        try:
            import boto3 as _boto3, tempfile as _tempfile
            from botocore.config import Config as _BotoConfig
            _s3 = _boto3.client(
                "s3", endpoint_url=minio_url,
                aws_access_key_id=minio_access, aws_secret_access_key=minio_secret,
                region_name="us-east-1", config=_BotoConfig(s3={"addressing_style": "path"}),
            )
            _suffix = ".zip" if weights_key.endswith(".zip") else ".pt"
            with _tempfile.NamedTemporaryFile(suffix=_suffix, delete=False) as _tmp:
                _s3.download_fileobj(weights_bucket, weights_key, _tmp)
                _local_weights = _tmp.name
            _sz = os.path.getsize(_local_weights)
            state["logs"].append(f"[modal] Downloaded {_sz} bytes — pushing to Modal Volume…")
            volume_name = f"medimage-{job_id}"
            subprocess.run(
                ["python3", "-m", "modal", "volume", "create", volume_name],
                env=env, capture_output=True, text=True,
            )
            _proc_vol = subprocess.run(
                ["python3", "-m", "modal", "volume", "put", "--force",
                 volume_name, _local_weights, "/model.pt"],
                env=env, capture_output=True, text=True, timeout=300,
            )
            os.unlink(_local_weights)
            if _proc_vol.returncode == 0:
                state["logs"].append(f"[modal] Weights pushed to Modal Volume '{volume_name}'")
            else:
                _err = (_proc_vol.stderr or _proc_vol.stdout or "")[:300]
                state["logs"].append(f"[modal] Volume push failed: {_err} — falling back to runtime download")
                volume_name = None
        except Exception as _e:
            state["logs"].append(f"[modal] Weight download failed: {_e} — falling back to runtime download")
            volume_name = None

    script_path = f"/tmp/medimage_modal_{job_id}.py"
    script = _modal_model_script(
        job_id=job_id, training_type=training_type, engine=engine,
        model_name=model_name, num_classes=num_classes, class_names=class_names,
        gpu_type=gpu_type, minio_url=minio_url, minio_access=minio_access,
        minio_secret=minio_secret, weights_bucket=weights_bucket, weights_key=weights_key,
        volume_name=volume_name,
    )
    with open(script_path, "w") as f:
        f.write(script)
    state["logs"].append(f"[modal] Script written, running modal deploy…")
    try:
        proc = subprocess.run(
            ["python3", "-m", "modal", "deploy", script_path],
            env=env, capture_output=True, text=True, timeout=360,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        # Join lines that are continuations (Modal wraps long URLs across lines with │ prefix)
        # Also join lines split with spaces/newlines inside URL blocks
        joined_output = _re.sub(r'\n\s*[│|]\s*', '', output)
        joined_output = _re.sub(r'\s*\n\s{4,}', '', joined_output)  # join indented continuations
        logs = [l for l in output.splitlines() if l.strip()]
        state["logs"] = logs[-40:]
        # Try to find inference endpoint URL (prefer /inference over /health)
        urls = _re.findall(r'https://[^\s│|<>]+\.modal\.run', joined_output)
        urls = [u.rstrip('/.,;)>') for u in urls]
        # Store inference URL as primary (for calling the model), fall back to health URL
        infer_url = next((u for u in urls if 'inference' in u), None)
        health_url = next((u for u in urls if 'health' in u), None)
        url = infer_url or health_url or (urls[0] if urls else None)
        if url:
            state.update(status="running", url=url)
            state["logs"].append(f"[modal] Deployed at {url}")
            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET modal_url=?, inference_provider='modal', "
                    "       updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (url, job_id),
                )
                conn.commit()
            # Background warmup — hit the health endpoint so Modal spins up the container
            # and loads the model before the first user request
            _warmup_url = health_url or url.replace("inference", "health")
            def _warmup(wu: str) -> None:
                import time as _time, requests as _req
                _time.sleep(5)  # let deploy settle
                try:
                    r = _req.get(wu, timeout=300, allow_redirects=True, verify=False)
                    state["logs"].append(f"[modal] Warmup complete (HTTP {r.status_code}) — container is ready")
                except Exception as _we:
                    state["logs"].append(f"[modal] Warmup ping: {_we}")
            threading.Thread(target=_warmup, args=(_warmup_url,), daemon=True).start()
        elif proc.returncode == 0:
            state.update(status="error")
            state["logs"].append("[modal] Deploy succeeded but could not parse URL from output")
        else:
            state.update(status="error")
            state["logs"].append(f"[modal] Deploy failed (exit {proc.returncode})")
    except subprocess.TimeoutExpired:
        state.update(status="error")
        state["logs"].append("[modal] Deploy timed out after 6 minutes")
    except Exception as exc:
        state.update(status="error")
        state["logs"].append(f"[modal] Exception: {exc}")


@app.post("/api/jobs/{job_id}/deploy-modal")
def model_deploy_modal(job_id: str, body: dict):
    """Deploy a model to Modal.com as a per-model web endpoint."""
    # Use body creds if provided, else fall back to saved DB credentials
    token_id = (body.get("token_id") or "").strip()
    token_secret = (body.get("token_secret") or "").strip()
    if not token_id or not token_secret:
        saved = _load_modal_creds()
        if saved:
            token_id = saved["token_id"]
            token_secret = saved["token_secret"]
    if not token_id or not token_secret:
        raise HTTPException(400, "No Modal credentials — save them in Modal Configuration first")
    # Look up job for metadata
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Model not found")
    state = _get_modal_deploy_state(job_id)
    if state["status"] == "deploying":
        raise HTTPException(409, "Deploy already in progress")
    gpu_type    = body.get("gpu_type", "T4")
    num_workers = int(body.get("num_workers", 1) or 1)
    state.update(status="deploying", url=None, gpu_type=gpu_type, num_workers=num_workers,
                 logs=["Starting Modal deploy…"])
    minio_url = MINIO_URL  # internal Docker URL — download happens on API server (not Modal cloud)
    minio_access = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    s3_path = (row["s3_weights_path"] or "").strip()
    # Handle both "bucket/key" and "s3://bucket/key" formats
    if s3_path.startswith("s3://"):
        s3_path_stripped = s3_path[5:]  # remove "s3://"
    else:
        s3_path_stripped = s3_path
    weights_bucket = s3_path_stripped.split("/")[0] if "/" in s3_path_stripped else "medimage-weights"
    weights_key    = s3_path_stripped.split("/", 1)[1] if "/" in s3_path_stripped else s3_path_stripped
    # Parse class names from DB job notes if present
    try:
        notes = json.loads(row["notes"] or "{}")
        class_names = notes.get("class_names", [])
        num_classes = notes.get("num_classes", 2)
    except Exception:
        class_names = []
        num_classes = 2
    threading.Thread(
        target=_run_modal_model_deploy,
        args=(
            job_id, token_id, token_secret,
            gpu_type,
            row["training_type"] or "", row["engine"] or "", row["model_name"] or "",
            num_classes, class_names,
            minio_url, minio_access, minio_secret,
            weights_bucket, weights_key,
        ),
        daemon=True,
    ).start()
    return {"status": "deploying"}


@app.get("/api/jobs/{job_id}/deploy-modal/status")
def model_deploy_modal_status(job_id: str):
    state = _get_modal_deploy_state(job_id)
    # Sync from DB if running
    if state["status"] == "idle":
        with get_db() as conn:
            row = conn.execute("SELECT modal_url, inference_provider FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row and row["modal_url"] and row["inference_provider"] == "modal":
            state.update(status="running", url=row["modal_url"])
    return {"status": state["status"], "url": state.get("url"), "logs": state["logs"][-30:]}


@app.post("/api/jobs/{job_id}/deploy-modal/stop")
def model_deploy_modal_stop(job_id: str, body: dict = {}):
    """Stop the Modal app for this model."""
    app_name = f"medimage-{job_id}"
    logs = []
    token_id = (body.get("token_id") or "").strip()
    token_secret = (body.get("token_secret") or "").strip()
    if not token_id or not token_secret:
        saved = _load_modal_creds()
        if saved:
            token_id = saved["token_id"]
            token_secret = saved["token_secret"]
    if token_id and token_secret:
        env = os.environ.copy()
        env["MODAL_TOKEN_ID"] = token_id
        env["MODAL_TOKEN_SECRET"] = token_secret
        try:
            # Find the deployed app ID by listing apps and matching description
            list_proc = subprocess.run(
                ["python3", "-m", "modal", "app", "list"],
                env=env, capture_output=True, text=True, timeout=30,
            )
            app_id = None
            for line in list_proc.stdout.splitlines():
                # Lines look like: │ ap-XXXX │ medimage-c21fc0f1… │ deployed │ ...
                if "deployed" in line and app_name[:20] in line:
                    parts = [p.strip() for p in line.split("│") if p.strip()]
                    if parts and parts[0].startswith("ap-"):
                        app_id = parts[0]
                        break
            identifier = app_id or app_name
            logs.append(f"[modal stop] Stopping {identifier}…")
            proc = subprocess.run(
                ["python3", "-m", "modal", "app", "stop", "-y", identifier],
                env=env, capture_output=True, text=True, timeout=60,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            logs += [l for l in output.splitlines() if l.strip()][-10:]
            logs.append(f"[modal stop] exit code {proc.returncode}")
        except Exception as exc:
            logs.append(f"[modal stop] {exc}")
    else:
        logs.append("[modal stop] No token provided — skipping modal app stop")
    # Clear DB
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET modal_url='', inference_provider='' WHERE id=?",
            (job_id,),
        )
        conn.commit()
    state = _get_modal_deploy_state(job_id)
    state.update(status="idle", url=None, logs=logs)
    return {"ok": True, "logs": logs}


# Legacy: keep old global endpoint for backward compat (used by RayCluster page modal tab)
_modal_infer_state: dict = {"status": "idle", "url": None, "logs": []}

@app.post("/api/modal/inference/deploy")
def modal_infer_deploy(body: dict):
    raise HTTPException(400, "Use /api/jobs/{job_id}/deploy-modal for per-model deployment")

@app.get("/api/modal/inference/status")
def modal_infer_status_ep():
    return {"status": _modal_infer_state["status"], "url": _modal_infer_state["url"], "logs": []}


# ─── Ray Serve Inference Deploy ───────────────────────────────────────────────

_ray_serve_state: dict = {
    "status": "idle",   # idle | deploying | running | error
    "url": None,
    "logs": [],
}

_SERVE_APP_PY = textwrap.dedent("""
    from ray import serve
    from fastapi import FastAPI, File, Form, UploadFile
    from typing import Optional

    _app = FastAPI()

    @serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
    @serve.ingress(_app)
    class InferenceApp:
        def __init__(self):
            self.model = None

        @_app.post("/inference")
        async def infer(
            self,
            training_type: str = Form(...),
            model_id: str = Form(""),
            conf_threshold: float = Form(0.5),
            image: Optional[UploadFile] = File(None),
            prompt: str = Form(""),
            system_prompt: str = Form(""),
        ):
            if training_type in ("llm-text", "vlm-finetune"):
                return {"type": training_type, "response": f"Echo: {prompt[:80]}", "tokens": 0}
            return {"type": training_type, "predictions": [], "simulated": False}

        @_app.get("/health")
        async def health(self):
            return {"status": "ok", "model_loaded": self.model is not None}
""")

# ─── Per-model Ray Serve script ───────────────────────────────────────────────
_SERVE_MODEL_PY = textwrap.dedent("""
    import os, io, zipfile, json, time
    from pathlib import Path

    # ── env vars ──────────────────────────────────────────────────────────────
    TRAINING_TYPE  = os.environ["TRAINING_TYPE"]
    ENGINE         = os.environ.get("ENGINE", "Ultralytics")
    MODEL_NAME     = os.environ.get("MODEL_NAME", "")
    MODEL_ID       = os.environ.get("MODEL_ID", "model")
    MINIO_URL      = os.environ.get("MINIO_URL", "http://localhost:9000")
    MINIO_ACCESS   = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET   = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    WEIGHTS_BUCKET = os.environ.get("WEIGHTS_BUCKET", "medimage-weights")
    WEIGHTS_KEY    = os.environ.get("WEIGHTS_KEY", "")
    NUM_CLASSES    = int(os.environ.get("NUM_CLASSES", "2"))
    CLASS_NAMES    = json.loads(os.environ.get("CLASS_NAMES", "[]")) or [f"class_{i}" for i in range(NUM_CLASSES)]
    ROUTE_PREFIX   = f"/model/{MODEL_ID}"

    def _download_weights():
        import boto3
        from botocore.exceptions import ClientError
        if not WEIGHTS_KEY:
            return None
        s3 = boto3.client("s3", endpoint_url=MINIO_URL,
                          aws_access_key_id=MINIO_ACCESS, aws_secret_access_key=MINIO_SECRET,
                          region_name="us-east-1",
                          config=boto3.session.Config(s3={"addressing_style": "path"}))
        local_dir = Path(f"/tmp/mweights/{MODEL_ID}")
        local_dir.mkdir(parents=True, exist_ok=True)
        raw = local_dir / "raw_weights"
        try:
            s3.download_file(WEIGHTS_BUCKET, WEIGHTS_KEY, str(raw))
        except ClientError as _ce:
            _code = _ce.response.get("Error", {}).get("Code", "")
            if _code in ("404", "NoSuchKey", "400", "BadRequest"):
                print(f"[serve] No fine-tuned weights found at {WEIGHTS_KEY} (code={_code}) — will load pretrained")
                return None
            raise
        with open(raw, "rb") as f:
            magic = f.read(4)
        if magic[:2] == b"PK":
            out = local_dir / "extracted"
            out.mkdir(exist_ok=True)
            with zipfile.ZipFile(raw) as z:
                z.extractall(out)
            return out
        else:
            pt = local_dir / "model.pt"
            raw.rename(pt)
            return pt

    # ── Init Ray and deploy ───────────────────────────────────────────────────
    import ray
    from ray import serve

    _ray_address = os.environ.get("RAY_URL", "ray://100.68.53.118:10001")
    _ray_address = _ray_address.replace("http://", "ray://").replace(":8265", ":10001") if _ray_address else _ray_address
    ray.init(address=_ray_address, ignore_reinit_error=True, log_to_driver=False)

    # ── Deployment class — plain __call__, NO @serve.ingress(FastAPI()) ───────
    @serve.deployment(
        num_replicas=1,
        ray_actor_options={
            "num_cpus": 1,
            "resources": _node_resource,
        },
        name=f"medimage-{MODEL_ID}",
    )
    class ModelInference:
        def __init__(self):
            self.model = None
            self.tokenizer = None
            self._hf_processor = None
            self._hf_id2label = None
            self._smp_engine = False
            self.labels = CLASS_NAMES
            self._load_error = None
            try:
                weights = _download_weights()
                self._load(weights)
                print(f"[serve] Model {MODEL_ID} ({TRAINING_TYPE}/{ENGINE}) loaded (fine-tuned={weights is not None})")
            except Exception as _e:
                self._load_error = str(_e)
                print(f"[serve] WARNING: Model load failed for {MODEL_ID}: {_e}")

        def _load(self, wp):
            # wp=None means load pretrained from MODEL_NAME
            tt, eng = TRAINING_TYPE, ENGINE
            if eng == "HuggingFace" and tt in ("detection", "classification", "segmentation"):
                import torch
                from transformers import AutoImageProcessor
                id2label = {i: n for i, n in enumerate(CLASS_NAMES)} if CLASS_NAMES else {}
                num_cls = NUM_CLASSES
                self._hf_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
                if wp is not None:
                    pt_path = wp if (hasattr(wp, "is_file") and wp.is_file()) else (wp / "model.pt" if hasattr(wp, "__truediv__") else wp)
                    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    cfg_dict = ckpt.get("config", {})
                    id2label = ckpt.get("id2label", {int(k): v for k, v in cfg_dict.get("id2label", {}).items()}) or id2label
                    num_cls = len(id2label) or NUM_CLASSES
                self._hf_id2label = id2label
                label2id = {v: k for k, v in id2label.items()}
                if tt == "detection":
                    from transformers import AutoModelForObjectDetection
                    m = AutoModelForObjectDetection.from_pretrained(
                        MODEL_NAME, num_labels=num_cls,
                        id2label=id2label, label2id=label2id,
                        ignore_mismatched_sizes=True)
                elif tt == "classification":
                    from transformers import AutoModelForImageClassification
                    m = AutoModelForImageClassification.from_pretrained(
                        MODEL_NAME, num_labels=num_cls,
                        id2label=id2label, label2id=label2id,
                        ignore_mismatched_sizes=True)
                else:
                    from transformers import AutoModelForSemanticSegmentation
                    m = AutoModelForSemanticSegmentation.from_pretrained(
                        MODEL_NAME, num_labels=num_cls, ignore_mismatched_sizes=True)
                if wp is not None:
                    m.load_state_dict(ckpt["model_state_dict"], strict=False)
                self.model = m.eval()
                return
            if tt in ("detection", "segmentation") and eng == "Ultralytics":
                from ultralytics import YOLO
                # wp=None: load pretrained YOLO from MODEL_NAME (e.g. yolov8n.pt)
                self.model = YOLO(str(wp) if wp is not None else (MODEL_NAME or "yolov8n.pt"))
            elif tt == "classification":
                import torch, timm
                arch = MODEL_NAME.replace("-", "_")
                if wp is None:
                    self.model = timm.create_model(arch, pretrained=True, num_classes=NUM_CLASSES).eval()
                else:
                    ckpt = torch.load(str(wp), map_location="cpu", weights_only=False)
                    if isinstance(ckpt, torch.nn.Module):
                        self.model = ckpt.eval()
                        return
                    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    m = timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES)
                    m.load_state_dict(state, strict=False)
                    self.model = m.eval()
            elif tt == "segmentation" and eng == "Segmentation Models PyTorch":
                import torch, segmentation_models_pytorch as smp
                if wp is None:
                    m = smp.Unet(encoder_name="resnet34", in_channels=3, classes=NUM_CLASSES)
                    self.model = m.eval()
                    self._smp_engine = True
                else:
                    ckpt_path = wp if wp.is_file() else (wp / "model.pt")
                    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                    arch = ckpt.get("arch", "unet")
                    enc  = ckpt.get("encoder", "resnet34")
                    nc   = ckpt.get("num_classes", NUM_CLASSES)
                    m = getattr(smp, arch.capitalize())(encoder_name=enc, in_channels=3, classes=nc)
                    m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                    self.model = m.eval()
                    self._smp_engine = True
            elif tt in ("llm-text", "vlm-finetune"):
                import torch
                if tt == "vlm-finetune":
                    from transformers import AutoProcessor, AutoModelForVision2Seq
                    from peft import PeftModel
                    self._hf_processor = AutoProcessor.from_pretrained(MODEL_NAME)
                    self.model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
                else:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    from peft import PeftModel
                    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                    self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
                if wp is not None:
                    self.model = PeftModel.from_pretrained(self.model, str(wp))
            else:
                import torch
                if wp is not None:
                    pt = wp if (wp.is_file() if hasattr(wp, "is_file") else True) else next(wp.glob("*.pt"), None)
                    if pt:
                        self.model = torch.load(str(pt), map_location="cpu", weights_only=False)

        async def _infer(self, image_bytes=None, conf=0.5, prompt="", system_prompt=""):
            if self.model is None:
                return {"error": self._load_error or "Model not loaded"}
            tt = TRAINING_TYPE
            try:
                # ── 1. LLM / VLM text generation (must be before HF image check) ──
                if tt in ("llm-text", "vlm-finetune"):
                    import torch
                    if not prompt:
                        return {"error": "Prompt required for LLM inference"}
                    full_prompt = (system_prompt + "\\n" + prompt).strip() if system_prompt else prompt
                    if tt == "vlm-finetune" and self._hf_processor is not None:
                        from PIL import Image as _PILImage
                        import io as _io
                        img_obj = None
                        if image_bytes:
                            img_obj = _PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
                            messages = [{"role": "user", "content": [
                                {"type": "image"},
                                {"type": "text", "text": full_prompt},
                            ]}]
                        else:
                            messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
                        chat_text = self._hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = self._hf_processor(text=[chat_text], images=[img_obj] if img_obj else None, return_tensors="pt").to(self.model.device)
                    else:
                        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                    t0 = time.time()
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    ms = int((time.time() - t0) * 1000)
                    n_tok = out.shape[1] - inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else 0
                    processor = self._hf_processor if (tt == "vlm-finetune" and self._hf_processor) else self.tokenizer
                    resp = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True) if hasattr(inputs, "input_ids") else processor.decode(out[0], skip_special_tokens=True)
                    return {"type": tt, "response": resp, "tokens_generated": n_tok,
                            "inference_time_ms": ms, "model_name": MODEL_NAME}

                # ── 2. HuggingFace image models (detection / classification / segmentation) ──
                if self._hf_processor is not None:
                    if not image_bytes:
                        return {"error": "Image required"}
                    from PIL import Image as _Img
                    import torch
                    img = _Img.open(io.BytesIO(image_bytes)).convert("RGB")
                    inputs = self._hf_processor(images=img, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    if tt == "detection":
                        target_sizes = torch.tensor([img.size[::-1]])
                        res = self._hf_processor.post_process_object_detection(
                            outputs, threshold=conf, target_sizes=target_sizes)[0]
                        dets = []
                        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                            name = (self._hf_id2label or {}).get(label.item(), f"class_{label.item()}")
                            x1, y1, x2, y2 = box.tolist()
                            dets.append({"class_name": name, "confidence": round(float(score), 4),
                                         "bbox": [x1, y1, x2 - x1, y2 - y1]})
                        return {"type": "detection", "detections": dets}
                    elif tt == "classification":
                        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
                        id2label = self._hf_id2label or {}
                        preds = sorted([{"label": id2label.get(i, f"class_{i}"), "confidence": p}
                                        for i, p in enumerate(probs)], key=lambda x: -x["confidence"])
                        return {"type": "classification", "predictions": preds[:5],
                                "top_label": preds[0]["label"]}
                    elif tt == "segmentation":
                        logits = outputs.logits
                        up = torch.nn.functional.interpolate(
                            logits, size=img.size[::-1], mode="bilinear", align_corners=False)
                        mask = up.argmax(dim=1)[0]
                        total_px = mask.numel()
                        id2label = self._hf_id2label or {}
                        segs = []
                        for ci in mask.unique().tolist():
                            px = int((mask == ci).sum())
                            segs.append({"class_name": id2label.get(ci, f"class_{ci}"),
                                         "class_id": ci, "pixel_count": px,
                                         "coverage": round(px / total_px, 4)})
                        segs.sort(key=lambda x: -x["pixel_count"])
                        return {"type": "segmentation", "segments": segs, "num_classes": len(segs)}
                    else:
                        return {"error": f"HuggingFace inference not implemented for type={tt}"}

                # ── 3. Ultralytics YOLO (detection + instance segmentation) ──
                if tt in ("detection", "segmentation") and hasattr(self.model, "predict"):
                    if not image_bytes:
                        return {"error": "Image required"}
                    from PIL import Image as _Img
                    img = _Img.open(io.BytesIO(image_bytes)).convert("RGB")
                    results = self.model.predict(img, conf=conf, verbose=False)
                    if tt == "detection":
                        dets = []
                        for r in results:
                            for box in r.boxes:
                                ci = int(box.cls[0])
                                name = self.labels[ci] if ci < len(self.labels) else f"class_{ci}"
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                dets.append({"class_name": name, "confidence": float(box.conf[0]),
                                             "bbox": [x1, y1, x2 - x1, y2 - y1]})
                        return {"type": "detection", "detections": dets}
                    else:
                        masks = []
                        for r in results:
                            if r.masks is None: continue
                            for i, _ in enumerate(r.masks):
                                ci = int(r.boxes.cls[i])
                                name = self.labels[ci] if ci < len(self.labels) else f"class_{ci}"
                                masks.append({"class_name": name, "confidence": float(r.boxes.conf[i])})
                        return {"type": "segmentation", "masks": masks, "num_masks": len(masks)}

                # ── 4. Segmentation Models PyTorch (semantic segmentation) ──
                if tt == "segmentation" and self._smp_engine:
                    import torch
                    import torchvision.transforms as T
                    from PIL import Image as _Img
                    if not image_bytes:
                        return {"error": "Image required"}
                    img = _Img.open(io.BytesIO(image_bytes)).convert("RGB")
                    orig_w, orig_h = img.size
                    tfm = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    with torch.no_grad():
                        logits = self.model(tfm(img).unsqueeze(0))
                    mask = logits.argmax(dim=1)[0]
                    total_px = mask.numel()
                    segs = []
                    for ci in mask.unique().tolist():
                        px = int((mask == ci).sum())
                        name = self.labels[ci] if ci < len(self.labels) else f"class_{ci}"
                        segs.append({"class_name": name, "class_id": ci,
                                     "pixel_count": px, "coverage": round(px / total_px, 4)})
                    segs.sort(key=lambda x: -x["pixel_count"])
                    return {"type": "segmentation", "segments": segs,
                            "num_classes": len(segs), "image_size": [orig_w, orig_h]}

                # ── 5. PyTorch / TIMM classification ──
                if tt == "classification":
                    import torch
                    import torchvision.transforms as T
                    from PIL import Image as _Img
                    if not image_bytes:
                        return {"error": "Image required"}
                    img = _Img.open(io.BytesIO(image_bytes)).convert("RGB")
                    tfm = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    with torch.no_grad():
                        logits = self.model(tfm(img).unsqueeze(0))
                        probs = torch.softmax(logits, dim=1)[0].tolist()
                    preds = sorted([{"label": self.labels[i] if i < len(self.labels) else f"class_{i}",
                                     "confidence": p} for i, p in enumerate(probs)],
                                   key=lambda x: -x["confidence"])
                    return {"type": "classification", "predictions": preds, "top_label": preds[0]["label"]}

                return {"error": f"No inference handler for engine={ENGINE}, type={tt}"}
            except Exception as exc:
                return {"error": str(exc)}

        async def __call__(self, request):
            from starlette.responses import JSONResponse
            path = request.url.path
            method = request.method
            if method == "GET" and path.endswith("/health"):
                return JSONResponse({"status": "ok", "model_loaded": self.model is not None,
                                     "load_error": self._load_error,
                                     "model_id": MODEL_ID, "training_type": TRAINING_TYPE,
                                     "engine": ENGINE})
            if method != "POST":
                return JSONResponse({"error": "Not found"}, status_code=404)
            # Parse request
            ct = request.headers.get("content-type", "")
            conf, img_bytes, prompt, system_prompt = 0.5, None, "", ""
            if "multipart/form-data" in ct:
                form = await request.form()
                conf = float(form.get("conf_threshold", 0.5))
                prompt = form.get("prompt", "")
                system_prompt = form.get("system_prompt", "")
                img_file = form.get("image")
                if img_file and hasattr(img_file, "read"):
                    img_bytes = await img_file.read()
            elif "application/json" in ct:
                data = await request.json()
                conf = float(data.get("conf_threshold", 0.5))
                prompt = data.get("prompt", "")
                system_prompt = data.get("system_prompt", "")
                if "image" in data:
                    import base64 as _b64
                    img_bytes = _b64.b64decode(data["image"])
            else:
                img_bytes = await request.body()
            result = await self._infer(image_bytes=img_bytes, conf=conf,
                                       prompt=prompt, system_prompt=system_prompt)
            if isinstance(result, dict) and result.get("error"):
                return JSONResponse(result, status_code=500)
            return JSONResponse(result)

        async def infer_raw(self, image_bytes: bytes = None, conf: float = 0.5,
                            prompt: str = "", system_prompt: str = "") -> dict:
            # Direct Python call via Ray Client (bypasses HTTP firewall on port 8000)
            return await self._infer(image_bytes=image_bytes, conf=conf,
                                     prompt=prompt, system_prompt=system_prompt)

    # ── Deploy to Ray Serve (non-blocking: job exits after serve.run()) ─────
    _app_name = f"medimage-{MODEL_ID}"
    serve.run(ModelInference.bind(), name=_app_name, route_prefix=ROUTE_PREFIX, _blocking=False)
    print(f"[serve] {_app_name} submitted to Ray Serve at {ROUTE_PREFIX}")
    # Exit immediately — Ray Serve keeps the deployment alive independently of this job
""")

# ── Ray Actor deployment script (avoids Ray Serve + Job Agent issues) ────────
# Uses @ray.remote named detached actor instead of Ray Serve.
# No pandas/numpy ABI issue since we don't import ray.serve.
_RAY_ACTOR_PY = textwrap.dedent("""
    import os, io, zipfile, json, time
    from pathlib import Path

    TRAINING_TYPE  = os.environ["TRAINING_TYPE"]
    ENGINE         = os.environ.get("ENGINE", "Ultralytics")
    MODEL_NAME     = os.environ.get("MODEL_NAME", "")
    MODEL_ID       = os.environ.get("MODEL_ID", "model")
    MINIO_URL      = os.environ.get("MINIO_URL", "http://localhost:9000")
    MINIO_ACCESS   = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET   = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    WEIGHTS_BUCKET = os.environ.get("WEIGHTS_BUCKET", "medimage-weights")
    WEIGHTS_KEY    = os.environ.get("WEIGHTS_KEY", "")
    NUM_CLASSES    = int(os.environ.get("NUM_CLASSES", "2"))
    CLASS_NAMES    = json.loads(os.environ.get("CLASS_NAMES", "[]")) or [f"class_{i}" for i in range(NUM_CLASSES)]

    def _download_weights():
        '''Download weights via presigned HTTPS URL (more compatible than boto3 on
        heterogeneous Ray worker nodes).'''
        import requests as _req
        from botocore.exceptions import ClientError
        if not WEIGHTS_KEY:
            return None
        # Use unique path per actor instance to avoid conflicts with stale state
        import uuid as _uuid
        _unique = _uuid.uuid4().hex[:12]
        local_dir = Path(f"/tmp/mweights/{MODEL_ID}_{_unique}")
        local_dir.mkdir(parents=True, exist_ok=True)
        raw = local_dir / "raw_weights"
        # Get presigned URL from the API server (which is on the docker network
        # and has boto3 working correctly)
        try:
            _api_resp = _req.get(
                f"http://medimage-api:8000/api/internal/weights-url",
                params={"bucket": WEIGHTS_BUCKET, "key": WEIGHTS_KEY},
                timeout=15,
            )
            _api_resp.raise_for_status()
            _presigned = _api_resp.json()["url"]
        except Exception as _e:
            print(f"[actor] Failed to get presigned URL: {_e}", flush=True)
            return None
        try:
            with _req.get(_presigned, stream=True, timeout=300) as _resp:
                _resp.raise_for_status()
                with open(raw, "wb") as _f:
                    for _chunk in _resp.iter_content(1024 * 1024):
                        _f.write(_chunk)
        except Exception as _e:
            print(f"[actor] download failed: {_e}", flush=True)
            return None
        if not raw.is_file():
            print(f"[actor] weights at {WEIGHTS_KEY} is not a file — loading pretrained", flush=True)
            return None
        with open(raw, "rb") as f:
            magic = f.read(4)
        if magic[:2] == b"PK":
            out = local_dir / "extracted"
            out.mkdir(exist_ok=True)
            import zipfile
            with zipfile.ZipFile(raw) as z:
                z.extractall(out)
            return out
        else:
            pt = local_dir / "model.pt"
            raw.rename(pt)
            return pt

    import ray
    _ray_address = os.environ.get("RAY_ADDRESS", "")
    if not _ray_address:
        _ray_address = "auto"
    ray.init(address=_ray_address, ignore_reinit_error=True, log_to_driver=False)

    # Base packages needed by all inference paths
    _actor_pip = ["transformers>=4.40,<5.0", "accelerate>=0.28", "Pillow"]
    if ENGINE in ("PyTorch", "HuggingFace", "Segmentation Models PyTorch", "MONAI"):
        _actor_pip.append("torch")
    if TRAINING_TYPE in ("llm-text", "vlm-finetune"):
        _actor_pip += ["torch", "peft>=0.10"]

    _needs_gpu = TRAINING_TYPE in ("llm-text", "vlm-finetune") or ENGINE in ("HuggingFace", "PyTorch", "MONAI", "Segmentation Models PyTorch")
    _actor_opts = {"num_cpus": 1}
    if _needs_gpu:
        _actor_opts["num_gpus"] = 1

    @ray.remote(**_actor_opts)
    class ModelInferenceActor:
        def __init__(self):
            import subprocess as _sp, sys as _sys
            # Install packages needed by the actor — may run on a node where
            # they're not pre-installed. Keep this minimal and silent.
            _pkgs = ["transformers>=4.40,<5.0", "accelerate>=0.28", "Pillow",
                      "boto3", "numpy<2.0", "opencv-python-headless"]
            if ENGINE in ("HuggingFace", "PyTorch", "MONAI", "Segmentation Models PyTorch"):
                _pkgs.append("torch")
                _pkgs.append("torchvision")
            if ENGINE == "Segmentation Models PyTorch":
                _pkgs.append("segmentation-models-pytorch")
            if ENGINE == "Ultralytics":
                _pkgs.append("ultralytics")
            if TRAINING_TYPE in ("llm-text", "vlm-finetune"):
                _pkgs += ["peft>=0.10"]
            _sp.run([_sys.executable, "-m", "pip", "install", "-q"] + _pkgs,
                    capture_output=True, timeout=600)
            self.model = None
            self.tokenizer = None
            self._hf_processor = None
            self._hf_id2label = None
            self._smp_engine = False
            self.labels = CLASS_NAMES
            self._load_error = None
            try:
                weights = _download_weights()
                self._load(weights)
                print(f"[actor] Model {MODEL_ID} ({TRAINING_TYPE}/{ENGINE}) loaded ok")
            except Exception as _e:
                self._load_error = str(_e)
                print(f"[actor] Model load failed: {_e}")

        def health(self):
            return {"ok": True, "model_loaded": self.model is not None,
                    "load_error": self._load_error,
                    "model_id": MODEL_ID, "training_type": TRAINING_TYPE}

        def _load(self, wp):
            tt, eng = TRAINING_TYPE, ENGINE
            if eng == "HuggingFace" and tt in ("detection", "classification", "segmentation"):
                import torch
                from transformers import AutoImageProcessor
                id2label = {i: n for i, n in enumerate(CLASS_NAMES)} if CLASS_NAMES else {}
                num_cls = NUM_CLASSES
                self._hf_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
                if wp is not None:
                    pt_path = wp if (hasattr(wp, "is_file") and wp.is_file()) else (wp / "model.pt" if hasattr(wp, "__truediv__") else wp)
                    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    cfg_dict = ckpt.get("config", {})
                    id2label = ckpt.get("id2label", {int(k): v for k, v in cfg_dict.get("id2label", {}).items()}) or id2label
                    num_cls = len(id2label) or NUM_CLASSES
                self._hf_id2label = id2label
                label2id = {v: k for k, v in id2label.items()}
                if tt == "detection":
                    from transformers import AutoModelForObjectDetection
                    m = AutoModelForObjectDetection.from_pretrained(MODEL_NAME, num_labels=num_cls, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
                elif tt == "classification":
                    from transformers import AutoModelForImageClassification
                    m = AutoModelForImageClassification.from_pretrained(MODEL_NAME, num_labels=num_cls, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
                else:
                    from transformers import AutoModelForSemanticSegmentation
                    m = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME, num_labels=num_cls, ignore_mismatched_sizes=True)
                if wp is not None:
                    m.load_state_dict(ckpt["model_state_dict"], strict=False)
                self.model = m.eval()
                return
            if tt in ("detection", "segmentation") and eng == "Ultralytics":
                from ultralytics import YOLO
                self.model = YOLO(str(wp) if wp is not None else (MODEL_NAME or "yolov8n.pt"))
            elif tt in ("detection", "segmentation") and eng == "TorchVision":
                # Use SMP (segmentation-models-pytorch) — training code wrote SMP weights
                # (encoder.conv1.weight keys match SMP UNet with ResNet backbone).
                # NOTE: TorchVision's native Mask R-CNN is a different model family.
                import torch
                if "mask" in MODEL_NAME.lower() or "maskrcnn" in MODEL_NAME.lower():
                    # Allow explicit Mask R-CNN via MODEL_NAME
                    import torchvision
                    from torchvision.models.detection import maskrcnn_resnet50_fpn
                    m = maskrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES)
                else:
                    # Default: SMP UNet (matches what training produced)
                    import segmentation_models_pytorch as smp
                    arch = "Unet"  # SMP default
                    enc = "resnet34"  # matches training pipeline default
                    if "resnet50" in MODEL_NAME.lower(): enc = "resnet50"
                    elif "efficientnet" in MODEL_NAME.lower(): enc = "efficientnet-b0"
                    m = smp.Unet(encoder_name=enc, encoder_weights=None,
                                 in_channels=3, classes=NUM_CLASSES)
                if wp is not None:
                    ckpt = torch.load(str(wp), map_location="cpu", weights_only=False)
                    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    m.load_state_dict(state, strict=False)
                self.model = m.eval()
            elif tt == "classification" and eng != "MONAI":
                import torch, timm
                arch = MODEL_NAME.replace("-", "_")
                if wp is None:
                    self.model = timm.create_model(arch, pretrained=True, num_classes=NUM_CLASSES).eval()
                else:
                    ckpt = torch.load(str(wp), map_location="cpu", weights_only=False)
                    if isinstance(ckpt, torch.nn.Module):
                        self.model = ckpt.eval()
                        return
                    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    m = timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES)
                    m.load_state_dict(state, strict=False)
                    self.model = m.eval()
            elif tt == "segmentation" and eng == "Segmentation Models PyTorch":
                import torch, segmentation_models_pytorch as smp
                if wp is None:
                    m = smp.Unet(encoder_name="resnet34", in_channels=3, classes=NUM_CLASSES)
                    self.model = m.eval()
                    self._smp_engine = True
                else:
                    ckpt_path = wp if wp.is_file() else (wp / "model.pt")
                    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                    arch = ckpt.get("arch", "unet")
                    enc  = ckpt.get("encoder", "resnet34")
                    nc   = ckpt.get("num_classes", NUM_CLASSES)
                    m = getattr(smp, arch.capitalize())(encoder_name=enc, in_channels=3, classes=nc)
                    m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                    self.model = m.eval()
                    self._smp_engine = True
            elif eng == "MONAI" and tt == "classification":
                import torch
                from monai.networks.nets import DenseNet121
                # Load checkpoint to detect in_channels and num_classes
                in_ch, n_cls = 1, NUM_CLASSES
                if wp is not None:
                    pt_path = wp if (hasattr(wp, "is_file") and wp.is_file()) else (wp / "model.pt" if hasattr(wp, "__truediv__") else wp)
                    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    # Detect in_channels from first conv weight shape
                    for k, v in state.items():
                        if "conv" in k and len(v.shape) == 4:
                            in_ch = v.shape[1]
                            break
                    # Detect num_classes from last linear layer
                    for k, v in state.items():
                        if "out" in k and len(v.shape) == 2:
                            n_cls = v.shape[0]
                    m = DenseNet121(spatial_dims=2, in_channels=in_ch, out_channels=n_cls)
                    m.load_state_dict(state, strict=False)
                    self.model = m.eval()
                else:
                    self.model = DenseNet121(spatial_dims=2, in_channels=in_ch, out_channels=NUM_CLASSES).eval()
            elif eng == "MONAI" and tt == "segmentation":
                import torch
                from monai.networks.nets import UNet
                in_ch = 1
                if wp is not None:
                    pt_path = wp if (hasattr(wp, "is_file") and wp.is_file()) else (wp / "model.pt" if hasattr(wp, "__truediv__") else wp)
                    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    for k, v in state.items():
                        if "conv" in k and len(v.shape) == 4:
                            in_ch = v.shape[1]
                            break
                    m = UNet(spatial_dims=2, in_channels=in_ch, out_channels=NUM_CLASSES,
                             channels=(16,32,64,128,256), strides=(2,2,2,2))
                    m.load_state_dict(state, strict=False)
                    self.model = m.eval()
                else:
                    self.model = UNet(spatial_dims=2, in_channels=in_ch, out_channels=NUM_CLASSES,
                                     channels=(16,32,64,128,256), strides=(2,2,2,2)).eval()
            elif tt in ("llm-text", "vlm-finetune"):
                import torch
                if tt == "vlm-finetune":
                    from transformers import AutoProcessor, AutoModelForVision2Seq
                    from peft import PeftModel
                    self._hf_processor = AutoProcessor.from_pretrained(MODEL_NAME)
                    self.model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
                else:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    from peft import PeftModel
                    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                    self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
                if wp is not None:
                    self.model = PeftModel.from_pretrained(self.model, str(wp))
            else:
                import torch
                if wp is not None:
                    pt = wp if (wp.is_file() if hasattr(wp, "is_file") else True) else next(wp.glob("*.pt"), None)
                    if pt:
                        self.model = torch.load(str(pt), map_location="cpu", weights_only=False)

        def infer(self, image_bytes=None, conf=0.5, prompt="", system_prompt=""):
            import base64 as _b64
            if isinstance(image_bytes, str) and len(image_bytes) > 20:
                try:
                    image_bytes = _b64.b64decode(image_bytes)
                except Exception:
                    pass
            if self.model is None:
                return {"error": self._load_error or "Model not loaded"}
            tt = TRAINING_TYPE
            try:
                # ── 1. LLM / VLM text generation (must be before HF image check) ──
                if tt in ("llm-text", "vlm-finetune"):
                    import torch
                    if not prompt:
                        return {"error": "Prompt required for LLM inference"}
                    full_prompt = (system_prompt + "\\n" + prompt).strip() if system_prompt else prompt
                    if tt == "vlm-finetune" and self._hf_processor is not None:
                        from PIL import Image as _PILImage
                        import io as _io
                        img_obj = None
                        if image_bytes:
                            img_obj = _PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
                            messages = [{"role": "user", "content": [
                                {"type": "image"},
                                {"type": "text", "text": full_prompt},
                            ]}]
                        else:
                            messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
                        chat_text = self._hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = self._hf_processor(text=[chat_text], images=[img_obj] if img_obj else None, return_tensors="pt").to(self.model.device)
                    else:
                        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                    t0 = time.time()
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    ms = int((time.time() - t0) * 1000)
                    n_tok = out.shape[1] - inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else 0
                    tps = round(n_tok / (ms / 1000), 2) if ms > 0 else 0
                    processor = self._hf_processor if (tt == "vlm-finetune" and self._hf_processor) else self.tokenizer
                    resp = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True) if hasattr(inputs, "input_ids") else processor.decode(out[0], skip_special_tokens=True)
                    return {"type": tt, "response": resp, "tokens_generated": n_tok, "tokens_per_second": tps, "inference_time_ms": ms, "model_name": MODEL_NAME}

                # ── 2. HuggingFace image models (detection / classification / segmentation) ──
                if self._hf_processor is not None:
                    if not image_bytes:
                        return {"error": "Image required"}
                    from PIL import Image as _Img
                    img = _Img.open(io.BytesIO(image_bytes)).convert("RGB")
                    results = self.model.predict(img, conf=conf, verbose=False)
                    if tt == "detection":
                        dets = []
                        for r in results:
                            for box in r.boxes:
                                ci = int(box.cls[0])
                                name = self.labels[ci] if ci < len(self.labels) else f"class_{ci}"
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                dets.append({"class_name": name, "confidence": float(box.conf[0]), "bbox": [x1, y1, x2 - x1, y2 - y1]})
                        return {"type": "detection", "detections": dets}
                    else:
                        masks = []
                        for r in results:
                            if r.masks is None: continue
                            for i, _ in enumerate(r.masks):
                                ci = int(r.boxes.cls[i])
                                name = self.labels[ci] if ci < len(self.labels) else f"class_{ci}"
                                masks.append({"class_name": name, "confidence": float(r.boxes.conf[i])})
                        return {"type": "segmentation", "masks": masks, "num_masks": len(masks)}

                if tt == "segmentation" and self._smp_engine:
                    import torch
                    import torchvision.transforms as T
                    from PIL import Image as _Img
                    if not image_bytes:
                        return {"error": "Image required"}
                    img = _Img.open(io.BytesIO(image_bytes)).convert("RGB")
                    orig_w, orig_h = img.size
                    tfm = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    with torch.no_grad():
                        logits = self.model(tfm(img).unsqueeze(0))
                    mask = logits.argmax(dim=1)[0]
                    total_px = mask.numel()
                    segs = [{"class_name": self.labels[ci] if ci < len(self.labels) else f"class_{ci}", "class_id": ci, "pixel_count": int((mask == ci).sum()), "coverage": round(int((mask == ci).sum()) / total_px, 4)} for ci in mask.unique().tolist()]
                    segs.sort(key=lambda x: -x["pixel_count"])
                    return {"type": "segmentation", "segments": segs, "num_classes": len(segs), "image_size": [orig_w, orig_h]}

                # ── TorchVision detection / segmentation (Mask R-CNN, Faster R-CNN) ──
                if tt in ("detection", "segmentation") and ENGINE == "TorchVision":
                    import torch
                    from PIL import Image as _Img
                    import numpy as _np
                    if not image_bytes:
                        return {"error": "Image required"}
                    img = _Img.open(io.BytesIO(image_bytes)).convert("RGB")
                    orig_w, orig_h = img.size
                    t0 = time.time()
                    # SMP UNet: input is a normalized [0,1] tensor, shape (1, 3, H, W)
                    img_t = T.functional.to_tensor(img).unsqueeze(0)
                    # ImageNet normalization (SMP default for ResNet encoders)
                    img_t = T.functional.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    with torch.no_grad():
                        logits = self.model(img_t)
                    ms = int((time.time() - t0) * 1000)
                    # logits shape: (1, num_classes, H, W)
                    probs = logits.softmax(dim=1)[0]
                    pred = probs.argmax(dim=0).cpu().numpy()  # (H, W)
                    total_px = int(pred.size)
                    # Collect unique classes
                    unique_classes = _np.unique(pred).tolist()
                    masks_out = []
                    # Background class is 0; skip it
                    for ci in unique_classes:
                        if ci == 0:
                            continue
                        name = self.labels[ci] if ci < len(self.labels) else f"class_{ci}"
                        binmask = (pred == ci).astype(_np.uint8)
                        pixel_count = int(binmask.sum())
                        coverage = round(pixel_count / total_px, 4)
                        # Extract polygon outline (simplified) for frontend draw
                        polygon = []
                        try:
                            import cv2 as _cv2  # type: ignore
                            contours, _ = _cv2.findContours(
                                binmask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
                            )
                            if contours:
                                c = max(contours, key=_cv2.contourArea)
                                if len(c) > 60:
                                    step = max(1, len(c) // 60)
                                    c = c[::step]
                                polygon = [
                                    [float(p[0][0]) / orig_w, float(p[0][1]) / orig_h]
                                    for p in c
                                ]
                        except ImportError:
                            pass
                        if not polygon:
                            # Fallback: bounding box of the mask
                            ys, xs = _np.where(binmask > 0)
                            if len(xs) > 0:
                                x1, x2 = int(xs.min()), int(xs.max())
                                y1, y2 = int(ys.min()), int(ys.max())
                                polygon = [
                                    [x1 / orig_w, y1 / orig_h],
                                    [x2 / orig_w, y1 / orig_h],
                                    [x2 / orig_w, y2 / orig_h],
                                    [x1 / orig_w, y2 / orig_h],
                                ]
                        masks_out.append({
                            "label": name,
                            "class_id": ci,
                            "confidence": 1.0,  # SMP doesn't have per-region confidence
                            "color": "#8b5cf6",
                            "polygon": polygon,
                            "mask_pixels": pixel_count,
                            "coverage": coverage,
                        })
                    return {"type": "segmentation", "masks": masks_out,
                            "count": len(masks_out), "inference_time_ms": ms,
                            "image_size": [orig_w, orig_h]}

                if tt == "classification":
                    import torch
                    import torchvision.transforms as T
                    from PIL import Image as _Img
                    if not image_bytes:
                        return {"error": "Image required"}
                    img = _Img.open(io.BytesIO(image_bytes))
                    eng = ENGINE
                    if eng == "MONAI":
                        # MONAI models expect grayscale or RGB — use same channel count as trained
                        from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor
                        import numpy as np
                        # Detect in_channels from model first conv
                        in_ch = 1
                        for p in self.model.parameters():
                            if len(p.shape) == 4:
                                in_ch = p.shape[1]
                                break
                        img_arr = np.array(img.convert("L") if in_ch == 1 else img.convert("RGB"), dtype=np.float32)
                        if in_ch == 1:
                            img_t = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                        else:
                            img_t = torch.tensor(img_arr.transpose(2,0,1)).unsqueeze(0)  # (1,3,H,W)
                        # Resize to 224x224
                        img_t = torch.nn.functional.interpolate(img_t, size=(224,224), mode="bilinear", align_corners=False)
                        # Normalize to [0,1]
                        img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8)
                        with torch.no_grad():
                            logits = self.model(img_t)
                            probs = torch.softmax(logits, dim=1)[0].tolist()
                    else:
                        img = img.convert("RGB")
                        tfm = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        with torch.no_grad():
                            logits = self.model(tfm(img).unsqueeze(0))
                            probs = torch.softmax(logits, dim=1)[0].tolist()
                    preds = sorted([{"label": self.labels[i] if i < len(self.labels) else f"class_{i}", "confidence": p} for i, p in enumerate(probs)], key=lambda x: -x["confidence"])
                    return {"type": "classification", "predictions": preds, "top_label": preds[0]["label"]}

                return {"error": f"No inference handler for engine={ENGINE}, type={tt}"}
            except Exception as exc:
                return {"error": str(exc)}

    # ── Deploy as named detached actor ───────────────────────────────────────
    _actor_name = f"model-{MODEL_ID}"
    # Kill existing actor with same name (redeploy)
    try:
        _existing = ray.get_actor(_actor_name, namespace="default")
        ray.kill(_existing)
        time.sleep(1)
        print(f"[actor] Killed existing actor {_actor_name}")
    except Exception:
        pass

    _actor = ModelInferenceActor.options(
        name=_actor_name,
        namespace="default",
        lifetime="detached",
        num_cpus=1,
    ).remote()

    print(f"[actor] Loading model on Ray cluster (up to 5 min)...")
    _deadline = time.time() + 300
    while time.time() < _deadline:
        try:
            _h = ray.get(_actor.health.remote(), timeout=30)
            if _h.get("ok"):
                print(f"ACTOR_READY:{_actor_name}")
                break
            else:
                print(f"[actor] Not ready: {_h}")
                time.sleep(5)
        except Exception as _e:
            print(f"[actor] Waiting: {_e}")
            time.sleep(5)
    else:
        print("[actor] TIMEOUT: actor did not become ready after 5 min")
        import sys; sys.exit(1)
    # NOTE: do NOT call ray.shutdown() — actor must persist on cluster
""")

# Per-job deploy state
_model_deploy_states: dict[str, dict] = {}


def _get_model_deploy_state(job_id: str) -> dict:
    if job_id not in _model_deploy_states:
        _model_deploy_states[job_id] = {"status": "idle", "url": None, "logs": []}
    return _model_deploy_states[job_id]


def _poll_model_job(ray_dashboard_url: str, submission_id: str, serve_url: str, job_id: str) -> None:
    """Poll Ray Job until SUCCEEDED/FAILED, then poll serve app status via HTTP."""
    state = _get_model_deploy_state(job_id)
    app_name = f"medimage-{job_id}"
    deadline = time.time() + 600  # 10 min total

    # Phase A: wait for job to finish (SUCCEEDED or FAILED)
    job_done = False
    while time.time() < deadline:
        time.sleep(5)
        try:
            r = httpx.get(f"{ray_dashboard_url.rstrip('/')}/api/jobs/{submission_id}", timeout=10)
            if r.is_success:
                d = r.json()
                status = d.get("status", "")
                msg = (d.get("message") or d.get("error_type") or "")[:300]
                state["logs"] = [f"Ray Job [{status}]{': ' + msg if msg else ''}"] 
                if status == "SUCCEEDED":
                    job_done = True
                    break
                elif status == "FAILED":
                    state.update(status="error", logs=[f"Deploy job FAILED: {msg}"])
                    return
                # RUNNING/PENDING — keep polling
        except Exception as exc:
            state["logs"] = [str(exc)]

    if not job_done:
        # Job might still be RUNNING (serve.run non-blocking exits fast)
        # Check serve app status directly
        pass

    # Phase B: poll serve application status via HTTP (no Ray Client)
    state["logs"] = [f"Job finished, waiting for serve app '{app_name}' to be RUNNING…"]
    for _ in range(60):  # up to 5 min
        time.sleep(5)
        try:
            rs = httpx.get(f"{ray_dashboard_url.rstrip('/')}/api/serve/applications/", timeout=10)
            if rs.is_success:
                apps = rs.json().get("applications", {})
                app_info = apps.get(app_name, {})
                app_status = app_info.get("status", "")
                state["logs"] = [f"Serve app status: {app_status}"]
                if app_status == "RUNNING":
                    state.update(status="running", url=serve_url, logs=[f"Model deployed and RUNNING at {serve_url}"])
                    with get_db() as conn:
                        conn.execute("UPDATE jobs SET ray_serve_url=?, inference_provider='ray' WHERE id=?",
                                     (serve_url, job_id))
                        conn.commit()
                    return
                elif app_status == "DEPLOY_FAILED":
                    state.update(status="error", logs=[f"Serve app DEPLOY_FAILED: {app_info.get('message', '')[:300]}"])
                    return
        except Exception as exc:
            state["logs"] = [str(exc)]
        # Also probe /health directly
        try:
            h = httpx.get(f"{serve_url.rstrip('/')}/health", timeout=5)
            if h.is_success:
                state.update(status="running", url=serve_url, logs=["Model inference server is running!"])
                with get_db() as conn:
                    conn.execute("UPDATE jobs SET ray_serve_url=?, inference_provider='ray' WHERE id=?",
                                 (serve_url, job_id))
                    conn.commit()
                return
        except Exception:
            pass

    state.update(status="error", logs=[f"Timeout: serve app '{app_name}' did not reach RUNNING after 10 min"])


def _submit_model_deploy(ray_dashboard_url: str, payload: dict, serve_url: str, job_id: str, max_wait: int = 600) -> None:
    """Deploy model as a named Ray Actor via Ray Jobs API.
    Named detached actors persist on the cluster after the job exits.
    """
    import subprocess, sys, tempfile, os as _os, requests as _req, json as _json
    state = _get_model_deploy_state(job_id)
    env_vars = payload.get("runtime_env", {}).get("env_vars", {})

    from urllib.parse import urlparse as _urlparse
    _p = _urlparse(ray_dashboard_url)
    _dashboard = f"{_p.scheme}://{_p.netloc}"

    # Phase 1: wait for at least one worker
    state["logs"] = ["Checking Ray cluster for available workers…"]
    deadline = time.time() + max_wait
    start = time.time()
    while time.time() < deadline:
        elapsed = int(time.time() - start)
        workers = _count_ray_workers(ray_dashboard_url)
        if workers > 0:
            state["logs"] = [f"✓ {workers} worker node(s) online. Deploying model actor…"]
            break
        state["logs"] = [f"No Ray workers running ({elapsed}s elapsed). Waiting…"]
        time.sleep(20)
    else:
        state.update(status="error", logs=["Timed out — no Ray workers came online."])
        return

    # Phase 2: write actor script and submit via Ray Jobs API
    state["logs"] = ["Submitting model deploy via Ray Jobs API…"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(_RAY_ACTOR_PY)
        script_path = f.name
    # Read the script and embed as base64 in the entrypoint to avoid working_dir
    import base64 as _b64lib
    with open(script_path, "rb") as _f:
        _script_b64 = _b64lib.b64encode(_f.read()).decode()

    remaining = max(600, int(deadline - time.time()))
    try:
        # Send the script via env var to avoid "Argument list too long" (argv limit).
        _deploy_env = dict(env_vars)
        _deploy_env["MEDIMAGE_DEPLOY_SCRIPT_B64"] = _script_b64
        _deploy_payload = {
            "entrypoint": "bash -c 'echo $MEDIMAGE_DEPLOY_SCRIPT_B64 | base64 -d > /tmp/deploy_script.py && python3 /tmp/deploy_script.py'",
            "runtime_env": {
                "env_vars": _deploy_env,
                "pip": ["transformers>=4.40,<5.0", "accelerate>=0.28", "Pillow",
                        "boto3", "numpy<2.0", "torch", "peft>=0.10", "bitsandbytes>=0.43"],
            },
        }
        _resp = _req.post(f"{_dashboard}/api/jobs/", json=_deploy_payload, timeout=30)
        _resp.raise_for_status()
        _submit_id = _resp.json().get("submission_id") or _resp.json().get("job_id")
        state["logs"] = [f"Deploy job submitted: {_submit_id}. Waiting for completion…"]
    except Exception as _e:
        try:
            _os.unlink(script_path)
        except Exception:
            pass
        state.update(status="error", logs=[f"Deploy job submit failed: {_e}"])
        return

    # Phase 3: poll job status until done
    _last_status = ""
    while time.time() < deadline:
        try:
            _st = _req.get(f"{_dashboard}/api/jobs/{_submit_id}", timeout=10).json()
            _ray_status = _st.get("status", "UNKNOWN")
        except Exception:
            _ray_status = _last_status
        if _ray_status != _last_status:
            state["logs"].append(f"Job status: {_ray_status}")
            _last_status = _ray_status
        if _ray_status in ("SUCCEEDED", "FAILED", "STOPPED"):
            break
        time.sleep(5)
    else:
        try:
            _os.unlink(script_path)
        except Exception:
            pass
        state.update(status="error", logs=[f"Deploy timed out after {remaining}s"])
        return

    # Get logs
    try:
        _logs_resp = _req.get(f"{_dashboard}/api/jobs/{_submit_id}/logs", timeout=30)
        _logs = _logs_resp.json().get("logs", "") if _logs_resp.ok else ""
    except Exception:
        _logs = ""
    try:
        _os.unlink(script_path)
    except Exception:
        pass

    if _ray_status != "SUCCEEDED":
        _tail = _logs[-500:] if _logs else "no logs"
        state.update(status="error", logs=[f"Deploy failed ({_ray_status}): {_tail}"])
        return

    if "ACTOR_READY:" not in _logs:
        _tail = _logs[-400:] if _logs else "no output"
        state.update(status="error", logs=[f"Deploy completed but actor not ready: {_tail}"])
        return

    # Phase 4: actor is ready — update state and DB
    _actor_name = f"model-{job_id}"
    state.update(status="running", url=serve_url, logs=[f"Model actor '{_actor_name}' is running on Ray cluster!"])
    with get_db() as conn:
        conn.execute("UPDATE jobs SET ray_serve_url=?, inference_provider='ray' WHERE id=?",
                     (serve_url, job_id))
        conn.commit()


def _make_serve_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("serve_app.py", _SERVE_APP_PY)
    buf.seek(0)
    return buf.read()


def _poll_ray_serve_app(ray_dashboard_url: str, serve_url: str) -> None:
    _ray_serve_state["logs"] = ["Waiting for Ray Serve app to be ready…"]
    for _ in range(120):  # up to 10 min
        time.sleep(5)
        try:
            r = httpx.get(f"{ray_dashboard_url.rstrip('/')}/api/serve/applications/", timeout=10)
            if r.is_success:
                apps = r.json().get("applications", {})
                app_info = apps.get("medimage-inference", {})
                if not app_info:
                    _ray_serve_state["logs"] = ["App deploying, waiting for registration…"]
                    continue
                status = app_info.get("status", "")
                message = app_info.get("message", "")
                _ray_serve_state["logs"] = [f"[{status}] {message}"] if message else [f"Status: {status}"]
                if status == "RUNNING":
                    _ray_serve_state.update(status="running", url=serve_url, logs=["Inference server is running!"])
                    with get_db() as conn:
                        conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", ("global_ray_serve_url", serve_url))
                        conn.commit()
                    return
                elif status == "UNHEALTHY":
                    _ray_serve_state["status"] = "error"
                    _ray_serve_state["logs"] = [f"Unhealthy: {message}"]
                    return
            else:
                _ray_serve_state["logs"] = [f"HTTP {r.status_code}: {r.text[:100]}"]
        except Exception as exc:
            _ray_serve_state["logs"] = [str(exc)]
    _ray_serve_state["status"] = "error"
    _ray_serve_state["logs"].append("Timeout waiting for serve deployment (10 min)")


def _count_ray_workers(ray_dashboard_url: str) -> int:
    """Return number of non-head connected nodes, or -1 on error."""
    try:
        r = httpx.get(f"{ray_dashboard_url}/nodes?view=summary", timeout=10)
        if r.is_success:
            data = r.json()
            # Response: {"data": {"summary": [{...}, ...]}} — each entry is a node
            nodes = data.get("data", {}).get("summary", [])
            if isinstance(nodes, list):
                # Count nodes that are ALIVE and not the head (raylet_state == ALIVE, isHeadNode == False or similar)
                workers = [n for n in nodes if n.get("raylet", {}).get("state", "") == "ALIVE"
                           and not n.get("raylet", {}).get("isHeadNode", False)]
                # Fallback: if isHeadNode not present, subtract 1 for head
                if not any("isHeadNode" in n.get("raylet", {}) for n in nodes):
                    return max(0, len([n for n in nodes if n.get("raylet", {}).get("state") == "ALIVE"]) - 1)
                return len(workers)
        # Fallback: try /api/cluster
        r2 = httpx.get(f"{ray_dashboard_url}/api/cluster", timeout=10)
        if r2.is_success:
            data2 = r2.json()
            total = (data2.get("data", {}).get("summary", {}).get("numNodes")
                     or data2.get("numNodes", 0))
            return max(0, int(total) - 1)
    except Exception:
        pass
    return -1  # unknown


def _submit_ray_job_with_retry(ray_dashboard_url: str, payload: dict, serve_url: str, max_wait: int = 600) -> None:
    """Wait for a worker to appear, then submit the Ray Job."""
    deadline = time.time() + max_wait
    start = time.time()

    # Phase 1: wait for at least one worker
    _ray_serve_state["logs"] = ["Checking Ray cluster for available workers…"]
    while time.time() < deadline:
        elapsed = int(time.time() - start)
        workers = _count_ray_workers(ray_dashboard_url)
        if workers > 0:
            _ray_serve_state["logs"] = [f"✓ {workers} worker node(s) online. Submitting job…"]
            break
        if workers == 0:
            _ray_serve_state["logs"] = [
                f"No Ray workers running ({elapsed}s elapsed). "
                "Waiting for Modal to scale up a worker node (can take 1–3 min)…",
                "Tip: set min_workers=1 in your Modal Ray cluster to avoid cold-start delays.",
            ]
        else:
            _ray_serve_state["logs"] = [f"Cluster reachable, waiting for workers ({elapsed}s)…"]
        time.sleep(20)
    else:
        _ray_serve_state.update(status="error", logs=[
            f"Timed out after {max_wait}s — no Ray workers came online.",
            "Fix: set min_workers ≥ 1 in your Modal Ray cluster config so at least one worker is always running.",
        ])
        return

    # Phase 2: submit job — retry until deadline
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        elapsed = int(time.time() - start)
        attempt_payload = {**payload, "submission_id": f"medimage-inference-{int(time.time())}"}
        try:
            r = httpx.post(f"{ray_dashboard_url}/api/jobs/", json=attempt_payload, timeout=30)
            if r.is_success:
                job_id = r.json().get("submission_id") or r.json().get("job_id", attempt_payload["submission_id"])
                _ray_serve_state["logs"] = [f"Job submitted: {job_id}"]
                _poll_ray_job(ray_dashboard_url, job_id, serve_url)
                return
            text = r.text
            if "get_target_agent" in text or r.status_code in (500, 503):
                _ray_serve_state["logs"] = [f"Attempt {attempt} ({elapsed}s): agent not ready yet, retrying in 15s…"]
                time.sleep(15)
                continue
            _ray_serve_state.update(status="error", logs=[f"Ray Jobs API error: {text[:400]}"])
            return
        except Exception as exc:
            _ray_serve_state["logs"] = [f"Connection error (attempt {attempt}): {exc}"]
            time.sleep(20)
        if time.time() >= deadline:
            break

    _ray_serve_state.update(status="error", logs=[
        "Worker found but job agent not ready after multiple attempts.",
        "Please check your Ray cluster's job agent configuration.",
    ])


def _poll_ray_job(ray_dashboard_url: str, job_id: str, serve_url: str) -> None:
    for _ in range(120):
        time.sleep(5)
        try:
            r = httpx.get(f"{ray_dashboard_url.rstrip('/')}/api/jobs/{job_id}", timeout=10)
            if r.is_success:
                d = r.json()
                status = d.get("status", "")
                msg = d.get("message", "") or d.get("error_type", "")
                _ray_serve_state["logs"] = [f"[{status}] {msg}"] if msg else [f"Job status: {status}"]
                if status in ("SUCCEEDED", "RUNNING"):
                    # Poll serve health
                    for _ in range(24):
                        time.sleep(5)
                        try:
                            h = httpx.get(f"{serve_url.rstrip('/')}/health", timeout=5)
                            if h.is_success:
                                _ray_serve_state.update(status="running", url=serve_url, logs=["Inference server is running!"])
                                with get_db() as conn:
                                    conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", ("global_ray_serve_url", serve_url))
                                    conn.commit()
                                return
                        except Exception:
                            pass
                    _ray_serve_state.update(status="error", logs=["Job finished but /health unreachable"])
                    return
                elif status == "FAILED":
                    _ray_serve_state.update(status="error", logs=[f"Job FAILED: {msg}"])
                    return
        except Exception as exc:
            _ray_serve_state["logs"] = [str(exc)]
    _ray_serve_state.update(status="error", logs=["Timeout waiting for job (10 min)"])


@app.post("/api/ray/clear-dead")
async def ray_clear_dead_nodes(request: Request):
    """Drain dead nodes from Ray cluster via the dashboard API."""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    import requests as _req
    try:
        r = _req.get(f"{RAY_URL}/nodes?view=summary", timeout=5)
        nodes = r.json().get("data", {}).get("summary", [])
    except Exception as e:
        raise HTTPException(502, f"Cannot reach Ray dashboard: {e}")

    dead = [n for n in nodes if n.get("raylet", {}).get("state") == "DEAD"]
    if not dead:
        return {"cleared": 0, "message": "No dead nodes found"}

    cleared, errors = 0, []
    seen = set()
    for node in dead:
        node_id = (node.get("raylet") or {}).get("nodeId") or node.get("ip")
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        try:
            dr = _req.post(
                f"{RAY_URL}/api/node_manager/drain_node",
                json={"node_id": node_id, "reason": 1, "reason_message": "manual cleanup"},
                timeout=5,
            )
            if dr.ok:
                cleared += 1
            else:
                errors.append(f"{node_id}: {dr.status_code}")
        except Exception as e:
            errors.append(f"{node_id}: {e}")

    return {"cleared": cleared, "total_dead": len(seen), "errors": errors}


@app.post("/api/ray/serve/stop-all")
async def ray_serve_stop_all(request: Request):
    """Stop every running Ray Serve deploy from the medimage API.

    Called by the test-all bulk run as a pre-flight (and by the Stop-All
    button on the Models page) to free the GPUs that persistent deploy
    actors are holding. On a 4-GPU cluster, 4 deploys eat all the GPUs
    and every subsequent training job sits PENDING until the reconciler
    kills it — running stop-all first keeps the cluster usable for the
    whole test-all sweep.

    Two sources of actors to kill:
      1. DB-tracked: jobs with ray_serve_url set + inference_provider='ray'
      2. Orphan: any alive `model-*` actor on the cluster that the DB
         doesn't know about (e.g. from a previous test-all run where the
         API was restarted before the DB row was written)
    """
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    import requests as _req
    import base64 as _b64lib
    import concurrent.futures as _cf

    # 1) Collect DB-tracked deploys
    with get_db() as conn:
        db_rows = conn.execute(
            "SELECT id, ray_serve_url FROM jobs "
            "WHERE ray_serve_url != '' AND inference_provider='ray' "
            "ORDER BY finished_at DESC"
        ).fetchall()
    db_ids = {r["id"] for r in db_rows}

    # 2) Find orphan model-* actors (alive but not in DB)
    orphan_actor_names = []
    try:
        _ar = _req.get(f"{RAY_URL}/api/v0/actors", timeout=5)
        if _ar.ok:
            for a in _ar.json().get("data", {}).get("result", {}).get("result", []):
                name = a.get("name", "")
                st = a.get("state", "")
                if name.startswith("model-") and st == "ALIVE":
                    jid = name[len("model-"):]
                    if jid not in db_ids:
                        orphan_actor_names.append(name)
    except Exception:
        pass

    # Build work list: (job_id, actor_name). For DB-tracked, name =
    # "model-{jid}". For orphans, name is what we discovered.
    work = [(r["id"], f"model-{r['id']}") for r in db_rows] + \
           [(None, name) for name in orphan_actor_names]
    if not work:
        return {"stopped": 0, "failed": 0, "orphans": 0, "ids": []}

    # Stronger kill: no_restart=True (Ray supervisor can otherwise
    # respawn the actor on the next health probe) + a follow-up
    # ray.get_actor() check to confirm the death actually stuck.
    # Without this, the actor state stays ALIVE for minutes even
    # though ray.kill() returned — letting stale deploys keep
    # holding the GPU.
    def _kill_one(item) -> tuple[str | None, bool, str]:
        jid, actor_name = item
        kill_script = (
            "import ray, sys, time, os\n"
            "ray.init(address='auto', ignore_reinit_error=True, log_to_driver=False)\n"
            "actor_name = os.environ.get('ACTOR_NAME','')\n"
            "if not actor_name:\n"
            "    print('NO_ACTOR_NAME', flush=True); sys.exit(1)\n"
            "try:\n"
            "    actor = ray.get_actor(actor_name, namespace='default')\n"
            "    print(f'FOUND:{actor_name}', flush=True)\n"
            "    ray.kill(actor, no_restart=True)\n"
            "    print(f'KILLED:{actor_name}', flush=True)\n"
            "    time.sleep(2)\n"
            "    try:\n"
            "        ray.get_actor(actor_name, namespace='default')\n"
            "        print(f'STILL_ALIVE:{actor_name}', flush=True)\n"
            "    except Exception:\n"
            "        print(f'CONFIRMED_DEAD:{actor_name}', flush=True)\n"
            "except Exception as e:\n"
            "    print(f'NOT_FOUND:{e}', flush=True)\n"
        )
        b64 = _b64lib.b64encode(kill_script.encode()).decode()
        try:
            r = _req.post(
                f"{RAY_URL}/api/jobs/",
                json={"entrypoint": f"bash -c 'echo {b64} | base64 -d > /tmp/kill.py && ACTOR_NAME={actor_name} python3 /tmp/kill.py'",
                      "runtime_env": {}},
                timeout=15,
            )
            r.raise_for_status()
            sub_id = r.json().get("submission_id") or r.json().get("job_id")
            for _ in range(15):
                time.sleep(1)
                st = _req.get(f"{RAY_URL}/api/jobs/{sub_id}", timeout=5).json()
                if st.get("status") in ("SUCCEEDED", "FAILED", "STOPPED"):
                    break
            if jid is not None:
                with get_db() as _c:
                    _c.execute("UPDATE jobs SET ray_serve_url='', inference_provider='' WHERE id=?", (jid,))
                    _c.commit()
            return (jid, True, "")
        except Exception as e:
            return (jid, False, str(e))

    stopped, failed, errors = [], [], []
    # Up to 4 kills in parallel (matches the cluster GPU count)
    with _cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_kill_one, item) for item in work]
        for f in _cf.as_completed(futs):
            jid, ok, err = f.result()
            if ok:
                stopped.append(jid)
            else:
                failed.append(jid)
                errors.append({"id": jid, "error": err})

    # Wait until at least one GPU is free — same reason as the per-model
    # stop endpoint above. The bulk loop's pre-flight calls this and then
    # immediately submits the first training job, so we must hold until
    # the cluster's GPU accounting reflects the freed slot.
    try:
        for _wait_i in range(30):
            time.sleep(1)
            _cs = _req.get(f"{RAY_URL}/api/cluster_status", timeout=4).json()
            _ru = _cs.get("data", {}).get("autoscalingStatus", "")
            try:
                _gpu_chunk = _ru.split("ResourceUsage:")[1].split("\n")[0]
                _used_g = float(_gpu_chunk.split("GPU,")[0].rsplit(" ", 1)[-1])
                _total_g = float(_gpu_chunk.split("GPU,")[1].split(" ", 1)[0].rstrip("/"))
                if _used_g < _total_g:
                    break
            except Exception:
                pass
        else:
            pass  # caller will see PENDING; reconciler will clean up
    except Exception:
        pass

    return {"stopped": len(stopped), "failed": len(failed), "orphans": len(orphan_actor_names), "ids": stopped, "errors": errors}


@app.post("/api/ray/jobs/{submission_id}/stop")
def ray_job_stop(submission_id: str, request: Request):
    """Stop a single Ray job by submission_id. Used by the reconciler and the
    test-all UI to free GPUs that are stuck on a PENDING job."""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    ok = _stop_ray_submission(submission_id)
    return {"stopped": ok, "submission_id": submission_id}


@app.get("/api/ray/jobs/reconcile")
def ray_reconcile(request: Request):
    """Run one reconciliation pass: stop stuck PENDING Ray jobs + sync DB
    state for finished jobs that the training thread forgot to update.
    Returns a summary so the UI can show what was cleaned up."""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    return _reconcile_ray_jobs()


@app.post("/api/ray-serve/deploy")
async def ray_infer_deploy(body: dict):
    ray_dashboard_url = (body.get("ray_dashboard_url") or "").rstrip("/")
    if not ray_dashboard_url:
        raise HTTPException(status_code=400, detail="ray_dashboard_url required")
    if _ray_serve_state["status"] == "deploying":
        raise HTTPException(status_code=409, detail="Deploy already in progress")

    # Derive serve URL (port 8000)
    try:
        from urllib.parse import urlparse
        p = urlparse(ray_dashboard_url)
        serve_url = f"{p.scheme}://{p.hostname}:8000"
    except Exception:
        serve_url = ray_dashboard_url.replace(":8265", ":8000")

    import base64 as _b64
    # Embed serve_app.py inline so no working_dir fetch needed
    script = _SERVE_APP_PY + textwrap.dedent("""
        import ray
        ray.init(address="auto", log_to_driver=False)
        serve.run(InferenceApp.bind(), name="medimage-inference", route_prefix="/")
        print("medimage-inference deployed successfully")
        import time
        while True:
            time.sleep(60)
    """)
    script_b64 = _b64.b64encode(script.encode()).decode()
    submission_id = f"medimage-inference-{int(time.time())}"
    payload = {
        "submission_id": submission_id,
        "entrypoint": "python -c \"import base64,os; exec(base64.b64decode(os.environ['_SCRIPT']).decode())\"",
        "runtime_env": {
            "pip": ["ray[serve]", "fastapi", "uvicorn", "python-multipart", "pillow"],
            "env_vars": {"_SCRIPT": script_b64, "RAY_ADDRESS": "", "RAY_ENABLE_AUTO_CONNECT": "0"},
        },
    }

    _ray_serve_state.update(status="deploying", url=serve_url, logs=["Checking Ray cluster…"])
    threading.Thread(
        target=_submit_ray_job_with_retry,
        args=(ray_dashboard_url, payload, serve_url),
        kwargs={"max_wait": 600},
        daemon=True,
    ).start()
    return {"status": "deploying", "serve_url": serve_url}


@app.get("/api/ray-serve/script.zip")
def get_serve_script_zip():
    """Serve the Ray Serve app zip for runtime_env working_dir."""
    return Response(
        content=_make_serve_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=script.zip"},
    )


@app.get("/api/ray-serve/status")
def ray_infer_status_ep():
    return {
        "status": _ray_serve_state["status"],
        "url": _ray_serve_state["url"],
        "logs": _ray_serve_state["logs"][-20:],
    }


# ── GPU stats via nvidia-smi on Ray cluster ──────────────────────────────────
_gpu_stats_cache: dict = {"ts": 0.0, "data": None}

# ── GPU utilization cache: receives data from gpu-monitor sidecar ──────────
# Structure: { "hostname1": {"util": 12.0, "gpus": [...], "ts": 1781...}, ... }
_gpu_util_per_node: dict = {}

# ── GPU status cache: consumes the team's /api/gpu-status (which already
# filters to Ray-managed GPUs and attaches ray_node_index + ray_gpu_indices).
# Keyed by ray_node_index so we can attribute data to the right Ray node
# without brittle list-slicing across clusters.
# Structure: {
#   <ray_node_index>: {
#     "hostname": "...", "internal_ip": "...",
#     "ts": <float>, "gpus": [{"index","util_pct","power_watt",...}, ...]
#   }, ...
# }
_gpu_status_cache: dict = {}
GPU_STATUS_URL = os.getenv("GPU_STATUS_URL", f"{RAY_URL.rsplit(':', 1)[0]}:8085/api/gpu-status")

# Per-GPU TDP map (Watts). Used as the denominator for power-based activity
# estimation. Falls back to 700W (H200) when the model is unknown.
GPU_TDP_W = {
    "H200": 700, "H100": 700, "A100": 400, "L40S": 350, "L40": 300,
    "A10": 150, "L4": 72, "T4": 70, "V100": 300, "A40": 300,
    "RTX 4090": 450, "RTX 3090": 350, "RTX 3080": 320, "RTX A6000": 300,
}


def _tdp_for(name: str) -> int:
    if not name:
        return 700
    n = name.upper()
    for k, w in GPU_TDP_W.items():
        if k.upper() in n:
            return w
    return 700


def _estimate_activity_pct(util_pct: int, power_watt: float, tdp_w: int) -> int:
    """nvidia-smi's utilization.gpu is a 1s rolling window that under-reports
    whenever the GPU is mid-batch (data loader, sync, CPU prep). Power draw
    is a more reliable proxy: idle H200 ≈ 30W, full load ≈ 700W. When
    util_pct is 0 but power is significantly above idle (~50W+), report an
    activity estimate derived from power."""
    if util_pct and util_pct > 0:
        return int(util_pct)
    if power_watt and tdp_w > 0:
        est = int(round(power_watt / tdp_w * 100))
        return max(0, min(100, est))
    return 0


@app.put("/api/gpu-util")
def update_gpu_util(body: dict):
    """Endpoint for gpu-monitor sidecar containers to push real GPU utilization.
    Accepts:
      { "gpu_util_pct": 35.2, "hostname": "node1", "gpus": [{ "index": 0, "util_pct": 35, ... }] }
    Or simple:
      { "gpu_util_pct": 35.2 }
    """
    global _gpu_util_per_node
    hostname = body.get("hostname", "")
    util = float(body.get("gpu_util_pct", 0))
    gpus = body.get("gpus", [])
    ts = time.time()

    if hostname:
        _gpu_util_per_node[hostname] = {"util": util, "gpus": gpus, "ts": ts}
    else:
        # Legacy: cluster-wide average, store under empty key
        _gpu_util_per_node[""] = {"util": util, "gpus": gpus, "ts": ts}

    # Also write to file for persistence across restarts
    try:
        import json as _json
        with open("/tmp/gpu_util_cache.json", "w") as f:
            _json.dump({"per_node": _gpu_util_per_node, "ts": ts}, f)
    except Exception:
        pass

    return {"ok": True, "hostname": hostname, "gpu_util_pct": util}


def _get_cluster_gpu_util():
    """Get cluster-wide average GPU utilization from sidecar cache."""
    global _gpu_util_per_node

    # Load from file if cache is empty (after restart)
    if not _gpu_util_per_node:
        try:
            import json as _json
            with open("/tmp/gpu_util_cache.json") as f:
                cached = _json.load(f)
                _gpu_util_per_node = cached.get("per_node", {})
        except Exception:
            pass

    # Prune stale entries (> 60s old)
    now = time.time()
    _gpu_util_per_node = {k: v for k, v in _gpu_util_per_node.items() if now - v.get("ts", 0) < 60}

    if not _gpu_util_per_node:
        return 0.0, {}

    # Calculate cluster average from all per-node entries
    utils = [v["util"] for k, v in _gpu_util_per_node.items() if k and v.get("util", 0) > 0]
    if utils:
        return round(sum(utils) / len(utils), 1), _gpu_util_per_node

    # Fallback to legacy cluster-wide value
    legacy = _gpu_util_per_node.get("", {})
    return legacy.get("util", 0), _gpu_util_per_node

def _poll_gpu_status():
    """Fetch the team's /api/gpu-status (which filters to Ray-managed GPUs
    and tags each with ray_node_index) and store it in _gpu_status_cache
    keyed by ray_node_index. Runs as a daemon thread so the UI gets fresh
    data without paying the per-request HTTP cost."""
    global _gpu_status_cache
    import requests as _req
    while True:
        try:
            r = _req.get(GPU_STATUS_URL, timeout=5)
            if r.ok:
                nodes = r.json()
                now = time.time()
                new_cache = {}
                for n in nodes:
                    rni = n.get("ray_node_index")
                    if rni is None:
                        continue
                    gpus = []
                    for g in n.get("gpus", []):
                        tdp = _tdp_for(g.get("name", ""))
                        power = float(g.get("power_watt") or 0)
                        util = int(g.get("util_pct") or 0)
                        # dmon is a per-GPU object: {sm_pct, mem_pct, dec_pct,
                        # enc_pct, jpg_pct, ofa_pct}. We surface sm_pct as the
                        # canonical SM utilization (it averages over a 1s
                        # window so it's less spiky than --query-gpu util.gpu)
                        # and keep mem_pct around for a future memory-channel
                        # chart.
                        dmon = g.get("dmon") or {}
                        sm_pct = int(dmon.get("sm_pct") or 0)
                        dmon_mem_pct = int(dmon.get("mem_pct") or 0)
                        gpus.append({
                            "index": int(g.get("index", 0)),
                            "name": g.get("name", "GPU"),
                            "util_pct": util,
                            "power_watt": power,
                            "tdp_w": tdp,
                            "activity_pct": _estimate_activity_pct(util, power, tdp),
                            "sm_pct": sm_pct,
                            "dmon_mem_pct": dmon_mem_pct,
                            "mem_used_mb": int(g.get("mem_used_mb") or 0),
                            "mem_total_mb": int(g.get("mem_total_mb") or 0),
                            "temp_c": int(g.get("temp_c") or 0),
                        })
                    new_cache[int(rni)] = {
                        "hostname": n.get("hostname", "?"),
                        "internal_ip": n.get("internal_ip", ""),
                        "ts": now,
                        "gpus": gpus,
                    }
                _gpu_status_cache = new_cache
        except Exception as e:
            print(f"[gpu-status] poll error: {e}", file=sys.stderr)
        time.sleep(2)


# Start the poller daemon (only once per process).
if not getattr(_poll_gpu_status, "_started", False):
    _t = threading.Thread(target=_poll_gpu_status, daemon=True, name="gpu-status-poller")
    _t.start()
    _poll_gpu_status._started = True


@app.get("/api/ray/gpu-stats")
def ray_gpu_stats():
    """Return per-node GPU info from the Ray Dashboard API.
    Since nvidia-smi cannot run on the head node (no NVML) and GPU resources
    are fully allocated (blocking entrypoint_num_gpus jobs), we parse the
    cluster status and node summary for allocation data instead."""
    import requests as _req
    now = time.time()
    cached = _gpu_stats_cache.get("data")
    if cached and (now - _gpu_stats_cache.get("ts", 0)) < 5:
        return cached

    nodes_data = []
    total_gpu_allocated = 0
    total_gpu_count = 0
    total_cpu_pct = 0

    # 1. Get cluster-level + per-node resource usage from cluster_status
    node_gpu_usage: dict[str, dict] = {}
    try:
        cs = _req.get(f"{RAY_URL}/api/cluster_status", timeout=5).json()
        lmr = cs.get("data", {}).get("clusterStatus", {}).get("loadMetricsReport", {})
        usage = lmr.get("usage", {})
        total_cpu_pct = (usage.get("CPU", [0, 1])[0] / usage.get("CPU", [0, 1])[1] * 100) if usage.get("CPU", [0, 1])[1] else 0
        total_gpu_allocated = int(usage.get("GPU", [0, 0])[0])
        total_gpu_count = int(usage.get("GPU", [0, 0])[1])
        # Per-node usage from usageByNode
        for _nid, nu in lmr.get("usageByNode", {}).items():
            ng = nu.get("GPU", [0, 0])
            ncpu = nu.get("CPU", [0, 0])
            node_gpu_usage[_nid] = {
                "gpu_alloc": int(ng[0]),
                "gpu_total": int(ng[1]),
                "cpu_pct": round(ncpu[0] / ncpu[1] * 100, 1) if ncpu[1] else 0,
            }
    except Exception:
        pass

    # 2. Fall back to local sidecar data if the team's endpoint isn't
    # reachable. Used for the cluster-wide average only — per-GPU detail
    # comes from the team's cache below.
    _cluster_gpu_util, _per_node_util = _get_cluster_gpu_util()
    _max_gpu_detail: list = []
    for _nk, _nv in _per_node_util.items():
        if _nk:
            _ngpus = _nv.get("gpus", [])
            if len(_ngpus) > len(_max_gpu_detail):
                _max_gpu_detail = _ngpus

    # Build per-node GPU count from usageByNode (Ray knows how many GPUs
    # each node registered via --num-gpus, which is the correct count for
    # NVLink clusters where nvidia-smi shows ALL GPUs on every node).
    _node_gpu_total_by_id: dict[str, int] = {}
    for _nid, _nu in node_gpu_usage.items():
        _gt = _nu.get("gpu_total", 0)
        if _gt > 0:
            _node_gpu_total_by_id[_nid] = _gt

    # Prune stale entries from the team's gpu-status cache (older than 30s).
    _status_ts_fresh = {k: v for k, v in _gpu_status_cache.items() if now - v.get("ts", 0) < 30}

    # 4. Get per-node details from nodes?view=summary
    try:
        ns = _req.get(f"{RAY_URL}/nodes?view=summary", timeout=5).json()
        for n_idx, n in enumerate(ns.get("data", {}).get("summary", [])):
            hn = n.get("hostname", "?")
            if hn.startswith("?"):
                continue
            ip = n.get("ip", "")
            cpu_pct = n.get("cpu", 0)
            mem_info = n.get("mem", [0, 0, 0, 0])
            mem_total = mem_info[1] if len(mem_info) > 1 else 0
            mem_used = mem_info[3] if len(mem_info) > 3 else 0

            # GPU count: prefer Ray's per-node total from usageByNode,
            # then resourcesTotal, then worker slots (in that order).
            node_id = n.get("raylet", {}).get("nodeId", "")
            gpus_from_ray = _node_gpu_total_by_id.get(node_id, 0)
            gpus_from_resources = round(n.get("raylet", {}).get("resourcesTotal", {}).get("GPU", 0))
            gpu_count = gpus_from_ray or gpus_from_resources or 0

            # Per-GPU detail: prefer the team's gpu-status (already filtered
            # to Ray-managed GPUs and keyed by ray_node_index). Fall back to
            # the local sidecar slice (works for NVLink, wrong for heterogeneous
            # clusters) and finally to an empty list.
            node_gpus_detail: list = []
            node_gpu_util = 0.0
            rni = n.get("raylet", {}).get("nodeIndex") if isinstance(n.get("raylet"), dict) else None
            # Try a few ways to match: explicit ray_node_index from the team's
            # feed, then n_idx (positional match), then IP.
            status_match = _status_ts_fresh.get(n_idx)
            if status_match is None and ip:
                for v in _status_ts_fresh.values():
                    if v.get("internal_ip") == ip:
                        status_match = v
                        break
            if status_match:
                node_gpus_detail = status_match.get("gpus", [])
                if node_gpus_detail:
                    node_gpu_util = round(
                        sum(g["activity_pct"] for g in node_gpus_detail) / len(node_gpus_detail), 1
                    )
            elif _max_gpu_detail and gpu_count > 0:
                # Legacy fallback: slice the longest sidecar array. Correct
                # only for NVLink clusters where every node sees every GPU.
                node_gpus_detail = _max_gpu_detail[:gpu_count]
                if _cluster_gpu_util > 0:
                    node_gpu_util = _cluster_gpu_util

            nodes_data.append({
                "hostname": hn,
                "ip": ip,
                "cpu_pct": round(cpu_pct, 1),
                "mem_total_bytes": mem_total,
                "mem_used_bytes": mem_used,
                "gpus_allocated": gpu_count,
                "gpu_util_pct": round(node_gpu_util, 1),
                "gpus_detail": node_gpus_detail,
            })
    except Exception:
        pass

    # Cluster-wide activity: prefer the team's data (which uses
    # power-based activity estimation), fall back to sidecar cluster avg.
    team_activity_values = []
    for v in _status_ts_fresh.values():
        for g in v.get("gpus", []):
            team_activity_values.append(g.get("activity_pct", 0))
    if team_activity_values:
        cluster_gpu_util = round(sum(team_activity_values) / len(team_activity_values), 1)
        gpu_active_count = len(team_activity_values)
    else:
        cluster_gpu_util = _cluster_gpu_util
        gpu_active_count = len(_max_gpu_detail)

    result = {
        "gpus_allocated": total_gpu_allocated,
        "gpus_total": total_gpu_count or len(nodes_data),
        "cpu_pct": round(total_cpu_pct, 1),
        "nodes": nodes_data,
        "cluster": {
            "gpu_allocation_pct": round(total_gpu_allocated / total_gpu_count * 100, 1) if total_gpu_count else 0,
            "gpu_util_pct": cluster_gpu_util,
            "gpu_active_count": gpu_active_count,
            "gpu_total_count": total_gpu_count or 0,
        },
    }
    _gpu_stats_cache.update(ts=now, data=result)
    return result


@app.post("/api/deploy")
async def deploy_model_generic(body: dict):
    """Generic deploy endpoint used by test-all and DeployModels.
    Accepts { job_id, model_id, model_name, provider }."""
    job_id     = body.get("job_id", "") or body.get("model_id", "")
    model_name = body.get("model_name", "")
    provider   = body.get("provider", "ray")

    if provider == "ray":
        with get_db() as conn:
            row = conn.execute(
                "SELECT id, status, ray_serve_url, inference_provider FROM jobs WHERE id=? OR model_name=? ORDER BY created_at DESC LIMIT 1",
                (job_id, model_name),
            ).fetchone()
        if row and row["status"] == "completed":
            if row["inference_provider"] == "ray" and row["ray_serve_url"]:
                return {"status": "running", "url": row["ray_serve_url"]}
            return await deploy_model_to_ray(row["id"], body)
        raise HTTPException(404, f"No completed training job found for model '{model_name}' (id={job_id})")
    raise HTTPException(400, f"Provider '{provider}' not supported via /api/deploy. Use /api/jobs/{{id}}/deploy-ray directly.")


@app.post("/api/jobs/{job_id}/deploy-ray")
async def deploy_model_to_ray(job_id: str, body: dict = {}):
    """Deploy a specific trained model to Ray Serve."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=? AND status='completed'", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found or not completed")
    job = dict(row)

    ray_dashboard_url = (body.get("ray_dashboard_url") or RAY_URL or "").rstrip("/")
    if not ray_dashboard_url:
        raise HTTPException(status_code=400, detail="ray_dashboard_url required")

    state = _get_model_deploy_state(job_id)
    if state["status"] == "deploying":
        raise HTTPException(status_code=409, detail="Deploy already in progress for this model")

    # Derive serve URL: replace 8265 → 8000, route prefix = /model/{job_id}
    try:
        from urllib.parse import urlparse as _urlparse
        _p = _urlparse(ray_dashboard_url)
        base_serve = f"{_p.scheme}://{_p.hostname}:8000"
    except Exception:
        base_serve = ray_dashboard_url.replace(":8265", ":8000")
    serve_url = f"{base_serve}/model/{job_id}"

    # MinIO URL reachable from Ray workers (use public IP, not Docker service name)
    minio_url_for_ray = _resolve_minio_url_for_ray()

    import base64 as _b64

    # Pip dependencies based on training_type / engine
    # The script runs ON the cluster as a Ray Job — no Ray Client connection needed.
    # numpy<2.0 ensures ABI compatibility with cluster's pandas.
    engine = job.get("engine", "Ultralytics")
    tt = job.get("training_type", "detection")
    pip_base = ["fastapi", "uvicorn", "python-multipart", "pillow", "boto3", "numpy<2.0"]
    if tt in ("detection", "segmentation") and engine == "Ultralytics":
        pip_base += ["ultralytics"]
    elif engine == "MONAI":
        pip_base += ["torch", "torchvision", "monai"]
    elif engine == "HuggingFace":
        pip_base += ["torch", "torchvision", "transformers", "accelerate"]
    elif tt == "classification":
        pip_base += ["torch", "torchvision", "timm"]
    elif tt == "segmentation" and engine == "Segmentation Models PyTorch":
        pip_base += ["torch", "torchvision", "segmentation-models-pytorch"]
    elif tt in ("llm-text", "vlm-finetune"):
        pip_base += ["torch", "transformers", "peft", "accelerate"]

    num_classes = job.get("num_classes") or 2
    class_names_raw = job.get("class_names") or "[]"

    payload = {
        "runtime_env": {
            "pip": pip_base,
            "env_vars": {
                "TRAINING_TYPE":  tt,
                "ENGINE":         engine,
                "MODEL_NAME":     job.get("model_name", ""),
                "MODEL_ID":       job_id,
                "MINIO_URL":      minio_url_for_ray,
                "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
                "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
                "WEIGHTS_BUCKET": "medimage-weights",
                "WEIGHTS_KEY":    job.get("s3_weights_path") or f"{job_id}/best.pt",
                "NUM_CLASSES":    str(num_classes),
                "CLASS_NAMES":    class_names_raw if isinstance(class_names_raw, str) else json.dumps(class_names_raw),
            },
        },
    }

    state.update(status="deploying", url=serve_url, logs=["Submitting model to Ray cluster…"])
    threading.Thread(
        target=_submit_model_deploy,
        args=(ray_dashboard_url, payload, serve_url, job_id),
        kwargs={"max_wait": 600},
        daemon=True,
    ).start()
    return {"status": "deploying", "serve_url": serve_url}


@app.get("/api/jobs/{job_id}/deploy-ray/status")
def model_deploy_status(job_id: str):
    state = _get_model_deploy_state(job_id)
    # If in-memory state shows we're working (deploying) or done (running)
    # and we've already resolved a URL, trust it.
    if state["status"] in ("deploying", "running") and state.get("url"):
        return {
            "status": state["status"],
            "url": state["url"],
            "logs": state["logs"][-20:],
        }
    # Fallback: in-memory state was reset (e.g. after API restart). Read
    # the persisted deployment from the DB so the UI can offer Stop /
    # Undeploy on already-deployed models.
    with get_db() as conn:
        row = conn.execute(
            "SELECT inference_provider, ray_serve_url, modal_url FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return {"status": "idle", "url": None, "logs": []}
    provider = (row["inference_provider"] or "").strip()
    ray_url  = (row["ray_serve_url"] or "").strip()
    if provider == "ray" and ray_url:
        return {
            "status": "running",
            "url":    ray_url,
            "logs":   ["Recovered from DB after API restart"],
        }
    return {"status": "idle", "url": None, "logs": []}


@app.post("/api/jobs/{job_id}/deploy-ray/stop")
def model_deploy_stop(job_id: str):
    """Kill the named Ray Actor for this model via Ray Jobs API (no Ray Client)."""
    actor_name = f"model-{job_id}"
    logs = []
    try:
        import requests as _req
        import base64 as _b64lib
        # Stronger kill: no_restart=True (otherwise Ray's supervisor may
        # respawn the actor on the next health probe) + a follow-up
        # ray.get_actor() check to confirm the death actually stuck
        # before returning. The previous plain ray.kill() reported
        # KILLED but the actor state stayed ALIVE for minutes, which
        # let stale deploys keep holding GPU resources.
        _kill_script = (
            "import ray, sys, time, os\n"
            "ray.init(address='auto', ignore_reinit_error=True, log_to_driver=False)\n"
            "actor_name = os.environ.get('ACTOR_NAME','')\n"
            "if not actor_name:\n"
            "    print('NO_ACTOR_NAME', flush=True); sys.exit(1)\n"
            "try:\n"
            "    actor = ray.get_actor(actor_name, namespace='default')\n"
            "    print(f'FOUND:{actor_name}', flush=True)\n"
            "    ray.kill(actor, no_restart=True)\n"
            "    print(f'KILLED:{actor_name}', flush=True)\n"
            "    time.sleep(2)\n"
            "    try:\n"
            "        ray.get_actor(actor_name, namespace='default')\n"
            "        print(f'STILL_ALIVE:{actor_name}', flush=True)\n"
            "    except Exception:\n"
            "        print(f'CONFIRMED_DEAD:{actor_name}', flush=True)\n"
            "except Exception as e:\n"
            "    print(f'NOT_FOUND:{e}', flush=True)\n"
        )
        _b64 = _b64lib.b64encode(_kill_script.encode()).decode()
        _payload = {
            "entrypoint": f"bash -c 'echo {_b64} | base64 -d > /tmp/kill_actor.py && ACTOR_NAME={actor_name} python3 /tmp/kill_actor.py'",
            "runtime_env": {},
        }
        # Use the same dashboard URL as deploy
        _dashboard = "http://100.68.53.118:8265"
        _resp = _req.post(f"{_dashboard}/api/jobs/", json=_payload, timeout=15)
        _resp.raise_for_status()
        _submit_id = _resp.json().get("submission_id") or _resp.json().get("job_id")
        # Wait for completion (kill should be fast)
        for _ in range(20):
            time.sleep(1)
            try:
                _st = _req.get(f"{_dashboard}/api/jobs/{_submit_id}", timeout=5).json()
                if _st.get("status") in ("SUCCEEDED", "FAILED", "STOPPED"):
                    break
            except Exception:
                pass
        # Fetch result from logs
        try:
            _lr = _req.get(f"{_dashboard}/api/jobs/{_submit_id}/logs", timeout=10)
            if _lr.ok:
                _lt = _lr.json().get("logs", "")
                if "CONFIRMED_DEAD:" in _lt:
                    logs.append(f"[stop] Actor '{actor_name}' killed + confirmed dead")
                elif "KILLED:" in _lt and "STILL_ALIVE:" in _lt:
                    logs.append(f"[stop] Actor '{actor_name}' kill reported but state stayed ALIVE — Ray supervisor didn't reap")
                elif "KILLED:" in _lt:
                    logs.append(f"[stop] Actor '{actor_name}' killed (verification inconclusive)")
                elif "NOT_FOUND:" in _lt:
                    logs.append(f"[stop] Actor '{actor_name}' not found (already stopped)")
                else:
                    logs.append(f"[stop] Job exited: {_lt[-200:]}")
        except Exception as _e:
            logs.append(f"[stop] Could not fetch kill job logs: {_e}")
    except Exception as e:
        logs.append(f"[stop] warning: {e}")
    # After the actor is killed, wait until a GPU is actually free
    # on the cluster. The kill completes on Ray's side but the resource
    # accounting + actor reaping is async — without this wait the next
    # training job submitted right after the stop API returns would
    # still see "GPU: 4/4 used" and get stuck in PENDING. The test-all
    # bulk loop calls this endpoint between every iteration precisely
    # so we hold the loop until a GPU slot opens up.
    try:
        import requests as _req2
        for _wait_i in range(30):  # up to ~30s
            time.sleep(1)
            _cs = _req2.get(f"{RAY_URL}/api/cluster_status", timeout=4).json()
            _ru = _cs.get("data", {}).get("autoscalingStatus", "")
            # ResourceUsage: 7.0/240.0 CPU, 4.0/4.0 GPU, ...
            try:
                _gpu_chunk = _ru.split("ResourceUsage:")[1].split("\n")[0]
                _used_g = float(_gpu_chunk.split("GPU,")[0].rsplit(" ", 1)[-1])
                _total_g = float(_gpu_chunk.split("GPU,")[1].split(" ", 1)[0].rstrip("/"))
                if _used_g < _total_g:
                    logs.append(f"[stop] GPU available: {_used_g}/{_total_g} (waited {_wait_i+1}s)")
                    break
            except Exception:
                pass
        else:
            logs.append("[stop] GPU still not free after 30s — caller may see PENDING")
    except Exception as _e:
        logs.append(f"[stop] GPU-wait check failed (non-fatal): {_e}")
    # Clear DB state
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET ray_serve_url='', inference_provider='' WHERE id=?",
            (job_id,),
        )
        conn.commit()
    state = _get_model_deploy_state(job_id)
    state.update(status="idle", url=None, logs=logs)
    return {"ok": True, "logs": logs}


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


@app.post("/api/text-datasets/create-sample")
def create_sample_text_dataset(body: dict | None = None):
    """Create a small sample text dataset for smoke-testing LLM/VLM training.

    Writes a JSONL file with the requested format (alpaca | sharegpt |
    plain) and registers it in the text_datasets table. If a sample
    already exists, the existing id is returned.
    """
    body = body or {}
    fmt = (body.get("format") or "alpaca").lower()
    if fmt not in ("alpaca", "sharegpt", "plain"):
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")
    n_rows = int(body.get("rows") or 20)
    n_rows = max(1, min(n_rows, 200))
    name = (body.get("name") or f"sample-{fmt}").strip() or f"sample-{fmt}"

    # If a sample with this name already exists, just return it.
    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM text_datasets WHERE name = ?", (name,)
        ).fetchone()
    if existing:
        d = dict(existing)
        d.pop("preview", None)
        return {"created": False, **d}

    ds_id = str(uuid.uuid4())[:8]
    save_dir = "/data/text-datasets"
    _os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{ds_id}.jsonl"

    # Generate sample rows. Use small medical-industrial seed to make the
    # smoke test reflect what the platform will actually fine-tune.
    if fmt == "alpaca":
        topics = [
            ("What is the recommended torque for an M8 bolt?",
             "For a standard M8 bolt (grade 8.8), apply 25 Nm of torque with a light oil film."),
            ("How do I identify a defective PCB solder joint?",
             "Look for: dull grainy surface, insufficient fillet, exposed copper pad, or a tombstone shape."),
            ("What causes a YOLOv8 model to overfit on small datasets?",
             "Insufficient data augmentation, too many epochs, batch size too small, or a backbone too large for the dataset size."),
            ("How do I read a bearing fault vibration spectrum?",
             "Identify the rotating frequency (1× RPM), then check for harmonics and sidebands; outer race defects appear at BPFO frequencies."),
        ]
        rows = [
            {"instruction": inst, "input": "", "output": out}
            for inst, out in (topics * ((n_rows // len(topics)) + 1))[:n_rows]
        ]
    elif fmt == "sharegpt":
        topics = [
            ("What is the difference between spot welding and projection welding?",
             "Spot welding concentrates current at a single point via electrode tips; projection welding uses pre-formed projections on one workpiece to localize current without precision electrode alignment."),
            ("How can I detect surface cracks in metal castings?",
             "Use magnetic particle inspection for ferromagnetic alloys, or dye-penetrant testing for non-magnetic materials. Visual inspection under oblique lighting catches >0.5mm cracks."),
            ("What is the optimal learning rate for fine-tuning a 7B LLM with LoRA?",
             "For Unsloth 4-bit + LoRA r=16, start at 2e-4 with cosine schedule; reduce to 5e-5 if loss diverges in the first 50 steps."),
        ]
        rows = [
            {"conversations": [
                {"from": "human", "value": q},
                {"from": "gpt",   "value": a},
            ]}
            for q, a in (topics * ((n_rows // len(topics)) + 1))[:n_rows]
        ]
    else:  # plain
        topics = [
            "Edge inference on Jetson Orin Nano achieves 25 FPS with quantized YOLOv8n at 320×320 input.",
            "Defect recall improved from 0.78 to 0.91 after augmenting the training set with synthetic weld spatter overlays.",
            "Ray cluster auto-scales from 1 to 8 workers based on queued job depth; the dashboard exposes live worker count.",
        ]
        rows = [{"text": t} for t in (topics * ((n_rows // len(topics)) + 1))[:n_rows]]

    with open(save_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    size_bytes = _os.path.getsize(save_path)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO text_datasets (id, name, format, path, row_count, size_bytes, created_at) VALUES (?,?,?,?,?,?,?)",
            (ds_id, name, fmt, save_path, len(rows), size_bytes, time.time()),
        )
        conn.commit()
    return {
        "created":   True,
        "id":        ds_id,
        "name":      name,
        "format":    fmt,
        "row_count": len(rows),
        "size_bytes": size_bytes,
        "path":      save_path,
    }


# ─── HuggingFace Dataset Import ──────────────────────────────────────────────

_hf_import_jobs: dict = {}  # job_id → {status, progress, log, bucket, error, done_files, total_files}


class HFImportRequest(BaseModel):
    hf_dataset_id: str
    bucket_name: str
    max_files: int = 20
    # Optional explicit list of file paths inside the repo. When
    # provided, these are used verbatim (subject to max_files cap and
    # the 200 MB per-file size guard). When None, the legacy
    # "first N by priority (parquet → jsonl/json/csv → images → other)"
    # behaviour is used as a fallback.
    selected_files: list[str] | None = None


def _log_hf(job_id: str, line: str):
    ts = datetime.now().strftime("%H:%M:%S")
    _hf_import_jobs[job_id]["log"] += f"[{ts}] {line}\n"


def _run_hf_import(job_id: str, req: HFImportRequest, user_id: str = ""):
    import requests as _req
    import boto3 as _boto3
    from botocore.exceptions import ClientError as _CE
    # Use the submitter's saved HF token (so gated/private dataset repos
    # are accessible). Falls back to the API process env if absent.
    _hf_token = _load_user_hf_token(user_id) or os.getenv("HF_TOKEN", "") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    _hf_headers = {"Authorization": f"Bearer {_hf_token}"} if _hf_token else {}

    job = _hf_import_jobs[job_id]
    job["status"] = "running"
    try:
        # 1. Create MinIO bucket
        _log_hf(job_id, f"Connecting to MinIO ({MINIO_URL})...")
        s3 = _boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
        )
        try:
            s3.create_bucket(Bucket=req.bucket_name)
            _log_hf(job_id, f"✓ Bucket '{req.bucket_name}' created")
        except _CE as e:
            code = e.response["Error"]["Code"]
            if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                _log_hf(job_id, f"ℹ Bucket '{req.bucket_name}' already exists — reusing")
            else:
                raise Exception(f"MinIO error: {e.response['Error']['Message']}")

        job["progress"] = 10

        # 2. List files from HF dataset repo
        _log_hf(job_id, f"Fetching file tree: huggingface.co/datasets/{req.hf_dataset_id}...")
        tree_url = f"https://huggingface.co/api/datasets/{req.hf_dataset_id}/tree/main"
        r = _req.get(tree_url, headers=_hf_headers, timeout=30)
        if not r.ok:
            raise Exception(f"HuggingFace API returned HTTP {r.status_code} — dataset not found or is private")

        entries = r.json()
        # HF tree API returns type='file' or 'lfs' (LFS pointer), NOT
        # 'blob'. The old filter (`type == "blob"`) silently matched
        # nothing, so every HF import was uploading 0 files even though
        # the user had selected a real repo. This is what the picker
        # in the UI is fixing too (src/pages/Datasets.tsx).
        blobs = [e for e in entries if e.get("type") in ("file", "lfs")]

        # If no blobs at root, try data/ subdirectory
        if not blobs:
            r2 = _req.get(f"{tree_url}/data", headers=_hf_headers, timeout=30)
            if r2.ok:
                blobs += [e for e in r2.json() if e.get("type") == "blob"]

        # Remove gitattributes
        blobs = [b for b in blobs if not b["path"].lower().endswith(".gitattributes")]

        if not blobs:
            raise Exception("No downloadable files found in this dataset repo")

        # Sort: parquet → jsonl/json/csv → images → other
        def _prio(e: dict) -> int:
            p = e["path"].lower()
            if p.endswith(".parquet"): return 0
            if p.endswith((".jsonl", ".json", ".csv")): return 1
            if p.endswith((".jpg", ".jpeg", ".png", ".webp")): return 2
            if p.endswith(".txt"): return 3
            return 9

        # If the user picked specific files in the UI, use those
        # verbatim. Otherwise fall back to "first N by priority".
        if req.selected_files:
            wanted = set(req.selected_files)
            blobs = [b for b in blobs if b["path"] in wanted]
            blobs.sort(key=_prio)
            not_found = wanted - {b["path"] for b in blobs}
            if not_found:
                _log_hf(job_id, f"⚠ {len(not_found)} selected file(s) not in repo: {', '.join(list(not_found)[:3])}{'…' if len(not_found) > 3 else ''}")
        else:
            blobs.sort(key=_prio)
        blobs = blobs[:req.max_files]
        _log_hf(job_id, f"Found {len(blobs)} file(s) to download")
        job["progress"] = 15
        job["total_files"] = len(blobs)
        job["done_files"] = 0

        # 3. Download each file and upload to MinIO
        done = 0
        for i, blob in enumerate(blobs):
            path = blob["path"]
            size_bytes = blob.get("size", 0)

            # Skip files > 200 MB
            if size_bytes > 200 * 1024 * 1024:
                _log_hf(job_id, f"  ⚠ Skip {path} — too large ({size_bytes // 1024 // 1024} MB)")
                continue

            sz_str = (
                f"{size_bytes / 1024 / 1024:.1f} MB" if size_bytes > 1024 * 1024
                else f"{size_bytes // 1024} KB"
            )
            _log_hf(job_id, f"↓ [{i+1}/{len(blobs)}] {path}  ({sz_str})")

            hf_url = f"https://huggingface.co/datasets/{req.hf_dataset_id}/resolve/main/{path}"
            dl = _req.get(hf_url, headers=_hf_headers, timeout=120)
            if not dl.ok:
                _log_hf(job_id, f"  ⚠ HTTP {dl.status_code} — skipping")
                continue

            s3_key = path.lstrip("/")
            s3.put_object(Bucket=req.bucket_name, Key=s3_key, Body=dl.content)
            _log_hf(job_id, f"  ✓ s3://{req.bucket_name}/{s3_key}")
            done += 1
            job["done_files"] = done
            job["progress"] = 15 + int((i + 1) / len(blobs) * 80)

        job["progress"] = 100
        job["status"] = "completed"
        _log_hf(job_id, "")
        _log_hf(job_id, "=== Import complete ===")
        _log_hf(job_id, f"Bucket : {req.bucket_name}")
        _log_hf(job_id, f"Files  : {done}/{len(blobs)} uploaded")
        _log_hf(job_id, "MinIO console → http://localhost:9001")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        _log_hf(job_id, f"ERROR: {e}")


@app.get("/api/datasets/inspect-hf-file")
def inspect_hf_file(repo_id: str, path: str, request: Request):
    """Inspect a file in a HuggingFace dataset repo.

    For .zip archives: list the central directory via two HTTP Range
    requests (last 256 KB to find the EOCD record, then the central
    directory itself). NO full download — works even for multi-GB
    archives. The user sees the file's internal structure before
    committing to download hundreds of MB.

    For .tar / .tar.gz / .tgz archives: we need the full file because
    tar has no central directory — the entry headers are interleaved
    with the file data. Capped at 200 MB.

    For other files: return a stub (download URL) so the frontend can
    show a "preview not supported for this file type" message.
    """
    import requests as _req
    import zipfile as _zlib
    import tarfile as _tarlib
    import struct as _struct
    import io as _io

    _user_id = _current_user_id(request)
    _hf_token = _load_user_hf_token(_user_id) or os.getenv("HF_TOKEN", "") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    _hf_headers = {"Authorization": f"Bearer {_hf_token}"} if _hf_token else {}
    _hf_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"

    p_lower = path.lower()
    is_zip  = p_lower.endswith(".zip")
    is_tar  = p_lower.endswith(".tar") or p_lower.endswith(".tar.gz") or p_lower.endswith(".tgz")
    # Text-previewable file types: markdown, plain text, json, csv,
    # yaml/yml, log. For these, we fetch a HEAD to get the size and a
    # capped Range download for the body so big files don't pin the
    # API. The body is returned as a UTF-8 string in the response.
    is_text = any(p_lower.endswith(ext) for ext in (
        ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".log", ".xml", ".html", ".py", ".js", ".ts",
    ))

    if not (is_zip or is_tar or is_text):
        return {
            "kind": "unsupported",
            "repo_id": repo_id,
            "path": path,
            "message": "Preview is available for archives (.zip / .tar / .tar.gz) and text files (.md / .txt / .json / .csv / .yaml / …).",
            "download_url": _hf_url,
        }

    try:
        if is_zip:
            # ── Two-Range-request zip central directory parser ──────────
            # 1) Get file size via HEAD (HF CDN returns Content-Length).
            # 2) Range-fetch the last 256 KB to find the EOCD record
            #    (EOCD signature PK\x05\x06 is at the very end, with up
            #    to 65535 bytes of comment + 22 bytes of record — the last
            #    256 KB window catches every EOCD even with max-length
            #    comments and disk-spanning data descriptors).
            # 3) Parse EOCD to find central directory offset + size.
            # 4) Range-fetch exactly the central directory bytes.
            # 5) Walk central directory entries, decoding per-entry
            #    fields. No full file download — works for 10 GB zips.

            head = _req.head(_hf_url, headers=_hf_headers, timeout=15, allow_redirects=True)
            file_size = int(head.headers.get("Content-Length") or 0)
            if file_size <= 0:
                # Some HF mirrors don't return Content-Length on HEAD
                # (LFS). Fall back to a single range request sized to
                # a safe window starting at file_size-256KB.
                file_size = None
                tail = b""
                r = _req.get(_hf_url, headers={**_hf_headers, "Range": "bytes=-262144"}, timeout=30)
                if r.status_code == 206:
                    tail = r.content
                elif r.status_code == 200:
                    # Server ignored Range and returned full file —
                    # bail to the legacy full-download path below.
                    tail = r.content
                    file_size = len(tail)
                # else: server returned 416 or error — fall through.
            else:
                tail_size = min(262144, file_size)
                r = _req.get(
                    _hf_url,
                    headers={**_hf_headers, "Range": f"bytes={file_size - tail_size}-{file_size - 1}"},
                    timeout=30,
                )
                tail = r.content if r.status_code == 206 else b""

            if not tail:
                # HEAD didn't return a size, or range request failed.
                # Fall back to legacy full download (capped at 200 MB).
                r = _req.get(_hf_url, headers=_hf_headers, timeout=120, stream=True)
                buf = b""
                total = 0
                for chunk in r.iter_content(chunk_size=1 << 20):
                    buf += chunk
                    total += len(chunk)
                    if total > 200 * 1024 * 1024:
                        return {"kind": "too_large", "repo_id": repo_id, "path": path, "size_bytes": total, "message": "Archive >200 MB; partial listing not supported yet."}
                zf = _zlib.ZipFile(_io.BytesIO(buf))
                entries = [{"path": i.filename, "size": i.file_size, "compressed": i.compress_size, "is_dir": i.is_dir()} for i in zf.infolist()]
                zf.close()
                entries.sort(key=lambda e: (not e["is_dir"], e["path"]))
                return {
                    "kind": "zip", "repo_id": repo_id, "path": path,
                    "size_bytes": total, "entry_count": len(entries),
                    "entries": entries[:500], "truncated": len(entries) > 500,
                    "method": "full_download_fallback",
                }

            # 2) Find EOCD. Search backwards for the 4-byte signature.
            eocd_idx = tail.rfind(b"PK\x05\x06")
            if eocd_idx < 0:
                # EOCD not in last 256 KB — file is malformed, or has
                # a comment longer than 256 KB (ZIP spec max is 65535,
                # so this means the file is broken). Fall back to full.
                return {"kind": "too_large", "repo_id": repo_id, "path": path, "size_bytes": len(tail), "message": "Could not locate EOCD record in last 256 KB; archive may be malformed."}
            eocd = tail[eocd_idx:eocd_idx + 22]
            if len(eocd) < 22:
                return {"kind": "too_large", "repo_id": repo_id, "path": path, "size_bytes": len(tail), "message": "Truncated EOCD record."}
            _sig, disk_num, cd_disk, n_entries_this, n_entries_total, cd_size, cd_offset, comment_len = _struct.unpack("<IHHHHIIH", eocd)
            if disk_num != 0 or cd_disk != 0:
                # Multi-disk zip — not supported in partial-read mode
                return {"kind": "too_large", "repo_id": repo_id, "path": path, "size_bytes": len(tail), "message": "Multi-disk zip not supported; full download required."}
            if file_size is None:
                # No Content-Length from HEAD; can't safely compute
                # absolute offsets for the next Range request.
                return {"kind": "too_large", "repo_id": repo_id, "path": path, "size_bytes": len(tail), "message": "Server did not return file size; cannot range-fetch central directory."}
            # 3+4) Range-fetch the central directory (cd_offset is
            # relative to the start of the archive).
            cd_start = cd_offset
            cd_end   = cd_offset + cd_size - 1
            if cd_start + cd_size > file_size:
                return {"kind": "too_large", "repo_id": repo_id, "path": path, "size_bytes": file_size, "message": "Central directory offset exceeds file size; archive may be truncated."}
            r = _req.get(
                _hf_url,
                headers={**_hf_headers, "Range": f"bytes={cd_start}-{cd_end}"},
                timeout=60,
            )
            if r.status_code != 206:
                return {"kind": "too_large", "repo_id": repo_id, "path": path, "size_bytes": file_size, "message": f"Central directory range request returned HTTP {r.status_code}."}
            cd_bytes = r.content

            # 5) Walk central directory entries. Each CD entry:
            #   signature "PK\x01\x02" (4 bytes) + 42 bytes of fixed
            #   fields + variable filename + extra + comment.
            # Fixed fields (after sig): 11 H's + 5 I's = 16 placeholders,
            # 42 bytes. Field order: ver ver_need flags method mod_time
            #   mod_date crc(I) comp_size(I) uncomp_size(I) fname_len
            #   extra_len comment_len disk_no int_attr ext_attr(I)
            #   local_header_offset(I)
            entries = []
            i = 0
            while i < len(cd_bytes) - 46:
                if cd_bytes[i:i+4] != b"PK\x01\x02":
                    break
                (
                    _v, _vm, _flags, _method, _mod_time, _mod_date,
                    _crc, comp_size, uncomp_size,
                    fname_len, extra_len, comment_len, _disk_no,
                    _int_attr, _ext_attr, _local_off
                ) = _struct.unpack(
                    "<HHHHHHIIIHHHHHII", cd_bytes[i + 4:i + 46]
                )
                # Data-descriptor variant: if the GP flag is set, the
                # 12-byte CRC/sizes are in a trailing data descriptor
                # after the file data, not here. uncomp_size will be 0
                # in that case. We don't need the real size for preview,
                # so 0 is fine.
                name = cd_bytes[i+46:i+46+fname_len].decode("utf-8", errors="replace")
                # local_header_offset (start of the local file entry)
                # is the most reliable pointer to the file's actual
                # bytes — but we don't fetch the local headers for the
                # preview, only the central directory.
                is_dir = name.endswith("/")
                entries.append({
                    "path": name,
                    "size": uncomp_size,
                    "compressed": comp_size,
                    "is_dir": is_dir,
                })
                i += 46 + fname_len + extra_len + comment_len
            entries.sort(key=lambda e: (not e["is_dir"], e["path"]))
            return {
                "kind": "zip",
                "repo_id": repo_id,
                "path": path,
                "size_bytes": file_size,
                "entry_count": len(entries),
                "entries": entries[:500],
                "truncated": len(entries) > 500,
                "method": "central_directory_range",  # the fast path
            }
        elif is_tar:
            # tar — index is at the end too, but tar has no central
            # directory. We need the whole file. Cap at 200 MB.
            r = _req.get(_hf_url, headers=_hf_headers, timeout=300, stream=True)
            total = 0
            buf = b""
            for chunk in r.iter_content(chunk_size=1 << 20):
                buf += chunk
                total += len(chunk)
                if total > 200 * 1024 * 1024:
                    return {"kind": "too_large", "repo_id": repo_id, "path": path, "size_bytes": total, "message": "Archive >200 MB; partial listing not supported yet."}
            tf = _tarlib.open(fileobj=_io.BytesIO(buf), mode="r:*")
            entries = []
            for info in tf:
                entries.append({
                    "path": info.name,
                    "size": info.size,
                    "is_dir": info.isdir(),
                })
            tf.close()
            entries.sort(key=lambda e: (not e["is_dir"], e["path"]))
            return {
                "kind": "tar",
                "repo_id": repo_id,
                "path": path,
                "size_bytes": total,
                "entry_count": len(entries),
                "entries": entries[:500],
                "truncated": len(entries) > 500,
            }
        elif is_text:
            # ── Text-file preview (.md / .txt / .json / .csv / …) ──
            # Cap at 256 KB — enough for a README, JSON schema, or the
            # first chunk of a CSV. Bigger files get truncated + a flag.
            TEXT_PREVIEW_CAP = 256 * 1024
            head = _req.head(_hf_url, headers=_hf_headers, timeout=15, allow_redirects=True)
            file_size = int(head.headers.get("Content-Length") or 0) or None
            r = _req.get(
                _hf_url,
                headers={**_hf_headers, "Range": f"bytes=0-{TEXT_PREVIEW_CAP - 1}"},
                timeout=60,
            )
            if r.status_code == 206:
                body_bytes = r.content
            elif r.status_code == 200:
                # Server ignored Range — cap the response ourselves.
                body_bytes = r.content[:TEXT_PREVIEW_CAP]
            else:
                raise _req.HTTPError(r.status_code, f"HTTP {r.status_code}")
            # Decode UTF-8 with replacement for binary files that
            # happen to share an extension. The frontend renders
            # whatever comes out as text.
            try:
                body_text = body_bytes.decode("utf-8")
            except UnicodeDecodeError:
                body_text = body_bytes.decode("utf-8", errors="replace")
            truncated = (file_size is None and len(body_bytes) == TEXT_PREVIEW_CAP) or \
                        (file_size is not None and file_size > len(body_bytes))
            return {
                "kind": "text",
                "repo_id": repo_id,
                "path": path,
                "size_bytes": file_size,
                "preview_bytes": len(body_bytes),
                "truncated": truncated,
                "content": body_text,
            }
    except _req.HTTPError as e:
        raise HTTPException(e.response.status_code, f"HF download failed: {e.response.reason}")
    except _zlib.BadZipFile:
        raise HTTPException(400, "File is not a valid zip archive")
    except Exception as e:
        raise HTTPException(500, f"Inspect failed: {e}")


@app.post("/api/datasets/import-hf")
def import_hf_dataset(req: HFImportRequest, request: Request):
    import re as _re
    if not _re.match(r"^[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]$", req.bucket_name):
        raise HTTPException(
            status_code=400,
            detail="Bucket name: 3–63 lowercase letters/numbers/hyphens, no leading/trailing hyphens",
        )
    if not req.hf_dataset_id.strip():
        raise HTTPException(400, detail="HuggingFace dataset ID is required")
    _user_payload = _extract_token_payload(request)
    user_id = _user_payload.get("sub", "") if _user_payload else ""

    job_id = str(uuid.uuid4())[:8]
    _hf_import_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "log": "",
        "bucket": req.bucket_name,
        "hf_id": req.hf_dataset_id,
        "error": None,
        "done_files": 0,
        "total_files": 0,
    }
    threading.Thread(target=_run_hf_import, args=(job_id, req, user_id), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/datasets/import-hf/{job_id}")
def get_hf_import_status(job_id: str):
    job = _hf_import_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/datasets/buckets")
def list_minio_buckets():
    try:
        import boto3 as _boto3
        s3 = _boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
        )
        resp = s3.list_buckets()
        buckets = [
            {"name": b["Name"], "created": b["CreationDate"].isoformat()}
            for b in resp.get("Buckets", [])
        ]
        return {"buckets": buckets}
    except Exception as e:
        return {"buckets": [], "error": str(e)}


@app.get("/api/dashboard/stats")
def dashboard_stats():
    """Aggregated stats for the Dashboard page. Returns LS task counts
    across all projects, job counts by status, MinIO bucket sizes, and
    the saved Ray cluster URL — in a single call."""
    import boto3 as _boto3
    import httpx as _httpx
    from botocore.config import Config as _BotoConfig

    # ── Label Studio: sum task_number / finished_task_number across projects ──
    ls_total = 0
    ls_labeled = 0
    ls_projects = 0
    ls_error: str | None = None
    try:
        r = _httpx.get(
            f"{LS_API_URL}/api/projects/?page_size=1000",
            headers={"Authorization": f"Token {LS_TOKEN}"},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            for p in data.get("results", []):
                ls_total   += int(p.get("task_number", 0) or 0)
                ls_labeled += int(p.get("finished_task_number", 0) or 0)
            ls_projects = data.get("count", len(data.get("results", [])))
        else:
            ls_error = f"LS /api/projects returned {r.status_code}"
    except Exception as e:
        ls_error = str(e)

    # ── Jobs: count by status from SQLite ──
    jobs_by_status: dict[str, int] = {}
    jobs_total = 0
    with get_db() as _jc:
        for r in _jc.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status").fetchall():
            jobs_by_status[r["status"] or "unknown"] = int(r["c"])
            jobs_total += int(r["c"])

    # ── MinIO: bucket sizes via boto3 (paginated internally) ──
    buckets: list[dict] = []
    buckets_error: str | None = None
    try:
        s3 = _boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
            config=_BotoConfig(s3={"addressing_style": "path"}),
        )
        for b in s3.list_buckets().get("Buckets", []):
            name = b["Name"]
            total_size = 0
            total_objects = 0
            token: str | None = None
            pages = 0
            while True:
                kwargs = {"Bucket": name, "MaxKeys": 1000}
                if token:
                    kwargs["ContinuationToken"] = token
                resp = s3.list_objects_v2(**kwargs)
                for obj in resp.get("Contents", []) or []:
                    total_size += int(obj.get("Size", 0) or 0)
                    total_objects += 1
                if not resp.get("IsTruncated"):
                    break
                token = resp.get("NextContinuationToken")
                pages += 1
                if pages >= 100:  # safety cap: 100k objects per bucket
                    break
            buckets.append({
                "name": name,
                "size_bytes": total_size,
                "object_count": total_objects,
            })
    except Exception as e:
        buckets_error = str(e)

    # ── Ray URL: prefer saved setting, fall back to env default ──
    ray_url = RAY_URL
    try:
        with get_db() as _sc:
            row = _sc.execute("SELECT value FROM settings WHERE key='ray_head_url'").fetchone()
        if row and row["value"]:
            ray_url = row["value"]
    except Exception:
        pass

    return {
        "label_studio": {
            "total_tasks": ls_total,
            "labeled":     ls_labeled,
            "projects":    ls_projects,
            "error":       ls_error,
        },
        "jobs": {
            "total":     jobs_total,
            "by_status": jobs_by_status,
        },
        "storage": {
            "buckets": buckets,
            "error":   buckets_error,
        },
        "ray": {
            "url": ray_url,
        },
    }


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/api/ls-goto/{project_id}")
async def ls_goto(project_id: int, request: Request):
    """Server-side LS auto-login: logs in via internal network, sets sessionid
    cookie for the browser's host (port-agnostic per RFC 6265), then redirects
    to the LS project page at the public URL."""
    import requests as _req
    from fastapi.responses import RedirectResponse as _Redir

    ls_internal  = LS_API_URL.rstrip("/")            # http://label-studio:8080
    ls_public    = LS_PUBLIC_URL.rstrip("/")          # http://100.68.3.42:8085
    ls_user      = os.getenv("LS_USER",  "admin@medimage.local")
    ls_pass      = os.getenv("LS_PASSWORD", "admin")
    target       = f"{ls_public}/projects/{project_id}/" if project_id > 0 else f"{ls_public}/projects/"

    try:
        sess = _req.Session()
        # 1. GET login page → collect CSRF token + csrftoken cookie
        r1 = sess.get(f"{ls_internal}/user/login/", timeout=5)
        import re as _re
        m = _re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', r1.text)
        csrf = m.group(1) if m else ""

        # 2. POST login → collect sessionid cookie
        sess.post(
            f"{ls_internal}/user/login/",
            data={"csrfmiddlewaretoken": csrf, "email": ls_user, "password": ls_pass},
            headers={"Referer": f"{ls_internal}/user/login/", "Origin": ls_internal},
            allow_redirects=False,
            timeout=5,
        )
        sessionid = sess.cookies.get("sessionid")
    except Exception:
        sessionid = None

    resp = _Redir(url=target, status_code=302)
    if sessionid:
        # Set sessionid for the browser's host (port-agnostic → applies to :8085)
        host = request.headers.get("x-forwarded-host", request.headers.get("host", "")).split(":")[0]
        resp.set_cookie("sessionid", sessionid, domain=host or None, path="/", samesite="lax", httponly=True)
    return resp


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

# UploadFile, File, FForm already imported above
from typing import Optional

# Color palette used when normalizing external (Modal) inference responses.
# Index is `hash(label) % len` for a stable color per class.
_INFER_COLORS = [
    "#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6",
    "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#06b6d4",
    "#a855f7", "#84cc16", "#eab308", "#22c55e", "#0ea5e9",
]


def _color_for(label: str) -> str:
    if not label:
        return _INFER_COLORS[0]
    return _INFER_COLORS[abs(hash(label)) % len(_INFER_COLORS)]


def _normalize_detection_bbox(bbox, img_w: int = 0, img_h: int = 0):
    """Normalize a detection bbox to [x1, y1, x2, y2] in [0,1] range.

    Accepts either:
      - normalized xyxy:  [x1, y1, x2, y2] (all in [0,1])
      - normalized xywh:  [x1, y1, w, h]   (all in [0,1])
      - absolute xywh:    [x1, y1, w, h]   (pixel values, requires img_w/img_h)
      - absolute xyxy:    [x1, y1, x2, y2] (pixel values, requires img_w/img_h)
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, a, b = [float(v) for v in bbox]
    # Heuristic: if either value > 1.5, treat all as absolute pixels
    absolute = max(abs(x1), abs(y1), abs(a), abs(b)) > 1.5
    if absolute:
        if not img_w or not img_h:
            # Can't normalize without image dims; return as-is and let client handle
            return [x1, y1, a, b]
        x1, a = x1 / img_w, a / img_w
        y1, b = y1 / img_h, b / img_h
    # Now in [0,1]. Detect xywh vs xyxy: in xywh, a >= x1 and b >= y1.
    if a >= x1 and b >= y1 and (a - x1) <= 1.5 and (b - y1) <= 1.5 and a <= 1.5 and b <= 1.5:
        return [max(0.0, min(1.0, x1)),
                max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, a)),
                max(0.0, min(1.0, b))]
    # Treat as xyxy
    return [min(x1, a), min(y1, b), max(x1, a), max(y1, b)]


def _normalize_inference_response(data: dict, training_type: str, model_name: str) -> dict:
    """Normalize external (Modal/Ray) responses to match the frontend
    Detection/SegMask/ClsResult interfaces. Adds label/color/count/model_name
    and converts xywh→xyxy bboxes when needed.
    """
    if not isinstance(data, dict):
        return data
    data["model_name"] = data.get("model_name") or model_name
    tt = data.get("type") or training_type

    if tt == "detection":
        dets_raw = data.get("detections") or []
        dets = []
        for d in dets_raw:
            if not isinstance(d, dict):
                continue
            label = d.get("label") or d.get("class_name") or d.get("class") or f"class_{len(dets)}"
            try:
                conf = float(d.get("confidence", d.get("score", 0.0)))
            except (TypeError, ValueError):
                conf = 0.0
            bbox = _normalize_detection_bbox(d.get("bbox") or d.get("box") or [])
            dets.append({
                "label":      str(label),
                "confidence": conf,
                "bbox":       bbox,
                "color":      d.get("color") or _color_for(str(label)),
            })
        data["detections"] = dets
        data["count"]      = len(dets)

    elif tt == "segmentation":
        # Accept either "masks" (with polygon) or "segments" (coverage only)
        masks_raw = data.get("masks")
        if masks_raw is None:
            segs = data.get("segments") or []
            masks = []
            for s in segs:
                if not isinstance(s, dict):
                    continue
                label = s.get("label") or s.get("class_name") or f"class_{len(masks)}"
                try:
                    cov = float(s.get("coverage", s.get("area_pct", 0.0)))
                except (TypeError, ValueError):
                    cov = 0.0
                try:
                    conf = float(s.get("confidence", s.get("score", cov)))
                except (TypeError, ValueError):
                    conf = cov
                masks.append({
                    "label":     str(label),
                    "confidence": conf,
                    "area_pct":  cov * 100.0 if cov <= 1.0 else cov,
                    "color":     s.get("color") or _color_for(str(label)),
                    "polygon":   s.get("polygon") or [],
                })
            data["masks"] = masks
        else:
            masks = []
            for m in masks_raw:
                if not isinstance(m, dict):
                    continue
                label = m.get("label") or m.get("class_name") or f"class_{len(masks)}"
                try:
                    conf = float(m.get("confidence", m.get("score", 0.0)))
                except (TypeError, ValueError):
                    conf = 0.0
                try:
                    area = float(m.get("area_pct", m.get("coverage", 0.0)))
                except (TypeError, ValueError):
                    area = 0.0
                masks.append({
                    "label":      str(label),
                    "confidence": conf,
                    "area_pct":   area,
                    "color":      m.get("color") or _color_for(str(label)),
                    "polygon":    m.get("polygon") or [],
                })
            data["masks"] = masks

    elif tt == "classification":
        preds_raw = data.get("predictions") or []
        preds = []
        for p in preds_raw:
            if not isinstance(p, dict):
                continue
            label = p.get("label") or p.get("class_name") or f"class_{len(preds)}"
            try:
                conf = float(p.get("confidence", p.get("score", 0.0)))
            except (TypeError, ValueError):
                conf = 0.0
            preds.append({"label": str(label), "confidence": conf})
        data["predictions"] = preds
        top_label = data.get("top_label")
        if not top_label and preds:
            top_label = preds[0]["label"]
            data["top_label"] = top_label
        if "top_confidence" not in data:
            if preds:
                data["top_confidence"] = preds[0]["confidence"]
            else:
                data["top_confidence"] = 0.0

    return data


@app.get("/api/inference/history")
async def list_inference_history(limit: int = 50, before: float = 0):
    """List inference history, newest first. Pass `before` (timestamp) to paginate."""
    limit = max(1, min(200, limit))
    with get_db() as conn:
        if before > 0:
            rows = conn.execute(
                "SELECT id, created_at, mode, model_id, model_name, model_type, "
                "image_name, thumbnail_key, image_key, user_prompt, system_prompt, result_json, "
                "inference_time_ms "
                "FROM inference_history WHERE created_at < ? ORDER BY created_at DESC LIMIT ?",
                (before, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, created_at, mode, model_id, model_name, model_type, "
                "image_name, thumbnail_key, image_key, user_prompt, system_prompt, result_json, "
                "inference_time_ms "
                "FROM inference_history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["result"] = json.loads(d.pop("result_json") or "{}")
        except Exception:
            d["result"] = {}
        out.append(d)
    return {"history": out}


@app.get("/api/inference/thumbnail/{history_id}")
async def get_inference_thumbnail(history_id: str):
    """Stream the thumbnail image bytes from MinIO (no auth, no presign — internal only)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT thumbnail_key FROM inference_history WHERE id = ?",
            (history_id,),
        ).fetchone()
    if not row or not row["thumbnail_key"]:
        raise HTTPException(404, "Thumbnail not found")
    key = row["thumbnail_key"]
    try:
        import boto3 as _boto3
        from botocore.config import Config as _BConfig
        from fastapi.responses import Response
        s3 = _boto3.client(
            "s3", endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_BConfig(signature_version="s3v4"),
        )
        obj = s3.get_object(Bucket="medimage-thumbnails", Key=key)
        data = obj["Body"].read()
        return Response(content=data, media_type="image/jpeg",
                      headers={"Cache-Control": "private, max-age=3600"})
    except HTTPException:
        raise
    except Exception as _e:
        raise HTTPException(502, f"Could not fetch thumbnail: {_e}")


@app.get("/api/inference/image/{history_id}")
async def get_inference_image(history_id: str):
    """Stream the ORIGINAL (full-res) image bytes from MinIO."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT image_key, image_name FROM inference_history WHERE id = ?",
            (history_id,),
        ).fetchone()
    if not row or not row["image_key"]:
        raise HTTPException(404, "Image not found")
    key = row["image_key"]
    try:
        import boto3 as _boto3
        from botocore.config import Config as _BConfig
        from fastapi.responses import Response
        s3 = _boto3.client(
            "s3", endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_BConfig(signature_version="s3v4"),
        )
        obj = s3.get_object(Bucket="medimage-thumbnails", Key=key)
        data = obj["Body"].read()
        # Detect content type from extension
        ext = key.rsplit(".", 1)[-1].lower() if "." in key else "jpg"
        ct = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
              "webp": "image/webp", "bmp": "image/bmp", "gif": "image/gif",
              "tif": "image/tiff", "tiff": "image/tiff", "dcm": "application/dicom"}.get(ext, "image/jpeg")
        return Response(content=data, media_type=ct,
                      headers={"Cache-Control": "private, max-age=3600"})
    except HTTPException:
        raise
    except Exception as _e:
        raise HTTPException(502, f"Could not fetch image: {_e}")


@app.delete("/api/inference/history/{history_id}")
async def delete_inference_history(history_id: str):
    """Delete one history entry + its thumbnail."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT thumbnail_key FROM inference_history WHERE id = ?",
            (history_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Not found")
        conn.execute("DELETE FROM inference_history WHERE id = ?", (history_id,))
        conn.commit()
    if row["thumbnail_key"]:
        try:
            import boto3 as _boto3
            from botocore.config import Config as _BConfig
            s3 = _boto3.client(
                "s3", endpoint_url=MINIO_URL,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                config=_BConfig(signature_version="s3v4"),
            )
            s3.delete_object(Bucket="medimage-thumbnails", Key=row["thumbnail_key"])
        except Exception:
            pass
    return {"ok": True}


@app.delete("/api/inference/history")
async def clear_inference_history():
    """Delete ALL history entries + their thumbnails."""
    with get_db() as conn:
        rows = conn.execute("SELECT thumbnail_key FROM inference_history").fetchall()
        conn.execute("DELETE FROM inference_history")
        conn.commit()
    if rows:
        try:
            import boto3 as _boto3
            from botocore.config import Config as _BConfig
            s3 = _boto3.client(
                "s3", endpoint_url=MINIO_URL,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                config=_BConfig(signature_version="s3v4"),
            )
            keys = [{"Key": r["thumbnail_key"]} for r in rows if r["thumbnail_key"]]
            if keys:
                for i in range(0, len(keys), 1000):
                    s3.delete_objects(Bucket="medimage-thumbnails", Delete={"Objects": keys[i:i+1000]})
        except Exception:
            pass
    return {"ok": True}


@app.get("/api/internal/weights-url")
async def get_presigned_weights_url(bucket: str, key: str, expires: int = 600):
    """Generate a presigned S3 URL for a weights object. Used by Ray actors
    to download model weights via plain HTTP (avoids boto3 incompatibilities
    on heterogeneous worker nodes)."""
    try:
        import boto3 as _boto3
        from botocore.config import Config as _BConfig
        s3 = _boto3.client(
            "s3", endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_BConfig(signature_version="s3v4"),
        )
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires,
        )
        return {"url": url}
    except Exception as _e:
        raise HTTPException(502, f"Could not generate presigned URL: {_e}")


@app.post("/api/inference/history")
async def save_inference_history(req: Request):
    """Save a new history entry. `thumbnail_base64` (optional) is decoded and uploaded to MinIO."""
    import base64 as _b64, time as _time, uuid as _uuid
    body = await req.json()
    hist_id = body.get("id") or str(_uuid.uuid4())
    user_id = body.get("user_id", "default")
    mode = body.get("mode", "text")
    model_id = body.get("model_id", "")
    model_name = body.get("model_name", "")
    model_type = body.get("model_type", "")
    image_name = body.get("image_name", "")
    user_prompt = body.get("user_prompt", "")
    system_prompt = body.get("system_prompt", "")
    result = body.get("result", {})
    inference_time_ms = body.get("inference_time_ms", 0)
    thumb_b64 = body.get("thumbnail_base64", "")
    image_b64_input = body.get("image_base64", "")
    thumbnail_key = ""
    image_key = ""
    try:
        import boto3 as _boto3
        from botocore.config import Config as _BConfig
        s3 = _boto3.client(
            "s3", endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=_BConfig(signature_version="s3v4"),
        )
        try:
            s3.head_bucket(Bucket="medimage-thumbnails")
        except Exception:
            s3.create_bucket(Bucket="medimage-thumbnails")
        # Thumbnail (small, used in history list)
        if thumb_b64:
            if "," in thumb_b64:
                thumb_b64 = thumb_b64.split(",", 1)[1]
            img_bytes = _b64.b64decode(thumb_b64)
            if len(img_bytes) > 0:
                thumbnail_key = f"{hist_id}_thumb.jpg"
                s3.put_object(
                    Bucket="medimage-thumbnails", Key=thumbnail_key,
                    Body=img_bytes, ContentType="image/jpeg",
                )
        # Original image (full resolution, used to restore the drop box)
        if image_b64_input:
            if "," in image_b64_input:
                image_b64_input = image_b64_input.split(",", 1)[1]
            orig_bytes = _b64.b64decode(image_b64_input)
            if len(orig_bytes) > 0:
                ext = (image_name or "").split(".")[-1] if "." in (image_name or "") else "jpg"
                # Cap at 25 MB to avoid quota bloat
                if len(orig_bytes) <= 25 * 1024 * 1024:
                    image_key = f"{hist_id}_orig.{ext}"
                    s3.put_object(
                        Bucket="medimage-thumbnails", Key=image_key,
                        Body=orig_bytes,
                        ContentType="image/jpeg" if ext == "jpg" else f"image/{ext}",
                    )
    except Exception as _e:
        print(f"[history] upload failed: {_e}", flush=True)
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO inference_history "
            "(id, user_id, created_at, mode, model_id, model_name, model_type, "
            " image_name, thumbnail_key, image_key, user_prompt, system_prompt, result_json, "
            " inference_time_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (hist_id, user_id, _time.time(), mode, model_id, model_name, model_type,
             image_name, thumbnail_key, image_key, user_prompt, system_prompt,
             json.dumps(result), inference_time_ms),
        )
        conn.commit()
    return {"id": hist_id, "thumbnail_key": thumbnail_key, "image_key": image_key}


@app.post("/api/inference")
async def run_inference(
    model_id: str = FForm(...),
    image: Optional[UploadFile] = File(None),
    prompt: str = FForm(""),
    system_prompt: str = FForm(""),
    conf_threshold: float = FForm(0.5),
):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND status = 'completed'", (model_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found or not completed")

    job = dict(row)
    training_type = job["training_type"]
    provider = job.get("inference_provider", "") or ""
    fname = (image.filename if image else None) or prompt[:20] or "upload"
    # seed RNG from filename for reproducible-ish demo results
    rng = random.Random(abs(hash(fname)) % (2**31))

    t_start = time.time()

    # ── Universal external routing (Modal or Ray Serve) ──────────────────────
    async def _route_ray_client(serve_url: str, job_id_: str) -> dict:
        """Call model Ray Actor via Ray Jobs API (avoids Ray Client port issues)."""
        import requests as _req, json as _json, logging as _logging, asyncio as _asyncio
        import base64 as _b64, tempfile, os as _os
        _log = _logging.getLogger("ray_infer")
        _dashboard = "http://100.68.53.118:8265"
        _actor_name = f"model-{job_id_}"
        _log.warning(f"[infer] Calling actor {_actor_name} via Ray Jobs API...")

        _img_bytes = None
        if image:
            _img_bytes = await image.read()

        _img_b64 = _b64.b64encode(_img_bytes).decode() if _img_bytes else ""
        _infer_timeout = (
            600 if training_type in ("llm-text", "vlm-finetune")
            else 600 if training_type == "segmentation"  # Mask R-CNN on CPU can be slow
            else 120
        )

        _script = textwrap.dedent("""\
            import ray, json, sys, base64, time, os
            ray.init(address="auto", ignore_reinit_error=True)
            actor_name = os.environ.get("MEDIMAGE_ACTOR_NAME", "")
            img_b64 = os.environ.get("MEDIMAGE_IMG_B64", "")
            prompt = os.environ.get("MEDIMAGE_PROMPT", "")
            sys_prompt = os.environ.get("MEDIMAGE_SYSTEM_PROMPT", "")
            conf = float(os.environ.get("MEDIMAGE_CONF", "0.5"))
            infer_timeout = int(os.environ.get("MEDIMAGE_INFER_TIMEOUT", "300"))
            try:
                actor = ray.get_actor(actor_name, namespace="default")
            except Exception as e:
                print(json.dumps({"error": f"Actor not found: {e}"}))
                sys.exit(1)
            for _i in range(120):
                try:
                    h = ray.get(actor.health.remote(), timeout=30)
                    if h.get("ok") and h.get("model_loaded"):
                        break
                    if h.get("load_error"):
                        print(json.dumps({"error": f"Model load failed: {h['load_error']}"}))
                        sys.exit(1)
                except Exception:
                    time.sleep(5)
            else:
                print(json.dumps({"error": "Actor not ready after 600s"}))
                sys.exit(1)
            result = ray.get(actor.infer.remote(
                img_b64, conf, prompt, sys_prompt
            ), timeout=infer_timeout)
            print("INFERENCE_RESULT:" + json.dumps(result))
        """)
        _script_b64 = _b64.b64encode(_script.encode()).decode()
        # Send image + prompt via env vars (no argv limit)
        _infer_env = {
            "MEDIMAGE_IMG_B64": _img_b64,
            "MEDIMAGE_PROMPT": prompt,
            "MEDIMAGE_SYSTEM_PROMPT": system_prompt,
            "MEDIMAGE_CONF": str(conf_threshold),
            "MEDIMAGE_INFER_TIMEOUT": str(_infer_timeout),
            "MEDIMAGE_ACTOR_NAME": _actor_name,
        }
        _payload = {
            "entrypoint": "bash -c 'echo $MEDIMAGE_INFER_SCRIPT_B64 | base64 -d > /tmp/infer_runner.py && python3 /tmp/infer_runner.py'",
            "runtime_env": {"env_vars": {**_infer_env, "MEDIMAGE_INFER_SCRIPT_B64": _script_b64}},
        }
        try:
            _resp = _req.post(f"{_dashboard}/api/jobs/", json=_payload, timeout=15)
            _resp.raise_for_status()
            _submit_id = _resp.json().get("submission_id") or _resp.json().get("job_id")
            _log.warning(f"[infer] Job submitted: {_submit_id}")
        except Exception as _e:
            _log.error(f"[infer] Job submit failed: {_e}")
            raise RuntimeError(f"Failed to submit inference job: {_e}")

        _deadline = time.time() + _infer_timeout + 60
        while time.time() < _deadline:
            try:
                _st = _req.get(f"{_dashboard}/api/jobs/{_submit_id}", timeout=10).json()
                _status = _st.get("status", "")
            except Exception:
                _status = "UNKNOWN"
            if _status in ("SUCCEEDED", "FAILED", "STOPPED"):
                break
            await _asyncio.sleep(3)
        else:
            raise RuntimeError(f"Inference job timed out after {_infer_timeout + 60}s")

        try:
            _logs_resp = _req.get(f"{_dashboard}/api/jobs/{_submit_id}/logs", timeout=15)
            _logs = _logs_resp.json().get("logs", "") if _logs_resp.ok else ""
        except Exception:
            _logs = ""

        if _status != "SUCCEEDED":
            _tail = _logs[-500:] if _logs else "no logs"
            _log.error(f"[infer] Job {_submit_id} status={_status}: {_tail}")
            raise RuntimeError(f"Inference failed ({_status}): {_tail}")

        for _line in _logs.splitlines():
            if _line.startswith("INFERENCE_RESULT:"):
                _result = _json.loads(_line[len("INFERENCE_RESULT:"):])
                _log.warning(f"[infer] Got result: {str(_result)[:200]}")
                if isinstance(_result, dict) and _result.get("error"):
                    raise RuntimeError(_result["error"])
                return _result

        _log.error(f"[infer] No INFERENCE_RESULT in logs: {_logs[-500:]}")
        raise RuntimeError(f"Inference completed but no result found in output")

    async def _route_external(base_url: str, api_key: str) -> dict:
        # Modal fastapi_endpoint URLs are the endpoint itself (no path needed)
        # Ray Serve / other URLs need /inference appended
        if "modal.run" in base_url:
            endpoint = base_url.rstrip("/")
        else:
            endpoint = base_url.rstrip("/") + "/inference"
        headers_ext: dict = {}
        if api_key:
            headers_ext["Authorization"] = f"Bearer {api_key}"
        is_text = training_type in ("llm-text", "vlm-finetune")
        if is_text:
            headers_ext["Content-Type"] = "application/json"
            payload: dict = {"prompt": prompt, "system_prompt": system_prompt, "training_type": training_type}
            if image:
                img_bytes = await image.read()
                payload["image"] = base64.b64encode(img_bytes).decode()
            async with httpx.AsyncClient(timeout=570.0, follow_redirects=True) as client:
                r = await client.post(endpoint, json=payload, headers=headers_ext)
        else:
            if not image:
                raise HTTPException(status_code=400, detail="Image required for this model type")
            img_bytes = await image.read()
            files = {"image": (image.filename or "image.jpg", img_bytes, "image/jpeg")}
            data = {"model_id": model_id, "conf_threshold": str(conf_threshold), "training_type": training_type}
            async with httpx.AsyncClient(timeout=570.0, follow_redirects=True) as client:
                r = await client.post(endpoint, headers=headers_ext, data=data, files=files)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data["inference_time_ms"] = round((time.time() - t_start) * 1000)
            data = _normalize_inference_response(data, training_type, job.get("model_name") or job.get("name") or model_id)
        return data

    if provider == "modal" and job.get("modal_url"):
        try:
            return await _route_external(job["modal_url"], job.get("modal_api_key") or "")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Modal endpoint error: {e}")

    if provider == "ray" and job.get("ray_serve_url"):
        try:
            return await _route_ray_client(job["ray_serve_url"], model_id)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Ray Serve endpoint error: {e}")

    # ── Global default fallback (per-model provider not set) ─────────────────
    if not provider:
        with get_db() as _gc:
            _grows = _gc.execute(
                "SELECT key, value FROM settings WHERE key IN "
                "('global_inference_provider','global_modal_url','global_modal_api_key','global_ray_serve_url')"
            ).fetchall()
        _gs = {r["key"]: r["value"] for r in _grows}
        _gprov = _gs.get("global_inference_provider", "")
        if _gprov == "modal" and _gs.get("global_modal_url"):
            try:
                return await _route_external(_gs["global_modal_url"], _gs.get("global_modal_api_key") or "")
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Modal endpoint error: {e}")
        if _gprov == "ray" and _gs.get("global_ray_serve_url"):
            try:
                return await _route_ray_client(_gs["global_ray_serve_url"], model_id)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Ray Serve endpoint error: {e}")

    raise HTTPException(
        status_code=400,
        detail="No inference provider configured for this model. Set a Ray Serve or Modal URL in the Models page.",
    )


# ─── Jupyter management ──────────────────────────────────────────────────────

_JUPYTER_CONTAINER = os.environ.get("JUPYTER_CONTAINER", "medimage-jupyter-1")


def _docker_request(method: str, path: str, body: bytes | None = None, timeout: int = 30) -> tuple[int, bytes]:
    """Single HTTP request to Docker socket (Connection: close)."""
    import socket as _socket
    sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect("/var/run/docker.sock")
    hdrs = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: localhost\r\n"
        f"Connection: close\r\n"
        f"Content-Type: application/json\r\n"
    )
    if body:
        hdrs += f"Content-Length: {len(body)}\r\n"
    hdrs += "\r\n"
    sock.sendall(hdrs.encode() + (body or b""))
    data = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
    sock.close()
    status_code = int(data.split(b" ")[1])
    body_start = data.find(b"\r\n\r\n") + 4
    return status_code, data[body_start:]


async def _docker_exec(container: str, cmd: list[str]) -> tuple[int, str]:
    """Run a command inside a container via Docker socket (no docker binary needed)."""
    import json as _json

    def _sync():
        # Step 1: Create exec instance
        payload = _json.dumps({"AttachStdout": True, "AttachStderr": True, "Cmd": cmd}).encode()
        sc, body = _docker_request("POST", f"/containers/{container}/exec", payload)
        if sc not in (200, 201):
            return sc, body.decode(errors="replace")
        exec_id = _json.loads(body)["Id"]

        # Step 2: Start exec — Docker closes connection when exec completes
        start_payload = _json.dumps({"Detach": False, "Tty": False}).encode()
        _, raw = _docker_request("POST", f"/exec/{exec_id}/start", start_payload, timeout=300)
        # Strip Docker stream multiplexing frames (8-byte header per frame)
        result = b""
        i = 0
        while i + 8 <= len(raw):
            frame_size = int.from_bytes(raw[i + 4:i + 8], "big")
            result += raw[i + 8:i + 8 + frame_size]
            i += 8 + frame_size
        if not result:
            result = raw

        # Step 3: Get exit code
        try:
            _, inspect = _docker_request("GET", f"/exec/{exec_id}/json", timeout=10)
            exit_code = _json.loads(inspect).get("ExitCode", 0)
        except Exception:
            exit_code = 0
        return exit_code, result.decode(errors="replace")

    return await asyncio.to_thread(_sync)


async def _docker_restart(container: str) -> bool:
    """Restart a container via Docker socket."""
    def _sync():
        sc, _ = _docker_request("POST", f"/containers/{container}/restart", timeout=60)
        return sc in (204, 200)
    return await asyncio.to_thread(_sync)


@app.get("/api/jupyter/version")
async def jupyter_version():
    """Return current and latest available JupyterLab version."""
    try:
        _, out = await _docker_exec(_JUPYTER_CONTAINER, ["pip", "show", "jupyterlab"])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Jupyter container: {e}")

    current = next(
        (ln.split(":", 1)[1].strip() for ln in out.splitlines() if ln.lower().startswith("version")),
        "unknown",
    )

    latest = "unknown"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get("https://pypi.org/pypi/jupyterlab/json")
            latest = resp.json()["info"]["version"]
    except Exception:
        pass

    return {"current": current, "latest": latest, "update_available": current != latest and latest != "unknown"}


_jupyter_update_log: list[str] = []
_jupyter_update_running = False


@app.post("/api/jupyter/update")
async def jupyter_update():
    """Upgrade JupyterLab inside the running container."""
    global _jupyter_update_running
    if _jupyter_update_running:
        return {"status": "already_running"}

    async def _do_update():
        global _jupyter_update_running, _jupyter_update_log
        _jupyter_update_running = True
        _jupyter_update_log = []
        try:
            _jupyter_update_log.append("$ pip install --upgrade jupyterlab")
            rc, out = await _docker_exec(
                _JUPYTER_CONTAINER,
                ["pip", "install", "--upgrade", "jupyterlab"],
            )
            _jupyter_update_log.extend(line for line in out.splitlines() if line.strip())
            if rc != 0:
                _jupyter_update_log.append(f"ERROR: pip exited {rc}")
                return
            _jupyter_update_log.append(f"$ restart {_JUPYTER_CONTAINER}")
            ok = await _docker_restart(_JUPYTER_CONTAINER)
            _jupyter_update_log.append("✓ JupyterLab updated and restarted" if ok else "ERROR: restart failed")
        except Exception as e:
            _jupyter_update_log.append(f"ERROR: {e}")
        finally:
            _jupyter_update_running = False

    asyncio.create_task(_do_update())
    return {"status": "started"}


@app.get("/api/jupyter/update-status")
def jupyter_update_status():
    return {"running": _jupyter_update_running, "log": _jupyter_update_log}


# ─── Auto-label endpoint ──────────────────────────────────────────────────────
class AutoLabelRequest(BaseModel):
    project_id: int
    model: str   # e.g. "medsam_vit_b", "yolov8s.pt", "facebook/detr-resnet-50"
    engine: str  # "MedSAM", "Ultralytics", "HuggingFace"
    task_ids: list[int] = []  # empty = all unlabeled tasks


@app.post("/api/autolabel/{project_id}")
async def auto_label(project_id: int, req: AutoLabelRequest, request: Request):
    """Run a pre-trained model to generate pre-annotations in Label Studio."""
    import asyncio as _aio
    task_id_str = f"autolabel-{project_id}-{req.engine}"

    async def _run():
        import requests as _req
        import base64 as _b64
        import io as _io
        import re as _re
        ls_internal = LS_API_URL.rstrip("/")
        headers = {"Authorization": f"Token {LS_TOKEN}", "Content-Type": "application/json"}

        # 1. Fetch tasks from LS
        params = {"project": project_id, "page_size": 500}
        r = _req.get(f"{ls_internal}/api/tasks/", headers=headers, params=params, timeout=30)
        r.raise_for_status()
        tasks = r.json().get("tasks", r.json()) if isinstance(r.json(), dict) else r.json()

        if req.task_ids:
            tasks = [t for t in tasks if t["id"] in req.task_ids]

        # Filter: only tasks with no annotations yet
        unlabeled = [t for t in tasks if not t.get("annotations")]
        if not unlabeled:
            return {"status": "done", "labeled": 0, "message": "No unlabeled tasks found"}

        print(f"[autolabel] {len(unlabeled)} unlabeled tasks, engine={req.engine}, model={req.model}")

        labeled = 0
        for task in unlabeled:
            try:
                image_url = task.get("data", {}).get("image", "")
                if not image_url:
                    continue

                # Download image
                if image_url.startswith("data:"):
                    _, data = image_url.split(",", 1)
                    img_bytes = _b64.b64decode(data)
                else:
                    url = _re.sub(r"https?://[^/]+", ls_internal, image_url)
                    resp = _req.get(url, headers={"Authorization": f"Token {LS_TOKEN}"}, timeout=30)
                    resp.raise_for_status()
                    img_bytes = resp.content

                from PIL import Image as _Img
                pil_img = _Img.open(_io.BytesIO(img_bytes)).convert("RGB")
                w, h = pil_img.size

                predictions = []

                if req.engine == "Ultralytics":
                    from ultralytics import YOLO as _YOLO
                    import numpy as _np, tempfile as _tmp2
                    mdl = _YOLO(req.model)
                    tmp_f = _tmp2.NamedTemporaryFile(suffix=".jpg", delete=False)
                    pil_img.save(tmp_f.name)
                    results = mdl(tmp_f.name, verbose=False)
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls_name = results[0].names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        predictions.append({
                            "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                            "value": {
                                "x": x1 / w * 100, "y": y1 / h * 100,
                                "width": (x2 - x1) / w * 100, "height": (y2 - y1) / h * 100,
                                "rectanglelabels": [cls_name],
                            },
                            "score": conf,
                        })

                elif req.engine == "MedSAM":
                    # Use center-box prompt for whole-image segmentation
                    import torch as _torch, numpy as _np, tempfile as _tmp2
                    try:
                        from segment_anything import sam_model_registry as _reg
                    except ImportError:
                        import subprocess, sys
                        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segment-anything"], capture_output=True)
                        from segment_anything import sam_model_registry as _reg
                    # Download checkpoint if needed
                    ckpt_path = "/tmp/medsam_vit_b.pth"
                    if not os.path.exists(ckpt_path):
                        import urllib.request
                        urllib.request.urlretrieve(
                            "https://huggingface.co/bowang/medsam/resolve/main/medsam_vit_b.pth",
                            ckpt_path,
                        )
                    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
                    sam = _reg["vit_b"](checkpoint=ckpt_path)
                    sam.to(device).eval()
                    from segment_anything import SamPredictor as _Pred
                    predictor = _Pred(sam)
                    predictor.set_image(_np.array(pil_img))
                    # Prompt: full-image bounding box
                    masks, _, _ = predictor.predict(
                        box=_np.array([[0, 0, w, h]]),
                        multimask_output=False,
                    )
                    mask = masks[0]
                    # Convert to RLE for LS brush annotation
                    from pycocotools import mask as _mask_util
                    import base64 as _b64enc
                    rle = _mask_util.encode(_np.asfortranarray(mask.astype(_np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    predictions.append({
                        "from_name": "tag", "to_name": "image", "type": "brushlabels",
                        "value": {
                            "format": "rle",
                            "rle": rle["counts"],
                            "brushlabels": ["foreground"],
                        },
                        "score": 0.9,
                    })

                elif req.engine == "HuggingFace":
                    from transformers import pipeline as _pipe
                    detector = _pipe("object-detection", model=req.model)
                    import tempfile as _tmp2
                    tmp_f = _tmp2.NamedTemporaryFile(suffix=".jpg", delete=False)
                    pil_img.save(tmp_f.name)
                    results = detector(tmp_f.name)
                    for det in results:
                        box = det["box"]
                        label = det["label"]
                        score = det["score"]
                        predictions.append({
                            "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                            "value": {
                                "x": box["xmin"] / w * 100, "y": box["ymin"] / h * 100,
                                "width": (box["xmax"] - box["xmin"]) / w * 100,
                                "height": (box["ymax"] - box["ymin"]) / h * 100,
                                "rectanglelabels": [label],
                            },
                            "score": score,
                        })

                if predictions:
                    payload = {
                        "task": task["id"],
                        "result": predictions,
                        "model_version": req.model,
                    }
                    pr = _req.post(f"{ls_internal}/api/predictions/",
                                   json=payload, headers=headers, timeout=30)
                    pr.raise_for_status()
                    labeled += 1
                    print(f"[autolabel] Task {task['id']} → {len(predictions)} predictions")

            except Exception as e:
                print(f"[autolabel] Task {task.get('id', '?')} failed: {e}")
                continue

        return {"status": "done", "labeled": labeled, "total": len(unlabeled)}

    result = await asyncio.to_thread(lambda: asyncio.run(_run()))
    return result
