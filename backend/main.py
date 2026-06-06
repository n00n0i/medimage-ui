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
import asyncio
import base64
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form as FForm, Request, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import io
import zipfile
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
_KC_PUBLIC_PATHS = {"/api/health"}
_KC_PUBLIC_PREFIXES = ("/api/ls-goto/", "/api/jupyter/")

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
        await asyncio.to_thread(_sync_verify_token, token)
    except Exception as exc:
        return JSONResponse({"detail": f"Invalid token: {exc}"}, status_code=401)
    return await call_next(request)


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


# ─── Training runner ─────────────────────────────────────────────────────────

LS_API_URL = os.getenv("LS_API_URL", "http://label-studio:8080")
LS_TOKEN   = os.getenv("LS_TOKEN", "medimage-ls-token-2026")
MINIO_URL  = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
# Public URL for MinIO accessible from Ray/Modal clusters (set to external/VPN IP)
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", MINIO_URL)
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

# ─── Real Training Helpers ────────────────────────────────────────────────────

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


def prepare_yolo_dataset_from_json(json_data, out_dir, ls_url, ls_token, task="detect"):
    """Convert Label Studio JSON export → YOLO detection or segmentation format.
    task='detect'  → cx cy w h (normalised)
    task='segment' → polygon points (4-corner rect from bbox, or actual polygonlabels)
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
            if image_url.startswith("data:"):
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
    train_dl = DataLoader(ImgDS(samples[:-n_val], tf_tr), batch_size=batch, shuffle=True)
    val_dl   = DataLoader(ImgDS(samples[-n_val:], tf_vl), batch_size=batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TIMM uses underscores: efficientnet-b2 → efficientnet_b2
    timm_name = model_name.replace("-", "_")
    print(f"[train] Device: {device} | model: {timm_name}")
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


def train_unsloth_llm(model_name, text_dataset, max_seq_len, lora_rank, quantization,
                       epochs, batch, grad_accum, lr, chat_template,
                       job_id, w_bucket, w_key, minio_url, minio_access, minio_secret):
    """Fine-tune an LLM with Unsloth + TRL SFTTrainer. Saves LoRA adapter to MinIO."""
    import zipfile as _zf, tempfile as _tmp, shutil as _sh
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset, Dataset as HFDataset
    import torch as _torch

    load_in_4bit = (quantization in ("4bit", "4-bit", "bnb-4bit"))
    print(f"[train] Loading {model_name}  4bit={load_in_4bit}  lora_r={lora_rank}  seq={max_seq_len}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_len,
        dtype=None, load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    # Load dataset  (HuggingFace dataset id or JSONL string)
    print(f"[train] Loading text dataset: {text_dataset}")
    try:
        ds = load_dataset(text_dataset, split="train")
    except Exception as e:
        print(f"[warn] HF load failed ({e}), treating as JSONL string / inline data")
        import json as _json
        rows = [_json.loads(l) for l in text_dataset.splitlines() if l.strip()]
        ds = HFDataset.from_list(rows)

    # Build conversation field
    text_field = None
    for candidate in ("text", "conversations", "messages", "instruction"):
        if candidate in ds.column_names:
            text_field = candidate
            break
    if text_field is None:
        raise RuntimeError(f"Dataset has no recognized text column. Columns: {ds.column_names}")
    print(f"[train] Dataset: {len(ds)} rows, text_field='{text_field}'")

    # Format conversations if needed
    if text_field in ("conversations", "messages"):
        def fmt_conv(ex):
            parts = ex[text_field]
            out = ""
            for turn in (parts if isinstance(parts, list) else []):
                role = turn.get("from", turn.get("role", "user"))
                content = turn.get("value", turn.get("content", ""))
                if role in ("human", "user"):
                    out += f"<|user|>\n{content}\n"
                else:
                    out += f"<|assistant|>\n{content}\n"
            return {"text": out}
        ds = ds.map(fmt_conv)
        text_field = "text"

    save_dir = f"/tmp/lora_{job_id}"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field=text_field,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=5,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=not _torch.cuda.is_bf16_supported(),
            bf16=_torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=save_dir,
            save_strategy="no",
        ),
    )
    trainer.train()
    print("[train] LLM training complete, saving LoRA adapter ...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    # Zip the adapter and upload
    zip_path = f"/tmp/lora_{job_id}.zip"
    with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED) as zf:
        for fp in Path(save_dir).rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(save_dir))
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)


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

    processor = AutoImageProcessor.from_pretrained(model_name)

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
                # DETR expects center format [cx,cy,w,h] normalised — matches YOLO
                annotations = {"image_id": idx, "annotations": [
                    {"bbox": b, "category_id": l, "area": b[2]*b[3], "iscrowd": 0}
                    for b, l in zip(boxes, labels)
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
            _cfg_type = type(AutoConfig.from_pretrained(model_name)).__name__.lower()
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
            ignore_mismatched_sizes=True
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
            ignore_mismatched_sizes=True
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
            batch_enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch_enc.items()}
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
                batch_enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch_enc.items()}
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
    print(f"WEIGHTS_UPLOADED:{w_key}")


def train_vlm_finetune(model_name, text_dataset, max_seq_len, lora_rank, quantization,
                        epochs, batch, grad_accum, lr, chat_template, job_id,
                        w_bucket, w_key, minio_url, minio_access, minio_secret,
                        img_dir=None):
    """Fine-tune a Vision Language Model (LLaVA-style) using HuggingFace transformers + PEFT LoRA."""
    import zipfile as _zf
    import torch
    from transformers import (
        AutoProcessor, AutoModelForVision2Seq,
        BitsAndBytesConfig, TrainingArguments, Trainer,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset, Dataset as HFDataset
    from PIL import Image as _Image
    import io as _io, base64 as _b64, glob as _glob

    print(f"[train] VLM fine-tune — model: {model_name}  lora_r={lora_rank}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quantization config
    load_in_4bit = quantization in ("4bit", "4-bit", "bnb-4bit")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_in_4bit else None

    print(f"[train] Loading processor and model ...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    vlm = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules="all-linear",
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    )
    vlm = get_peft_model(vlm, lora_cfg)
    vlm.print_trainable_parameters()

    # Load text/vision dataset
    print(f"[train] Loading dataset: {text_dataset}")
    try:
        ds = load_dataset(text_dataset, split="train")
    except Exception as e:
        print(f"[warn] HF load failed ({e}), trying as JSONL")
        import json as _json
        rows = [_json.loads(l) for l in text_dataset.splitlines() if l.strip()]
        ds = HFDataset.from_list(rows)

    # Collect local images if img_dir provided
    local_imgs = {}
    if img_dir and os.path.isdir(img_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in _glob.glob(os.path.join(img_dir, "**", ext), recursive=True):
                local_imgs[os.path.basename(p)] = p

    # Tokenize and format for VLM
    def preprocess(examples):
        texts, images = [], []
        for i in range(len(examples.get("text", examples.get("instruction", [""]*len(examples))))):
            # Build prompt
            instr_key = "text" if "text" in examples else "instruction"
            output_key = "output" if "output" in examples else "response"
            instruction = examples[instr_key][i] if instr_key in examples else ""
            output = examples[output_key][i] if output_key in examples else ""
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n{output}"
            texts.append(prompt)
            # Image: try to get from dataset or local
            img = None
            if "image" in examples and examples["image"][i] is not None:
                raw = examples["image"][i]
                if isinstance(raw, str) and raw.startswith("data:"):
                    _, data = raw.split(",", 1)
                    img = _Image.open(_io.BytesIO(_b64.b64decode(data))).convert("RGB")
                elif isinstance(raw, _Image.Image):
                    img = raw.convert("RGB")
            if img is None:
                img = _Image.new("RGB", (imgsz if 'imgsz' in dir() else 224,
                                          imgsz if 'imgsz' in dir() else 224), (128, 128, 128))
            images.append(img)
        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True, truncation=True, max_length=max_seq_len)
        enc["labels"] = enc["input_ids"].clone()
        return enc

    imgsz = 224  # default VLM image size
    print(f"[train] Preprocessing {len(ds)} examples ...")
    ds = ds.map(preprocess, batched=True, batch_size=4, remove_columns=ds.column_names)
    ds.set_format("torch")

    save_dir = f"/tmp/vlm_{job_id}"
    train_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=False, bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=vlm, args=train_args, train_dataset=ds)
    trainer.train()
    print("[train] VLM training complete. Saving LoRA adapter ...")
    vlm.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    zip_path = f"/tmp/vlm_{job_id}.zip"
    with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED) as zf:
        for fp in Path(save_dir).rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(save_dir))
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)
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
    model = YOLO(model_name)
    print(f"[train] Training started ...")
    results = model.train(
        data=yaml_path,
        epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr,
        optimizer=optimizer, project="/tmp/yolo_runs", name=job_id,
        exist_ok=True, verbose=True,
    )
    print(f"[train] Training done. Save dir: {results.save_dir}")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(results.save_dir) / "weights" / "last.pt"
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
    # 3-LLM. Unsloth LLM fine-tuning
    if engine == "Unsloth":
        if not text_dataset:
            raise RuntimeError("TEXT_DATASET env var is required for LLM fine-tuning")
        train_unsloth_llm(
            model_name, text_dataset, max_seq_len, lora_rank, quantization,
            epochs, batch, grad_accum, lr, chat_template,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )
        print("[train] Complete!")
        return

    # 3-VLM. HuggingFace VLM fine-tuning (LLaVA-style)
    if training_type == "vlm-finetune":
        train_vlm_finetune(
            model_name, text_dataset, max_seq_len, lora_rank, quantization,
            epochs, batch, grad_accum, lr, chat_template,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )
        print("[train] Complete!")
        return

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
    r = requests.get(dataset_url, stream=True, timeout=600)
    r.raise_for_status()
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

        # Try YOLO export first, fall back to JSON
        export_url = f"{LS_API_URL}/api/projects/{project_id}/export"
        fmt_order = [preferred_fmt] + [f for f in ("YOLO", "JSON") if f != preferred_fmt]
        for fmt in fmt_order:
            resp = _req.get(
                export_url,
                params={"exportType": fmt},
                headers={"Authorization": f"Token {LS_TOKEN}"},
                timeout=300,
                stream=True,
            )
            if resp.status_code == 200:
                _append_log(job_id, f"[export] Exported as {fmt} ({len(resp.content)//1024} KB)")
                export_data = resp.content
                break
        else:
            raise RuntimeError(
                f"Label Studio export failed for project {project_id} "
                f"(last HTTP status: {resp.status_code}) — {resp.text[:200]}"
            )

        bucket = "medimage-datasets"
        key    = f"{job_id}/dataset.zip"

        # If JSON export, wrap into a zip
        if fmt == "JSON":
            import io, zipfile as _zipfile
            buf = io.BytesIO()
            with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("annotations.json", export_data)
            export_data = buf.getvalue()
            _append_log(job_id, "[export] Wrapped JSON export into zip")

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

    # Export dataset from Label Studio (skip for LLM/VLM — uses HF text_dataset instead)
    engine = job.get("engine", "Ultralytics")
    training_type_for_export = job.get("training_type", "detection")
    _is_llm = engine == "Unsloth" or training_type_for_export in ("llm-text", "vlm-finetune")
    if _is_llm:
        dataset_url = ""
        _append_log(job_id, "[cluster] LLM engine — skipping LS export (uses text_dataset)")
    else:
        _set_step(job_id, "export")
        _append_log(job_id, f"[cluster] Exporting dataset from LS project {project_id} ...")
        # SMP segmentation and self-supervised need image data from JSON export
        # HuggingFace vision also needs JSON so we can download images from MinIO
        _needs_json = engine in ("Segmentation Models PyTorch", "HuggingFace") or training_type_for_export == "self-supervised"
        _export_fmt = "JSON" if _needs_json else "YOLO"
        dataset_url = _export_ls_to_minio(project_id, job_id, preferred_fmt=_export_fmt)
        _append_log(job_id, "[cluster] Dataset exported and staged in MinIO ✓")

    weights_bucket = "medimage-weights"
    weights_key    = f"{job_id}/best.pt"
    script_b64     = base64.b64encode(_VISION_TRAIN_SCRIPT.encode()).decode()
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
        "HF_TOKEN":        os.getenv("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
        # LLM-specific
        "TEXT_DATASET":    job.get("text_dataset", ""),
        "LORA_RANK":       str(job.get("lora_rank", 16)),
        "QUANTIZATION":    job.get("quantization", "4bit"),
        "MAX_SEQ_LEN":     str(job.get("max_seq_len", 2048)),
        "CHAT_TEMPLATE":   job.get("chat_template", "chatml"),
        "GRAD_ACCUM":      str(job.get("grad_accum", 4)),
    }

    # pip packages depend on engine
    if engine in ("PyTorch", "PyTorch+TIMM", "TIMM"):
        pip_pkgs = ["timm>=0.9", "boto3>=1.34", "requests", "Pillow"]
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
    elif engine == "Unsloth":
        pip_pkgs = ["unsloth", "trl>=0.8", "datasets>=2.18", "boto3>=1.34", "peft"]
    elif engine == "HuggingFace" or training_type == "vlm-finetune":
        pip_pkgs = ["transformers>=4.40", "peft>=0.10", "accelerate>=0.28",
                    "bitsandbytes>=0.43", "datasets>=2.18", "boto3>=1.34", "Pillow"]
    elif training_type == "self-supervised":
        pip_pkgs = ["boto3>=1.34", "requests", "Pillow"]  # torch/torchvision pre-installed
    else:
        # ultralytics is pre-installed on Ray cluster — do NOT reinstall (causes numpy binary conflict)
        pip_pkgs = ["boto3>=1.34", "pyyaml", "requests"]

    # Define the worker function inline so cloudpickle serializes the full body
    # (not a reference to 'main' module which doesn't exist on the Ray worker).
    # pip packages are installed inside the function via subprocess to avoid
    # requiring virtualenv on the worker node (runtime_env pip needs virtualenv).
    _pip_pkgs_for_worker = pip_pkgs
    def _run_training_inline(script_b64_arg: str, env_vars_arg: dict) -> dict:
        import os as _os, base64 as _b64, sys as _sys, subprocess as _sp
        from io import StringIO as _StringIO
        # Install required packages directly — avoids the virtualenv requirement
        if _pip_pkgs_for_worker:
            _sp.run(
                [_sys.executable, "-m", "pip", "install", "-q"] + _pip_pkgs_for_worker,
                capture_output=True,
            )
        for k, v in env_vars_arg.items():
            _os.environ[k] = v
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
        ray.init(address=ray_addr, ignore_reinit_error=True)
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
        },
    }


_ZERO_SHOT_MODELS = ("grounding-dino", "groundingdino", "owl-vit", "owlvit", "owlv2")

# Common model ID typo corrections
_MODEL_ID_FIXES = {
    "facebook/detr-resnet50": "facebook/detr-resnet-50",
    "facebook/detr-resnet101": "facebook/detr-resnet-101",
}


def _normalize_model_name(model_name: str) -> str:
    return _MODEL_ID_FIXES.get(model_name, model_name)


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
def submit_job(project_id: int, req: TrainRequest):
    req.model_name = _normalize_model_name(req.model_name)
    _check_zero_shot(req.model_name, req.engine)
    job_id = str(uuid.uuid4())[:8]
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
                cluster, status, progress, log, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',0,'',?)""",
            (
                job_id, name, project_id, dataset_label,
                req.training_type, req.model_name, req.engine,
                req.epochs, req.batch_size, req.learning_rate,
                req.optimizer, req.imgsz, req.notes,
                req.lora_rank, req.quantization, req.max_seq_len,
                req.chat_template, req.grad_accum, req.text_dataset,
                cluster,
                time.time(),
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
    """Check if the Ray Serve deployment for this job is RUNNING via Ray Client."""
    with get_db() as conn:
        row = conn.execute("SELECT ray_serve_url, inference_provider FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not row["ray_serve_url"]:
        return {"online": False, "status": "no_url"}
    try:
        import ray as _ray
        from ray import serve as _serve
        from urllib.parse import urlparse as _up
        _host = _up(row["ray_serve_url"]).hostname or "100.68.53.118"
        _addr = f"ray://{_host}:10001"
        if not _ray.is_initialized():
            _ray.init(address=_addr, ignore_reinit_error=True, log_to_driver=False)
        _status = _serve.status()
        _app_name = f"medimage-{job_id}"
        _info = _status.applications.get(_app_name)
        if _info is None:
            return {"online": False, "status": "not_deployed"}
        _s = str(_info.status)
        _online = "RUNNING" in _s or "HEALTHY" in _s
        return {"online": _online, "status": _s}
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


def _run_import(job_id: str, req: ImportModelRequest):
    """Register an imported / pretrained model. No weights download needed —
    the Ray Serve actor loads pretrained weights from MODEL_NAME at deploy time."""
    try:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (time.time(), job_id))
            conn.commit()

        _set_step(job_id, "validate")
        _append_log(job_id, f"Registering model: {req.model_name} ({req.engine} / {req.training_type})")
        _set_progress(job_id, 50)

        # Validate HuggingFace model exists (quick metadata fetch, no download)
        if req.source_type == "huggingface" and req.model_name:
            try:
                import urllib.request as _ur
                _url = f"https://huggingface.co/api/models/{req.model_name}"
                _ur.urlopen(_url, timeout=10)  # noqa: S310 — validating public HF model
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
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='global_modal_url'").fetchone()
    url = row["value"] if row else ""
    if not url:
        raise HTTPException(status_code=404, detail="No global Modal URL configured")
    with get_db() as conn:
        row2 = conn.execute("SELECT value FROM settings WHERE key='global_modal_api_key'").fetchone()
    key = row2["value"] if row2 else ""
    return await _ping_endpoint(url, key)


@app.get("/api/settings/inference/ray-status")
async def global_ray_status(url: str = ""):
    if not url:
        with get_db() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='global_ray_serve_url'").fetchone()
        url = (row["value"] if row else "") or ""
    if not url:
        return {"online": False}
    return await _ping_endpoint(url, "")


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
    token_id: str
    token_secret: str
    gpu_type: str = "T4"
    num_workers: int = 1


def _modal_script(req: ModalStartRequest) -> str:
    gpu_spec = f'gpu="{req.gpu_type}"' if req.gpu_type != "cpu" else "gpu=None"
    return textwrap.dedent(f"""\
        import modal, subprocess, time, socket

        app = modal.App("medimage-ray")

        # Shared store so workers can find the head node IP
        head_ip_store = modal.Dict.from_name("medimage-ray-head-ip", create_if_missing=True)

        ray_image = (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install("ray[serve]>=2.30", "fastapi", "uvicorn",
                         "python-multipart", "pillow")
        )

        # Head node — always 1 container, exposes Ray Dashboard.
        @app.function(image=ray_image, {gpu_spec}, timeout=86400, memory=16384, min_containers=1)
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
        @app.function(image=ray_image, {gpu_spec}, timeout=86400, memory=16384)
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

        # Step 1: send stop command
        try:
            r = subprocess.run(
                ["python3", "-m", "modal", "app", "stop", "medimage-ray"],
                env=env, timeout=60, capture_output=True, text=True,
            )
            if r.returncode != 0:
                out = (r.stdout + r.stderr)[-400:].strip()
                _modal_state["logs"].append(f"Stop command returned exit {r.returncode}: {out}")
                _modal_state.update(status="error", proc=None)
                return
            _modal_state["logs"].append("Stop command accepted. Verifying containers are down...")
        except subprocess.TimeoutExpired:
            _modal_state["logs"].append("Stop timed out — check modal.com dashboard manually.")
            _modal_state.update(status="error", proc=None)
            return
        except Exception as exc:
            _modal_state["logs"].append(f"Stop error: {exc}")
            _modal_state.update(status="error", proc=None)
            return

        # Step 2: verify via `modal app list` — poll up to ~60s
        import time as _time
        for attempt in range(12):
            _time.sleep(5)
            try:
                check = subprocess.run(
                    ["python3", "-m", "modal", "app", "list"],
                    env=env, timeout=15, capture_output=True, text=True,
                )
                output = check.stdout + check.stderr
                # If the app name no longer appears as 'deployed'/'running', it's down
                lines = [l for l in output.splitlines() if "medimage-ray" in l.lower()]
                still_up = any(
                    kw in l.lower() for l in lines
                    for kw in ("deployed", "running", "active")
                )
                if not still_up:
                    _modal_state["logs"].append(
                        f"✓ Confirmed: medimage-ray is no longer running on Modal."
                    )
                    _modal_state.update(status="idle", proc=None, ray_url=None)
                    return
                _modal_state["logs"].append(
                    f"Still shutting down... ({(attempt+1)*5}s)"
                )
            except Exception:
                pass  # network hiccup — keep polling

        # Timed out waiting for confirmation — still mark idle
        _modal_state["logs"].append(
            "Timed out waiting for shutdown confirmation. "
            "Cluster may still be stopping — check modal.com/apps."
        )
        _modal_state.update(status="idle", proc=None, ray_url=None)

    threading.Thread(target=_do_stop, daemon=True).start()
    return {"status": "stopping"}


@app.get("/api/modal/status")
def modal_status_ep():
    return {
        "status": _modal_state["status"],
        "ray_url": _modal_state["ray_url"],
        "logs": _modal_state["logs"][-30:],
        "num_workers": _modal_state["num_workers"],
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

    return textwrap.dedent(f"""\
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

_SERVE_APP_PY = textwrap.dedent("""\
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
_SERVE_MODEL_PY = textwrap.dedent("""\
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

    _ray_address = os.environ.get("RAY_ADDRESS", "ray://100.68.53.118:10001")
    if not ray.is_initialized():
        ray.init(address=_ray_address, ignore_reinit_error=True, log_to_driver=False)

    # Pin actors to the head node using the built-in internal head resource.
    # Ray always exposes node:__internal_head__ on the head node.
    _node_resource = {"node:__internal_head__": 0.001}

    # ── Deployment class — plain __call__, NO @serve.ingress(FastAPI()) ───────
    # NOTE: runtime_env pip is intentionally NOT used here because Ray cluster
    # nodes may not have virtualenv. All required packages must be pre-installed.
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
                self._load(weights)  # None → load pretrained from MODEL_NAME
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
                # ── 1. HuggingFace transformers (detection / classification / segmentation) ──
                if self._hf_processor is not None:
                    if not image_bytes:
                        return {"error": "Image required for this model type"}
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
                        logits = outputs.logits  # (1, C, H, W)
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

                # ── 2. Ultralytics YOLO (detection + instance segmentation) ─────────────
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

                # ── 3. Segmentation Models PyTorch (semantic segmentation) ───────────────
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
                        logits = self.model(tfm(img).unsqueeze(0))  # (1, C, H, W)
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

                # ── 4. PyTorch / TIMM classification ─────────────────────────────────────
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

                # ── 5. LLM / VLM text generation ─────────────────────────────────────────
                if tt in ("llm-text", "vlm-finetune"):
                    import torch
                    if not prompt:
                        return {"error": "Prompt required for LLM inference"}
                    full_prompt = (system_prompt + "\\n" + prompt).strip() if system_prompt else prompt
                    inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                    t0 = time.time()
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    ms = int((time.time() - t0) * 1000)
                    n_tok = out.shape[1] - inputs.input_ids.shape[1]
                    resp = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    return {"type": tt, "response": resp, "tokens_generated": n_tok,
                            "inference_time_ms": ms, "model_name": MODEL_NAME}

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
_RAY_ACTOR_PY = textwrap.dedent("""\
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
                print(f"[actor] No weights at {WEIGHTS_KEY} — loading pretrained")
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

    import ray
    _ray_address = os.environ.get("RAY_ADDRESS", "ray://100.68.53.118:10001")
    if not ray.is_initialized():
        ray.init(address=_ray_address, ignore_reinit_error=True, log_to_driver=False)

    @ray.remote(num_cpus=1)
    class ModelInferenceActor:
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
            if self.model is None:
                return {"error": self._load_error or "Model not loaded"}
            tt = TRAINING_TYPE
            try:
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
                        res = self._hf_processor.post_process_object_detection(outputs, threshold=conf, target_sizes=target_sizes)[0]
                        dets = []
                        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                            name = (self._hf_id2label or {}).get(label.item(), f"class_{label.item()}")
                            x1, y1, x2, y2 = box.tolist()
                            dets.append({"class_name": name, "confidence": round(float(score), 4), "bbox": [x1, y1, x2 - x1, y2 - y1]})
                        return {"type": "detection", "detections": dets}
                    elif tt == "classification":
                        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
                        id2label = self._hf_id2label or {}
                        preds = sorted([{"label": id2label.get(i, f"class_{i}"), "confidence": p} for i, p in enumerate(probs)], key=lambda x: -x["confidence"])
                        return {"type": "classification", "predictions": preds[:5], "top_label": preds[0]["label"]}
                    elif tt == "segmentation":
                        logits = outputs.logits
                        up = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
                        mask = up.argmax(dim=1)[0]
                        total_px = mask.numel()
                        id2label = self._hf_id2label or {}
                        segs = [{"class_name": id2label.get(ci, f"class_{ci}"), "class_id": ci, "pixel_count": int((mask == ci).sum()), "coverage": round(int((mask == ci).sum()) / total_px, 4)} for ci in mask.unique().tolist()]
                        segs.sort(key=lambda x: -x["pixel_count"])
                        return {"type": "segmentation", "segments": segs, "num_classes": len(segs)}
                    return {"error": f"HuggingFace inference not implemented for type={tt}"}

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

                if tt in ("llm-text", "vlm-finetune"):
                    import torch
                    if not prompt:
                        return {"error": "Prompt required for LLM inference"}
                    full_prompt = (system_prompt + "\\n" + prompt).strip() if system_prompt else prompt
                    inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                    t0 = time.time()
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    ms = int((time.time() - t0) * 1000)
                    n_tok = out.shape[1] - inputs.input_ids.shape[1]
                    resp = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    return {"type": tt, "response": resp, "tokens_generated": n_tok, "inference_time_ms": ms, "model_name": MODEL_NAME}

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
    """Deploy model as a named Ray Actor via Ray Client subprocess.
    Avoids Ray Serve (numpy ABI mismatch) and Job Agent (not running on this cluster).
    Named detached actors persist on the cluster after the subprocess exits.
    """
    import subprocess, sys, tempfile, os as _os
    state = _get_model_deploy_state(job_id)
    env_vars = payload.get("runtime_env", {}).get("env_vars", {})

    # Derive ray:// address from dashboard URL
    from urllib.parse import urlparse as _urlparse
    _p = _urlparse(ray_dashboard_url)
    _ray_addr = f"ray://{_p.hostname}:10001"

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

    # Phase 2: run actor deployment script in a subprocess (Ray Client, no Ray Serve)
    state["logs"] = ["Loading model weights and creating Ray actor (this may take several minutes)…"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_RAY_ACTOR_PY)
        script_path = f.name

    remaining = max(60, int(deadline - time.time()))
    try:
        proc_env = {**_os.environ, **env_vars, "RAY_ADDRESS": _ray_addr}
        result = subprocess.run(
            [sys.executable, script_path],
            env=proc_env,
            capture_output=True,
            text=True,
            timeout=remaining,
        )
        try:
            _os.unlink(script_path)
        except Exception:
            pass

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            err = stderr[-600:] if stderr else stdout[-600:] or "Unknown error"
            state.update(status="error", logs=[f"Deploy failed (exit {result.returncode}): {err}"])
            return

        if "ACTOR_READY:" not in stdout:
            msg = stdout[-400:] if stdout else (stderr[-400:] if stderr else "No output from deploy script")
            state.update(status="error", logs=[f"Deploy script completed but actor not ready: {msg}"])
            return

    except subprocess.TimeoutExpired:
        try:
            _os.unlink(script_path)
        except Exception:
            pass
        state.update(status="error", logs=[f"Deploy timed out after {remaining}s — model weights may be too large to load"])
        return
    except Exception as exc:
        try:
            _os.unlink(script_path)
        except Exception:
            pass
        state.update(status="error", logs=[f"Deploy subprocess error: {exc}"])
        return

    # Phase 3: actor is ready — update state and DB
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
        if not ray.is_initialized():
            ray.init(address="auto")
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
            "env_vars": {"_SCRIPT": script_b64},
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
    return {
        "status": state["status"],
        "url": state["url"],
        "logs": state["logs"][-20:],
    }


@app.post("/api/jobs/{job_id}/deploy-ray/stop")
def model_deploy_stop(job_id: str):
    """Kill the named Ray Actor for this model."""
    actor_name = f"model-{job_id}"
    ray_client_addr = "ray://100.68.53.118:10001"
    logs = []
    try:
        import ray as _ray
        if not _ray.is_initialized():
            _ray.init(address=ray_client_addr, ignore_reinit_error=True, log_to_driver=False)
        try:
            _actor = _ray.get_actor(actor_name, namespace="default")
            _ray.kill(_actor)
            logs.append(f"[stop] Actor '{actor_name}' killed successfully")
        except ValueError:
            logs.append(f"[stop] Actor '{actor_name}' not found (already stopped)")
    except Exception as e:
        logs.append(f"[stop] warning: {e}")
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


# ─── HuggingFace Dataset Import ──────────────────────────────────────────────

_hf_import_jobs: dict = {}  # job_id → {status, progress, log, bucket, error, done_files, total_files}


class HFImportRequest(BaseModel):
    hf_dataset_id: str
    bucket_name: str
    max_files: int = 20


def _log_hf(job_id: str, line: str):
    ts = datetime.now().strftime("%H:%M:%S")
    _hf_import_jobs[job_id]["log"] += f"[{ts}] {line}\n"


def _run_hf_import(job_id: str, req: HFImportRequest):
    import requests as _req
    import boto3 as _boto3
    from botocore.exceptions import ClientError as _CE

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
        r = _req.get(tree_url, timeout=30)
        if not r.ok:
            raise Exception(f"HuggingFace API returned HTTP {r.status_code} — dataset not found or is private")

        entries = r.json()
        blobs = [e for e in entries if e.get("type") == "blob"]

        # If no blobs at root, try data/ subdirectory
        if not blobs:
            r2 = _req.get(f"{tree_url}/data", timeout=30)
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
            dl = _req.get(hf_url, timeout=120)
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


@app.post("/api/datasets/import-hf")
def import_hf_dataset(req: HFImportRequest):
    import re as _re
    if not _re.match(r"^[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]$", req.bucket_name):
        raise HTTPException(
            status_code=400,
            detail="Bucket name: 3–63 lowercase letters/numbers/hyphens, no leading/trailing hyphens",
        )
    if not req.hf_dataset_id.strip():
        raise HTTPException(status_code=400, detail="HuggingFace dataset ID is required")

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
    threading.Thread(target=_run_hf_import, args=(job_id, req), daemon=True).start()
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
        """Call model Ray Actor via Ray Client (no Ray Serve dependency)."""
        import ray as _ray, asyncio as _asyncio
        _ray_addr = "ray://100.68.53.118:10001"
        if not _ray.is_initialized():
            _ray.init(address=_ray_addr, ignore_reinit_error=True, log_to_driver=False)
        _actor_name = f"model-{job_id_}"
        try:
            _actor = _ray.get_actor(_actor_name, namespace="default")
        except Exception:
            raise RuntimeError(f"Model actor '{_actor_name}' is not running. Please deploy the model first.")
        _img_bytes = None
        if image:
            _img_bytes = await image.read()
        # ray.get() is blocking — run in executor to avoid blocking the async loop
        _result = await _asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _ray.get(_actor.infer.remote(
                _img_bytes, conf_threshold, prompt, system_prompt
            ), timeout=120),
        )
        if isinstance(_result, dict) and _result.get("error"):
            raise RuntimeError(_result["error"])
        return _result

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
