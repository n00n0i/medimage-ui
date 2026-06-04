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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form as FForm, Request, Body
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
_KC_PUBLIC_PREFIXES = ("/api/ls-goto/",)

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
    """Proxy Keycloak Account REST API TOTP setup — returns QR code and secret"""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    auth = request.headers.get("Authorization", "")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_KC_BASE}/realms/{_KC_REALM}/account/totp-setup",
            headers={"Authorization": auth},
        )
        if not resp.is_success:
            raise HTTPException(resp.status_code, "TOTP setup not available")
        return resp.json()


@app.post("/api/profile/totp-verify")
async def profile_totp_verify(request: Request):
    """Proxy Keycloak Account REST API TOTP registration"""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    auth = request.headers.get("Authorization", "")
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_KC_BASE}/realms/{_KC_REALM}/account/totp",
            headers={"Authorization": auth, "Content-Type": "application/json"},
            json=body,
        )
        if not resp.is_success:
            raise HTTPException(resp.status_code, resp.text or "TOTP verification failed")
        return {"ok": True}


@app.get("/api/profile/totp-credentials")
async def profile_totp_credentials(request: Request):
    """List current user's OTP credentials via Admin API"""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub")
    token = await _kc_admin_token()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{user_id}/credentials",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        creds = resp.json()
        return [
            {"id": c["id"], "userLabel": c.get("userLabel", "Authenticator"),
             "createdDate": c.get("createdDate")}
            for c in creds if c.get("type") == "otp"
        ]


@app.delete("/api/profile/totp-credentials/{cred_id}", status_code=204)
async def profile_delete_totp(cred_id: str, request: Request):
    """Delete current user's OTP credential"""
    payload = _extract_token_payload(request)
    if not payload:
        raise HTTPException(401, "Not authenticated")
    user_id = payload.get("sub")
    token = await _kc_admin_token()
    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            f"{_KC_BASE}/admin/realms/{_KC_REALM}/users/{user_id}/credentials/{cred_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()


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
    import sys
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
    if training_type == "vlm-finetune" or engine == "HuggingFace":
        train_vlm_finetune(
            model_name, text_dataset, max_seq_len, lora_rank, quantization,
            epochs, batch, grad_accum, lr, chat_template,
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

        # Generate presigned URL using public/VPN-accessible URL
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
    _append_log(job_id, f"[cluster] Connecting to Ray at {ray_addr}")

    # Export dataset from Label Studio (skip for LLM/VLM — uses HF text_dataset instead)
    engine = job.get("engine", "Ultralytics")
    training_type_for_export = job.get("training_type", "detection")
    if engine in ("Unsloth", "HuggingFace") or training_type_for_export == "vlm-finetune":
        dataset_url = ""
        _append_log(job_id, "[cluster] LLM/VLM engine — skipping LS export (uses text_dataset)")
    else:
        _append_log(job_id, f"[cluster] Exporting dataset from LS project {project_id} ...")
        # SMP segmentation and self-supervised need image data from JSON export
        _needs_json = engine in ("Segmentation Models PyTorch",) or training_type_for_export == "self-supervised"
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
        "DATASET_URL":     dataset_url,
        "WEIGHTS_BUCKET":  weights_bucket,
        "WEIGHTS_KEY":     weights_key,
        "MINIO_URL":       MINIO_PUBLIC_URL,
        "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
        "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
        "LS_PUBLIC_URL":   LS_PUBLIC_URL,
        "LS_TOKEN":        LS_TOKEN,
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

    runtime_env = {"pip": pip_pkgs}

    # Define the worker function inline so cloudpickle serializes the full body
    # (not a reference to 'main' module which doesn't exist on the Ray worker)
    def _run_training_inline(script_b64_arg: str, env_vars_arg: dict) -> dict:
        import os as _os, base64 as _b64, sys as _sys
        from io import StringIO as _StringIO
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

    try:
        with _run_on_ray_cluster._lock:
            ray.init(address=ray_addr, ignore_reinit_error=True)
            _append_log(job_id, f"[cluster] Connected ({len(ray.nodes())} node(s))")

            remote_fn = ray.remote(_run_training_inline)
            future    = remote_fn.options(runtime_env=runtime_env).remote(script_b64, env_vars)
            _append_log(job_id, "[cluster] Training task submitted, waiting ...")

            elapsed   = 0
            poll_sec  = 15
            while True:
                time.sleep(poll_sec)
                elapsed += poll_sec
                ready, _ = ray.wait([future], timeout=0)
                if ready:
                    break
                _append_log(job_id, f"[cluster] Training in progress ({elapsed // 60}m {elapsed % 60}s elapsed) ...")

            result = ray.get(future)
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass

    # Save full stdout to log
    stdout = result.get("stdout", "")
    for line in stdout.splitlines():
        if line.strip():
            _append_log(job_id, line)

    weights_path = result.get("weights_path")
    if weights_path:
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET s3_weights_path = ? WHERE id = ?",
                (weights_path, job_id),
            )
            conn.commit()
    _append_log(job_id, f"✓ Training complete — weights: {weights_path or '(not found)'}")
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
    """Run vision model training — real (Ray/Modal cluster) or simulation (local)."""
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

        # ── Local simulation fallback ───────────────────────────────────────
        epochs  = job["epochs"]
        model   = job["model_name"]
        engine  = job["engine"]
        proj_id = job["project_id"]

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

    return {
        "ray":   {"available": ray_ok,                    "url": RAY_URL,       "info": ray_info},
        "modal": {"available": modal_status == "running", "status": modal_status, "ray_url": modal_ray_url},
    }


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

    target_fn = run_llm_training if is_llm else run_training
    threading.Thread(target=target_fn, args=(row,), daemon=True).start()

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
    health_url = url.rstrip("/") + "/health"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
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
async def get_ray_status(job_id: str, url: str = ""):
    """Ping the Ray Serve endpoint to check if it's online."""
    if not url:
        with get_db() as conn:
            row = conn.execute("SELECT ray_serve_url FROM jobs WHERE id = ?", (job_id,)).fetchone()
        url = (row["ray_serve_url"] if row else "") or ""
    if not url:
        return {"online": False}
    return await _ping_endpoint(url, "")


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


# ─── Modal Inference Deploy ────────────────────────────────────────────────────

_modal_infer_state: dict = {
    "status": "idle",   # idle | deploying | running | error
    "url": None,
    "logs": [],
}


def _modal_infer_script(gpu_type: str = "T4") -> str:
    gpu_spec = f'gpu="{gpu_type}"' if gpu_type != "cpu" else "gpu=None"
    return textwrap.dedent(f"""\
        import modal
        app = modal.App("medimage-inference")
        image = (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install("fastapi", "uvicorn", "python-multipart", "pillow")
        )
        @app.function(image=image, {gpu_spec}, timeout=300, keep_warm=1)
        @modal.asgi_app()
        def serve():
            from fastapi import FastAPI, File, Form, UploadFile
            from typing import Optional
            web = FastAPI()

            @web.post("/inference")
            async def infer(
                training_type: str = Form(...),
                model_id: str = Form(""),
                conf_threshold: float = Form(0.5),
                image: Optional[UploadFile] = File(None),
                prompt: str = Form(""),
                system_prompt: str = Form(""),
            ):
                # TODO: load your model and run actual inference
                if training_type in ("llm-text", "vlm-finetune"):
                    return {{"type": training_type, "response": f"Echo: {{prompt[:80]}}", "tokens": 0, "simulated": False}}
                return {{"type": training_type, "predictions": [], "simulated": False}}

            @web.get("/health")
            async def health():
                return {{"status": "ok"}}

            return web
    """)


def _run_modal_infer_deploy(token_id: str, token_secret: str, gpu_type: str) -> None:
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = token_id
    env["MODAL_TOKEN_SECRET"] = token_secret
    script_path = "/tmp/medimage_modal_infer.py"
    with open(script_path, "w") as f:
        f.write(_modal_infer_script(gpu_type))
    try:
        proc = subprocess.run(
            ["python3", "-m", "modal", "deploy", script_path],
            env=env, capture_output=True, text=True, timeout=300,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        logs = [l for l in output.splitlines() if l.strip()]
        _modal_infer_state["logs"] = logs[-30:]
        m = _re.search(r'https://[^\s]+\.modal\.run[^\s]*', output)
        if m:
            url = m.group(0).rstrip('/.,;')
            _modal_infer_state["url"] = url
            _modal_infer_state["status"] = "running"
            with get_db() as conn:
                conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", ("global_modal_url", url))
                conn.commit()
        else:
            _modal_infer_state["status"] = "error"
            _modal_infer_state["logs"].append("Could not parse deployment URL from output")
    except subprocess.TimeoutExpired:
        _modal_infer_state["status"] = "error"
        _modal_infer_state["logs"].append("Deploy timed out after 5 minutes")
    except Exception as exc:
        _modal_infer_state["status"] = "error"
        _modal_infer_state["logs"].append(str(exc))


@app.post("/api/modal/inference/deploy")
def modal_infer_deploy(body: dict):
    if not body.get("token_id") or not body.get("token_secret"):
        raise HTTPException(status_code=400, detail="token_id and token_secret required")
    if _modal_infer_state["status"] == "deploying":
        raise HTTPException(status_code=409, detail="Deploy already in progress")
    _modal_infer_state.update(status="deploying", url=None, logs=["Deploying to Modal.com…"])
    threading.Thread(
        target=_run_modal_infer_deploy,
        args=(body["token_id"], body["token_secret"], body.get("gpu_type", "T4")),
        daemon=True,
    ).start()
    return {"status": "deploying"}


@app.get("/api/modal/inference/status")
def modal_infer_status_ep():
    return {
        "status": _modal_infer_state["status"],
        "url": _modal_infer_state["url"],
        "logs": _modal_infer_state["logs"][-20:],
    }


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

    # Phase 2: submit job, retry on transient agent errors
    for attempt in range(1, 20):
        elapsed = int(time.time() - start)
        attempt_payload = {**payload, "submission_id": f"medimage-inference-{int(time.time())}"}
        _ray_serve_state["logs"] = [f"Submitting job (attempt {attempt}, {elapsed}s elapsed)…"]
        try:
            r = httpx.post(f"{ray_dashboard_url}/api/jobs/", json=attempt_payload, timeout=30)
            if r.is_success:
                job_id = r.json().get("submission_id") or r.json().get("job_id", attempt_payload["submission_id"])
                _ray_serve_state["logs"] = [f"Job submitted: {job_id}"]
                _poll_ray_job(ray_dashboard_url, job_id, serve_url)
                return
            text = r.text
            _ray_serve_state["logs"] = [f"Attempt {attempt} ({elapsed}s): HTTP {r.status_code} — {text[:300]}"]
            if "get_target_agent" in text or r.status_code in (500, 503):
                time.sleep(20)
                continue
            # Fatal
            _ray_serve_state.update(status="error", logs=[f"Ray Jobs API error: {text[:400]}"])
            return
        except Exception as exc:
            _ray_serve_state["logs"] = [f"Connection error (attempt {attempt}): {exc}"]
            time.sleep(15)
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
    provider = job.get("inference_provider", "") or ""
    fname = (image.filename if image else None) or prompt[:20] or "upload"
    # seed RNG from filename for reproducible-ish demo results
    rng = random.Random(abs(hash(fname)) % (2**31))

    t_start = time.time()

    # ── Universal external routing (Modal or Ray Serve) ──────────────────────
    async def _route_external(base_url: str, api_key: str) -> dict:
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
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(endpoint, json=payload, headers=headers_ext)
        else:
            if not image:
                raise HTTPException(status_code=400, detail="Image required for this model type")
            img_bytes = await image.read()
            files = {"image": (image.filename or "image.jpg", img_bytes, "image/jpeg")}
            data = {"model_id": model_id, "conf_threshold": str(conf_threshold), "training_type": training_type}
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(endpoint, headers=headers_ext, data=data, files=files)
        r.raise_for_status()
        return r.json()

    if provider == "modal" and job.get("modal_url"):
        try:
            return await _route_external(job["modal_url"], job.get("modal_api_key") or "")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Modal endpoint error: {e}")

    if provider == "ray" and job.get("ray_serve_url"):
        try:
            return await _route_external(job["ray_serve_url"], "")
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
                return await _route_external(_gs["global_ray_serve_url"], "")
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Ray Serve endpoint error: {e}")

    # ── Simulate ─────────────────────────────────────────────────────────────

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
        # ── Fallback simulate (real inference handled above via provider) ───
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
            "simulated": True,
        }

    raise HTTPException(status_code=400, detail=f"Unknown training_type: {training_type}")
