"""
VLM / LLM fine-tuning — separate from the deep-learning training path so
changes here don't affect YOLO / MONAI / nnU-Net / etc. and vice versa.

The Ray worker image bakes these deps in once at build time, so jobs use
the resolved versions without re-running `pip install` every time:

    transformers, peft, trl, bitsandbytes, accelerate, datasets, numpy<2.0
    pillow, boto3, requests

Two engines are supported:

  1. Unsloth (LLM, text-only)        — train_unsloth_llm
  2. HuggingFace transformers + PEFT (VLM, image+text) — train_vlm_finetune

A third path, train_unsloth_vlm, is kept for parity but is NOT routed from
main() — Unsloth's install tree conflicts with the cluster's torch, and
the bare-HF path works for every model Unsloth supports plus several it
doesn't (InternVL2, MedGemma, SmolVLM).

Two runtime quirks the Ray cluster has that we have to work around at
import time:

  - triton is partially installed: some submodules exist, others
    (triton.ops.matmul_perf_model, triton.compiler, ...) don't, and
    torch._inductor / bitsandbytes / torchao import them eagerly.
    setup_import_stubs() installs a lazy module that materialises
    missing children on demand. See setup_import_stubs docstring.

  - pandas on the cluster was compiled against numpy 1.x ABI (dtype
    size 96). Any 2.x numpy breaks pandas._libs.interval. numpy<2.0
    is pinned in the image so this is handled at build time, not here.
"""
import os
import re
import sys
import json
import base64
import zipfile
import tempfile
import subprocess
import importlib
import importlib.machinery
import types as _types
from pathlib import Path


# ── helpers ────────────────────────────────────────────────────────────────

def upload_to_minio(local_path, bucket, key, minio_url, minio_access, minio_secret):
    """Upload a file to MinIO and print the canonical s3:// path. The job
    log scanner uses the WEIGHTS_UPLOADED: prefix to record the final
    weight location in the database."""
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


# ── import-time stubs ──────────────────────────────────────────────────────
#
# The Ray cluster image has a partial triton install. transformers,
# bitsandbytes, and torch._inductor all do `import triton.X` at top
# level, and a few of those submodules are missing in the baked image.
# Instead of swapping the image, we install a lazy module under
# `triton` that materialises any missing child on attribute access.
# Idempotent — safe to call before every VLM/LLM job.

def setup_import_stubs():
    """Install sklearn + triton import-time stubs for the VLM/LLM training
    subprocess. The transformers Trainer import chain pulls
    sklearn -> pandas, and the cluster's pandas is built against numpy
    1.x ABI. We never call sklearn metrics during VLM/LLM training, so
    we stub them out.

    The top-level `sklearn` stub MUST have a real __spec__ (not None) —
    transformers.utils.import_utils line 153 does
        _sklearn_available = importlib.util.find_spec("sklearn") is not None
    and find_spec reads sys.modules[name].__spec__. A bare types.ModuleType
    has __spec__=None which makes find_spec raise ValueError instead of
    returning None. Use importlib.machinery.ModuleSpec to give it a real
    spec with submodule_search_locations so `import sklearn.metrics`
    resolves to our stubbed submodule.

    For triton, use a recursive `_TritonLazyPackage` class that
    lazy-creates any missing child submodule on attribute access
    — handles `triton.X`, `triton.X.Y`, `triton.X.Y.Z`, etc.

    Every lazy package has a real `__spec__` (ModuleSpec with
    is_package=True) so importlib.util.find_spec returns it
    cleanly instead of raising "ValueError: triton.__spec__
    is None". bitsandbytes.triton.triton_utils.is_triton_available
    does exactly that check.

    Every lazy package is also `__call__`-able as a no-op so
    call patterns like `triton.Config(...)` (used in torchao
    and bitsandbytes) succeed even when the actual
    `triton.Config` class doesn't exist in the partial triton
    install. The call returns the stub itself.
    """
    class _TritonLazyPackage(_types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self.__spec__ = importlib.machinery.ModuleSpec(
                name, None, is_package=True)
            if self.__path__:
                self.__spec__.submodule_search_locations = self.__path__

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            if name.startswith("_") and name not in ("__path__",):
                raise AttributeError(name)
            full = f"{self.__name__}.{name}"
            if full not in sys.modules:
                sub = _TritonLazyPackage(full)
                sys.modules[full] = sub
            return sys.modules[full]

    # ── sklearn stub ─────────────────────────────────────────────────────
    _sklearn_spec = importlib.machinery.ModuleSpec("sklearn", None, is_package=True)
    _sklearn_spec.submodule_search_locations = []
    _sklearn_stub = _types.ModuleType("sklearn")
    _sklearn_stub.__spec__ = _sklearn_spec
    sys.modules["sklearn"] = _sklearn_stub
    for _m in ("sklearn.metrics", "sklearn.utils",
               "sklearn.utils._param_validation", "sklearn.utils._chunking",
               "sklearn.exceptions"):
        if _m not in sys.modules:
            _m_mod = _types.ModuleType(_m)
            _m_mod.__spec__ = importlib.machinery.ModuleSpec(_m, None)
            sys.modules[_m] = _m_mod
    # Pre-set the names transformers actually tries to import
    sys.modules["sklearn.metrics"].f1_score = None
    sys.modules["sklearn.metrics"].matthews_corrcoef = None
    class _UndefinedMetricWarning(Warning):
        pass
    sys.modules["sklearn.exceptions"].UndefinedMetricWarning = _UndefinedMetricWarning

    # ── triton stub ──────────────────────────────────────────────────────
    if "triton" not in sys.modules:
        sys.modules["triton"] = _TritonLazyPackage("triton")
    else:
        _existing = sys.modules["triton"]
        try:
            _existing.__path__ = []
        except (AttributeError, TypeError):
            pass
        if not isinstance(_existing, _TritonLazyPackage):
            _lazy_triton = _TritonLazyPackage("triton")
            for _attr in dir(_existing):
                if not _attr.startswith("_") and not hasattr(_lazy_triton, _attr):
                    try:
                        setattr(_lazy_triton, _attr, getattr(_existing, _attr))
                    except (AttributeError, TypeError):
                        pass
            sys.modules["triton"] = _lazy_triton

    for _triton_pkg in ("triton.ops", "triton.ops.matmul_perf_model",
                        "triton.language", "triton.backends",
                        "triton.backends.compiler", "triton.compiler",
                        "triton.compiler.compiler",
                        "triton.runtime", "triton.testing"):
        if _triton_pkg not in sys.modules:
            sys.modules[_triton_pkg] = _TritonLazyPackage(_triton_pkg)

    sys.modules["triton.ops.matmul_perf_model"].early_config_prune = lambda *a, **kw: None
    sys.modules["triton.ops.matmul_perf_model"].estimate_matmul_time = lambda *a, **kw: 0.0

    if not hasattr(sys.modules["triton.language"], "dtype"):
        class _StubDtype:
            def __repr__(self):
                return "<stub triton.language.dtype>"
        sys.modules["triton.language"].dtype = _StubDtype()

    if not hasattr(sys.modules["triton.backends.compiler"], "AttrsDescriptor"):
        sys.modules["triton.backends.compiler"].AttrsDescriptor = type(
            "AttrsDescriptor", (), {}
        )


# ── torch pre-flight ──────────────────────────────────────────────────────
#
# The Ray image bakes a known-good torch wheel, but the host GPU's CUDA
# driver may not match it. If torch.cuda.is_available() is False at job
# start, try multiple PyTorch cu indexes in order. The order matches the
# one we used in main.py before the refactor — newest first so a 580.x
# driver picks cu128 or cu126, an older 470.x driver falls back to cu118.
# This is the only place in the codebase that force-reinstalls torch; the
# VLM/LLM training functions themselves never touch pip.

def preflight_torch_for_vlm():
    """If the baked torch can't see a GPU, try multiple cu wheels in
    order. Surfaces every attempt's pip output on total failure so the
    user can see what their driver supports."""
    _gpu_check = subprocess.run(
        [sys.executable, "-c",
         "import torch, sys; "
         "sys.exit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True, text=True, timeout=60,
    )
    if _gpu_check.returncode == 0:
        print("[vlm_train] torch already sees a GPU, no pre-flight needed")
        return

    _smi = subprocess.run(
        ["bash", "-lc", "nvidia-smi 2>&1 | head -20 || echo 'nvidia-smi not available'"],
        capture_output=True, text=True, timeout=10,
    )
    _smi_text = (_smi.stdout or "") + (_smi.stderr or "")
    print(f"[vlm_train] nvidia-smi (pre-flight):\n{_smi_text}", file=sys.stderr)

    # Order newest-first so CUDA 12.x/13.x drivers pick the right wheel:
    #   cu128 — driver >= 580.x (CUDA 12.8)
    #   cu126 — driver >= 555.x (CUDA 12.6)
    #   cu124 — driver >= 550.x (CUDA 12.4)
    #   cu121 — driver >= 530.x (CUDA 12.1)
    #   cu118 — driver >= 470.x (CUDA 11.8)
    #   cu117 — driver >= 460.x (CUDA 11.7)  (fallback)
    _torch_attempts = [
        ("https://download.pytorch.org/whl/cu124", "torch==2.4.0+cu124"),
        ("https://download.pytorch.org/whl/cu124", "torch==2.5.0+cu124"),
        ("https://download.pytorch.org/whl/cu126", "torch==2.7.0+cu126"),
        ("https://download.pytorch.org/whl/cu121", "torch==2.1.0+cu121"),
        ("https://download.pytorch.org/whl/cu118", "torch==2.0.0+cu118"),
        ("https://download.pytorch.org/whl/cu117", "torch==2.0.0+cu117"),
    ]
    _attempt_logs = []
    _last_err = ""
    for _pytorch_index, _torch_pin in _torch_attempts:
        print(f"[vlm_train] trying {_torch_pin} from {_pytorch_index} ...", file=sys.stderr)
        _pip_res = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--force-reinstall", "--no-deps",
             "--index-url", _pytorch_index,
             _torch_pin],
            capture_output=True, text=True, timeout=600,
        )
        _last_err = (f"  pin={_torch_pin}\n"
                     f"  index={_pytorch_index}\n"
                     f"  pip stdout: {_pip_res.stdout[:500]!r}\n"
                     f"  pip stderr: {_pip_res.stderr[:500]!r}\n"
                     f"  pip returncode: {_pip_res.returncode}")
        _attempt_logs.append(_last_err)
        if _pip_res.returncode != 0:
            print(f"[vlm_train] pip install {_torch_pin} failed:\n{_last_err}", file=sys.stderr)
            continue
        _probe = subprocess.run(
            [sys.executable, "-c",
             "import torch, sys, os, ctypes.util; "
             "_t = torch; "
             "print(f'torch.__file__={_t.__file__}', file=sys.stderr); "
             "print(f'torch.__version__={_t.__version__}', file=sys.stderr); "
             "print(f'torch.version.cuda={_t.version.cuda}', file=sys.stderr); "
             "print(f'torch.backends.cuda.is_built()={_t.backends.cuda.is_built()}', file=sys.stderr); "
             "_lc = ctypes.util.find_library('cuda'); "
             "print(f'ctypes.util.find_library(cuda)={_lc}', file=sys.stderr); "
             "print(f'LD_LIBRARY_PATH={os.environ.get(\"LD_LIBRARY_PATH\", \"(unset)\")}', file=sys.stderr); "
             "print(f'cuda_available={_t.cuda.is_available()}', file=sys.stderr); "
             "print(f'devices={_t.cuda.device_count() if _t.cuda.is_available() else 0}', file=sys.stderr); "
             "sys.exit(0 if _t.cuda.is_available() else 1)"],
            capture_output=True, text=True, timeout=60,
        )
        print(_probe.stderr.strip() or "(no probe output)", file=sys.stderr)
        if _probe.returncode == 0:
            print(f"[vlm_train] OK {_torch_pin} works", file=sys.stderr)
            return
        print(f"[vlm_train] {_torch_pin} installed but GPU not visible", file=sys.stderr)

    # All attempts failed
    _drv_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", _smi_text)
    _driver_cuda = _drv_match.group(1) if _drv_match else "(not found)"
    _all_attempts = "\n\n".join(
        f"=== attempt {i+1}/{len(_torch_attempts)} ===\n{log}"
        for i, log in enumerate(_attempt_logs))
    _libcuda = subprocess.run(
        ["bash", "-lc",
         "ldconfig -p 2>/dev/null | grep -E 'libcuda\\.so' | head -5; "
         "find / -name 'libcuda.so*' 2>/dev/null | head -5"],
        capture_output=True, text=True, timeout=30,
    )
    raise RuntimeError(
        f"All torch install attempts failed to produce a "
        f"GPU-visible build.\n\n"
        f"=== nvidia-smi output ===\n{_smi_text}\n\n"
        f"=== nvidia-smi parsed ===\n"
        f"driver CUDA: {_driver_cuda}\n"
        f"\n"
        f"=== every pip attempt ({len(_torch_attempts)}) ===\n"
        f"{_all_attempts}\n"
        f"\n"
        f"=== libcuda search ===\n"
        f"{_libcuda.stdout[:500] if _libcuda.stdout else '(not found)'}\n"
    )


# ── train functions ───────────────────────────────────────────────────────

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
    print(f"[train] Loading text dataset: {text_dataset}")
    try:
        ds = load_dataset(text_dataset, split="train")
    except Exception as e:
        print(f"[warn] HF load failed ({e}), treating as JSONL string / inline data")
        rows = [json.loads(l) for l in text_dataset.splitlines() if l.strip()]
        ds = HFDataset.from_list(rows)

    text_field = None
    for candidate in ("text", "conversations", "messages", "instruction"):
        if candidate in ds.column_names:
            text_field = candidate
            break
    if text_field is None:
        raise RuntimeError(f"Dataset has no recognized text column. Columns: {ds.column_names}")
    print(f"[train] Dataset: {len(ds)} rows, text_field='{text_field}'")

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
    zip_path = f"/tmp/lora_{job_id}.zip"
    with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED) as zf:
        for fp in Path(save_dir).rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(save_dir))
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)


def train_vlm_finetune(model_name, text_dataset, max_seq_len, lora_rank, quantization,
                        epochs, batch, grad_accum, lr, chat_template, job_id,
                        w_bucket, w_key, minio_url, minio_access, minio_secret,
                        dataset_url=None, project_id=0, img_dir=None):
    """Fine-tune a Vision Language Model using bare HuggingFace transformers + PEFT.

    We use the bare-HF path (not Unsloth) because the Ray cluster bakes a
    known-good torch wheel and Unsloth's install tree would force-replace
    it. The bare-HF path works for every model Unsloth supports plus
    InternVL2, MedGemma, and SmolVLM.

    Dataset resolution (in order):
    1. text_dataset if set — HF dataset id, local path, or raw JSONL
    2. dataset_url if set — LS project zip, extract annotations.json +
       images, build VLM rows from task annotations
    3. Error
    """
    import zipfile as _zf
    import torch
    from transformers import (
        AutoProcessor, AutoModelForVision2Seq,
        BitsAndBytesConfig, TrainingArguments, Trainer,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset, Dataset as HFDataset
    from PIL import Image as _Image
    import io as _io, base64 as _b64, glob as _glob, tempfile as _tf

    print(f"[train] VLM fine-tune (HF) -- model: {model_name}  lora_r={lora_rank}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            f"torch {torch.__version__} on this Ray worker cannot see a GPU "
            f"(cuda_available=False). VLM training needs CUDA. See the "
            f"pre-flight nvidia-smi / torch install output in the cluster log."
        )

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

    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules="all-linear",
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    )
    vlm = get_peft_model(vlm, lora_cfg)
    vlm.print_trainable_parameters()

    rows = None
    if text_dataset:
        print(f"[train] Loading text_dataset: {text_dataset[:80]}{'...' if len(text_dataset) > 80 else ''}")
        try:
            ds = load_dataset(text_dataset, split="train")
            rows = list(ds)
        except Exception as e:
            print(f"[warn] HF load failed ({e}), trying as JSONL")
            rows = [json.loads(l) for l in text_dataset.splitlines() if l.strip()]
    elif dataset_url:
        import requests as _req
        tmpdir = _tf.mkdtemp(prefix="vlm_data_")
        print(f"[train] Downloading dataset zip: {dataset_url}")
        r = _req.get(dataset_url, stream=True, timeout=600)
        r.raise_for_status()
        zip_path = os.path.join(tmpdir, "dataset.zip")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        with _zf.ZipFile(zip_path) as z:
            z.extractall(tmpdir)
        img_dir = img_dir or os.path.join(tmpdir, "images")
        ann_path = None
        for cand in (os.path.join(tmpdir, "annotations.json"),
                     os.path.join(tmpdir, "tasks.json"),
                     os.path.join(tmpdir, "data", "annotations.json")):
            if os.path.isfile(cand):
                ann_path = cand
                break
        if not ann_path:
            raise RuntimeError(
                f"No annotations.json / tasks.json in zip -- VLM needs image+text pairs. "
                f"Got: {os.listdir(tmpdir)[:20]}"
            )
        with open(ann_path) as f:
            tasks = json.load(f)
        print(f"[train] Building VLM rows from {len(tasks)} LS tasks in project {project_id} ...")
        rows = []
        for t in tasks:
            data = t.get("data") or {}
            anns = t.get("annotations") or t.get("predictions") or []
            img_ref = (data.get("image") or data.get("img")
                       or (data.get("upload") or {}).get("file") or "")
            text_out = ""
            for a in anns:
                res = a.get("result") or []
                for r_item in res:
                    if not isinstance(r_item, dict):
                        continue
                    v = r_item.get("value")
                    if isinstance(v, str) and v.strip():
                        text_out = v.strip()
                        break
                    if isinstance(v, list):
                        for choice in v:
                            if isinstance(choice, dict):
                                texts = [c.get("text", "") for c in choice.get("choices", []) if c.get("text")]
                                if texts:
                                    text_out = " ".join(texts)
                                    break
                        if text_out:
                            break
                if text_out:
                    break
            if not img_ref or not text_out:
                continue
            local_img = None
            cand_paths = [img_ref, os.path.join(img_dir, os.path.basename(img_ref))]
            for c in cand_paths:
                if os.path.isfile(c):
                    local_img = c
                    break
            if not local_img:
                continue
            rows.append({
                "text": data.get("prompt", data.get("question", "Describe this image.")),
                "output": text_out,
                "image": local_img,
            })
        if not rows:
            raise RuntimeError(
                f"Built 0 VLM rows from {len(tasks)} tasks. "
                f"Check that LS annotations have a 'text' value and images are in the zip."
            )
        print(f"[train] Built {len(rows)} VLM conversation rows from project {project_id}")
    else:
        raise RuntimeError(
            "VLM training needs either text_dataset (JSONL) or a project "
            "(dataset_url). Got neither -- go back to step 1 and pick a project."
        )

    local_imgs = {}
    if img_dir and os.path.isdir(img_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for p in _glob.glob(os.path.join(img_dir, "**", ext), recursive=True):
                local_imgs[os.path.basename(p)] = p

    def _resolve_image(raw):
        if raw is None:
            return None
        if isinstance(raw, _Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, str):
            if raw.startswith("data:"):
                _, data = raw.split(",", 1)
                return _Image.open(_io.BytesIO(_b64.b64decode(data))).convert("RGB")
            if raw in local_imgs:
                return _Image.open(local_imgs[raw]).convert("RGB")
            if os.path.isfile(raw):
                return _Image.open(raw).convert("RGB")
            for ext in ("", ".jpg", ".jpeg", ".png", ".webp"):
                cand = os.path.join(img_dir or "", raw + ext) if img_dir else ""
                if cand and os.path.isfile(cand):
                    return _Image.open(cand).convert("RGB")
        return None

    def preprocess(examples):
        texts, images = [], []
        for i in range(len(examples.get("text", examples.get("instruction", [""]*len(examples))))):
            instr_key = "text" if "text" in examples else "instruction"
            output_key = "output" if "output" in examples else "response"
            instruction = examples[instr_key][i] if instr_key in examples else ""
            output = examples[output_key][i] if output_key in examples else ""
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n{output}"
            texts.append(prompt)
            img = _resolve_image(examples.get("image", [None] * len(texts))[i] if "image" in examples else None)
            if img is None:
                img = _Image.new("RGB", (224, 224), (128, 128, 128))
            images.append(img)
        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True, truncation=True, max_length=max_seq_len)
        enc["labels"] = enc["input_ids"].clone()
        return enc

    print(f"[train] Preprocessing {len(rows)} examples ...")
    ds = HFDataset.from_list(rows)
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


def train_unsloth_vlm(model_name, text_dataset, max_seq_len, lora_rank, quantization,
                       epochs, batch, grad_accum, lr, chat_template, job_id,
                       w_bucket, w_key, minio_url, minio_access, minio_secret,
                       dataset_url=None, project_id=0, img_dir=None):
    """Fine-tune a VLM via Unsloth's FastVisionModel. Kept for parity
    but NOT routed from main() — Unsloth's install tree conflicts with
    the cluster's torch, and the bare-HF path in train_vlm_finetune
    covers the same models plus several Unsloth doesn't support
    (InternVL2, MedGemma, SmolVLM). Switch the main() routing here
    only if you also pin unsloth to a torch version the cluster has
    and accept the unsloth install chain."""
    import os as _os
    import socket as _sock
    import torch as _torch_diag
    print(f"[train] Unsloth VLM -- model: {model_name}  lora_r={lora_rank}  4bit={quantization in ('4bit','4-bit','bnb-4bit')}")
    print(f"[train]   torch={_torch_diag.__version__}  "
          f"cuda_available={_torch_diag.cuda.is_available()}  "
          f"cuda_version={_torch_diag.version.cuda}  "
          f"device_count={_torch_diag.cuda.device_count()}")
    if _torch_diag.cuda.is_available():
        for _di in range(_torch_diag.cuda.device_count()):
            print(f"[train]   device[{_di}]={_torch_diag.cuda.get_device_name(_di)}")
    print(f"[train]   hostname={_sock.gethostname()}  "
          f"NVIDIA_VISIBLE_DEVICES={_os.environ.get('NVIDIA_VISIBLE_DEVICES', '(unset)')}  "
          f"CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    _sp_diag = subprocess.run(
        ["bash", "-lc", "nvidia-smi -L 2>&1 | head -3 || echo 'nvidia-smi not available'"],
        capture_output=True, text=True, timeout=10,
    )
    for _ln in (_sp_diag.stdout or "").splitlines():
        if _ln.strip():
            print(f"[train]   nvidia-smi: {_ln.strip()}")
    if not _torch_diag.cuda.is_available():
        raise RuntimeError(
            f"torch {_torch_diag.__version__} cannot see a GPU on "
            f"hostname={_sock.gethostname()}. The VLM torch pre-flight "
            f"should have fixed this; if it didn't, the cluster's GPU "
            f"driver is incompatible with every torch wheel in the "
            f"pre-flight list (cu117/118/121/124/126)."
        )
    from unsloth import FastVisionModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset, Dataset as HFDataset
    import torch as _torch
    import zipfile as _zf, glob as _glob, tempfile as _tf
    from PIL import Image as _Image
    import io as _io, base64 as _b64
    import requests as _req
    load_in_4bit = quantization in ("4bit", "4-bit", "bnb-4bit")

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=load_in_4bit,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        random_state=3407,
    )

    rows = None
    if text_dataset:
        print(f"[train] Loading text_dataset: {text_dataset[:80]}{'...' if len(text_dataset) > 80 else ''}")
        try:
            ds = load_dataset(text_dataset, split="train")
            rows = list(ds)
        except Exception as e:
            print(f"[warn] HF load failed ({e}), trying as JSONL")
            rows = [json.loads(l) for l in text_dataset.splitlines() if l.strip()]
    elif dataset_url:
        tmpdir = _tf.mkdtemp(prefix="vlm_data_")
        print(f"[train] Downloading dataset zip: {dataset_url}")
        r = _req.get(dataset_url, stream=True, timeout=600)
        r.raise_for_status()
        zip_path = os.path.join(tmpdir, "dataset.zip")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        with _zf.ZipFile(zip_path) as z:
            z.extractall(tmpdir)
        img_dir = img_dir or os.path.join(tmpdir, "images")
        ann_path = None
        for cand in (os.path.join(tmpdir, "annotations.json"),
                     os.path.join(tmpdir, "tasks.json"),
                     os.path.join(tmpdir, "data", "annotations.json")):
            if os.path.isfile(cand):
                ann_path = cand
                break
        if not ann_path:
            raise RuntimeError(
                f"No annotations.json / tasks.json in zip -- VLM needs image+text pairs. "
                f"Got: {os.listdir(tmpdir)[:20]}"
            )
        with open(ann_path) as f:
            tasks = json.load(f)
        print(f"[train] Building VLM rows from {len(tasks)} LS tasks in project {project_id} ...")
        rows = []
        for t in tasks:
            data = t.get("data") or {}
            anns = t.get("annotations") or t.get("predictions") or []
            img_ref = (data.get("image") or data.get("img")
                       or (data.get("upload") or {}).get("file") or "")
            text_out = ""
            for a in anns:
                res = a.get("result") or []
                for r_item in res:
                    if not isinstance(r_item, dict):
                        continue
                    v = r_item.get("value")
                    if isinstance(v, str) and v.strip():
                        text_out = v.strip()
                        break
                    if isinstance(v, list):
                        for choice in v:
                            if isinstance(choice, dict):
                                texts = [c.get("text", "") for c in choice.get("choices", []) if c.get("text")]
                                if texts:
                                    text_out = " ".join(texts)
                                    break
                        if text_out:
                            break
                if text_out:
                    break
            if not img_ref or not text_out:
                continue
            local_img = None
            cand_paths = [img_ref, os.path.join(img_dir, os.path.basename(img_ref))]
            for c in cand_paths:
                if os.path.isfile(c):
                    local_img = c
                    break
            if not local_img:
                continue
            rows.append({
                "text": data.get("prompt", data.get("question", "Describe this image.")),
                "output": text_out,
                "image": local_img,
            })
        if not rows:
            raise RuntimeError(
                f"Built 0 VLM rows from {len(tasks)} tasks. "
                f"Check that LS annotations have a 'text' value and images are in the zip."
            )
        print(f"[train] Built {len(rows)} VLM conversation rows from project {project_id}")
    else:
        raise RuntimeError(
            "VLM training needs either text_dataset (JSONL) or a project "
            "(dataset_url). Got neither -- go back to step 1 and pick a project."
        )

    local_imgs = {}
    if img_dir and os.path.isdir(img_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for p in _glob.glob(os.path.join(img_dir, "**", ext), recursive=True):
                local_imgs[os.path.basename(p)] = p

    def _resolve_image(raw):
        if raw is None:
            return None
        if isinstance(raw, _Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, str):
            if raw.startswith("data:"):
                _, data = raw.split(",", 1)
                return _Image.open(_io.BytesIO(_b64.b64decode(data))).convert("RGB")
            if raw in local_imgs:
                return _Image.open(local_imgs[raw]).convert("RGB")
            if os.path.isfile(raw):
                return _Image.open(raw).convert("RGB")
            for ext in ("", ".jpg", ".jpeg", ".png", ".webp"):
                cand = os.path.join(img_dir or "", raw + ext) if img_dir else ""
                if cand and os.path.isfile(cand):
                    return _Image.open(cand).convert("RGB")
        return None

    def to_conversation(example):
        instr = example.get("text") or example.get("instruction") or ""
        out   = example.get("output") or example.get("response") or ""
        img   = _resolve_image(example.get("image"))
        if img is None:
            img = _Image.new("RGB", (224, 224), (128, 128, 128))
        user_content = [{"type": "image"}, {"type": "text", "text": instr}]
        return {
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": out}]},
            ],
            "images": [img],
        }

    print(f"[train] Converting {len(rows)} examples to conversation format ...")
    ds = HFDataset.from_list(rows)
    ds = ds.map(to_conversation, remove_columns=ds.column_names)

    save_dir = f"/tmp/vlm_{job_id}"
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=save_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            fp16=not _torch.cuda.is_bf16_supported(),
            bf16=_torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="no",
            max_length=max_seq_len,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        ),
    )
    trainer.train()

    print("[train] Unsloth VLM training complete. Saving LoRA adapter ...")
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    zip_path = f"/tmp/vlm_{job_id}.zip"
    with _zf.ZipFile(zip_path, "w", _zf.ZIP_DEFLATED) as zf:
        for fp in Path(save_dir).rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(save_dir))
    upload_to_minio(zip_path, w_bucket, w_key, minio_url, minio_access, minio_secret)
    print(f"[train] WEIGHTS_UPLOADED: s3://{w_bucket}/{w_key}")


# ── entry point ───────────────────────────────────────────────────────────

def main():
    """VLM/LLM training entry point. Reads env vars, runs torch pre-flight,
    installs import stubs, then dispatches to one of the train functions.

    Env vars consumed (set by main.py:_run_on_ray_cluster before exec):
        ENGINE, TRAINING_TYPE, MODEL_NAME, EPOCHS, BATCH_SIZE, LR,
        JOB_ID, PROJECT_ID, DATASET_URL, WEIGHTS_BUCKET, WEIGHTS_KEY,
        MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, LS_PUBLIC_URL,
        LS_TOKEN, HF_TOKEN, HUGGING_FACE_HUB_TOKEN,
        TEXT_DATASET, LORA_RANK, QUANTIZATION, MAX_SEQ_LEN,
        CHAT_TEMPLATE, GRAD_ACCUM
    """
    engine        = os.environ.get("ENGINE", "Ultralytics")
    training_type = os.environ.get("TRAINING_TYPE", "detection")
    model_name    = os.environ.get("MODEL_NAME", "yolov8n.pt")
    epochs        = int(os.environ.get("EPOCHS", "10"))
    batch         = int(os.environ.get("BATCH_SIZE", "16"))
    lr            = float(os.environ.get("LR", "0.001"))
    job_id        = os.environ.get("JOB_ID", "unknown")
    project_id    = int(os.environ.get("PROJECT_ID", "0") or 0)
    dataset_url   = os.environ.get("DATASET_URL", "")
    w_bucket      = os.environ.get("WEIGHTS_BUCKET", "medimage-weights")
    w_key         = os.environ.get("WEIGHTS_KEY", f"{job_id}/best.pt")
    minio_url     = os.environ.get("MINIO_URL", "http://minio:9000")
    minio_access  = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret  = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    hf_token      = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    text_dataset  = os.environ.get("TEXT_DATASET", "")
    lora_rank     = int(os.environ.get("LORA_RANK", "16"))
    quantization  = os.environ.get("QUANTIZATION", "4bit")
    max_seq_len   = int(os.environ.get("MAX_SEQ_LEN", "2048"))
    chat_template = os.environ.get("CHAT_TEMPLATE", "chatml")
    grad_accum    = int(os.environ.get("GRAD_ACCUM", "4"))

    print(f"[vlm_train] Job {job_id} -- engine={engine} type={training_type} model={model_name} epochs={epochs}")

    # Torch pre-flight: only meaningful for VLM (which uses CUDA). The
    # LLM path goes through Unsloth which has its own torch handling.
    if training_type == "vlm-finetune":
        preflight_torch_for_vlm()

    # Install import stubs before any training code runs. The transformers
    # Trainer / bitsandbytes import chain pulls triton / sklearn, and the
    # cluster's partial triton install + pandas-numpy ABI mismatch will
    # crash on bare import. The stubs fix both.
    setup_import_stubs()

    if engine == "Unsloth" and training_type != "vlm-finetune":
        if not text_dataset:
            raise RuntimeError("TEXT_DATASET env var is required for LLM fine-tuning")
        train_unsloth_llm(
            model_name, text_dataset, max_seq_len, lora_rank, quantization,
            epochs, batch, grad_accum, lr, chat_template,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
        )
        print("[vlm_train] Complete!")
        return

    if training_type == "vlm-finetune":
        if not text_dataset and not (dataset_url and project_id):
            raise RuntimeError(
                "VLM fine-tuning needs either text_dataset (JSONL) or a "
                "project (dataset_url). Go back to step 1 and pick a "
                "project, or upload a text dataset in Datasets."
            )
        train_vlm_finetune(
            model_name, text_dataset, max_seq_len, lora_rank, quantization,
            epochs, batch, grad_accum, lr, chat_template,
            job_id, w_bucket, w_key, minio_url, minio_access, minio_secret,
            dataset_url=dataset_url, project_id=project_id,
        )
        print("[vlm_train] Complete!")
        return

    raise RuntimeError(
        f"vlm_llm_train.main() called with engine={engine} type={training_type} "
        f"-- this entry point only handles Unsloth (LLM) and vlm-finetune (VLM). "
        f"Other engines use _VISION_TRAIN_SCRIPT in main.py."
    )


# ── Ray-remote wrapper ────────────────────────────────────────────────────
#
# Cloudpickle can't serialise a top-level function reference by name to
# the Ray worker (the worker's `main` module is a different process).
# We define the Ray-remote body inline in main.py:_run_on_ray_cluster
# so the full closure (including the imports from this module) is
# captured. The wrapper just delegates to main().

def run(env_vars):
    """Synchronous entry point for use from inside the Ray-remote
    closure. Sets env vars, captures stdout, runs main(), returns
    {stdout, weights_path}.

    `env_vars` is a dict of training parameters set by
    main.py:_run_on_ray_cluster.
    """
    from io import StringIO
    for k, v in env_vars.items():
        os.environ[k] = str(v)
    captured = StringIO()
    _orig_out, _orig_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = captured
    try:
        main()
    finally:
        sys.stdout, sys.stderr = _orig_out, _orig_err
    output = captured.getvalue()
    weights_path = next(
        (ln.split(":", 1)[1].strip()
         for ln in output.splitlines()
         if ln.startswith("WEIGHTS_UPLOADED:")),
        None,
    )
    return {"stdout": output, "weights_path": weights_path}
