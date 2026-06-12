#!/usr/bin/env python3
"""GPU utilization monitor — pushes nvidia-smi data to medimage-ui API.

Install on each GPU worker node (NOT in Docker — run on the host):
  1. Copy this script to the GPU worker node
  2. Run: python3 gpu-monitor.py
  3. Or set up as systemd service (see below)

Environment variables:
  GPU_UTIL_API_URL  – API endpoint (default: http://100.68.3.42:8083/api/gpu-util)
  GPU_UTIL_INTERVAL – Poll interval in seconds (default: 10)
  GPU_HOSTNAME      – Override hostname sent to API (default: auto-detect)

Systemd service (recommended for production):
  sudo tee /etc/systemd/system/gpu-monitor.service << 'EOF'
  [Unit]
  Description=GPU Utilization Monitor for medimage-ui
  After=network.target

  [Service]
  Type=simple
  ExecStart=/usr/bin/python3 /opt/gpu-monitor/gpu-monitor.py
  Restart=always
  RestartSec=5
  Environment=GPU_UTIL_API_URL=http://100.68.3.42:8083/api/gpu-util
  Environment=GPU_UTIL_INTERVAL=10

  [Install]
  WantedBy=multi-user.target
  EOF

  sudo systemctl daemon-reload
  sudo systemctl enable --now gpu-monitor
"""
import os, time, json, subprocess, socket, urllib.request, ssl

API_URL = os.getenv("GPU_UTIL_API_URL", "http://100.68.3.42:8083/api/gpu-util")
INTERVAL = int(os.getenv("GPU_UTIL_INTERVAL", "10"))
HOSTNAME = os.getenv("GPU_HOSTNAME") or socket.gethostname()

# nvidia-smi's `utilization.gpu` is a 1-second sample that lands on 0 whenever
# the GPU is mid-batch (data loader, gradient sync, CPU-side prep). To avoid
# flatlining during real training, sample several times per poll and take the
# MAX of (compute util, memory-controller util) — this catches both compute-
# bound kernels and memory-bound data-loader phases.
N_SAMPLES = int(os.getenv("GPU_UTIL_SAMPLES", "4"))
SAMPLE_DELAY_S = float(os.getenv("GPU_UTIL_SAMPLE_DELAY", "0.25"))


def _parse_smi_line(line, gpu_count):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7 or not parts[0].isdigit() or not parts[2].isdigit() or not parts[6].isdigit():
        return None
    return {
        "index": int(parts[0]),
        "name": parts[1],
        "util_pct": int(parts[2]),
        "mem_util_pct": int(parts[3]),
        "mem_used_mb": int(parts[4]),
        "mem_total_mb": int(parts[5]),
        "temp_c": int(parts[6]),
    }


def get_gpu_stats():
    try:
        # Collect multiple samples so we can take peak activity across the
        # whole poll window. nvidia-smi's util.gpu is a 1s rolling window, so
        # sampling every 250ms over ~1s gives 4 chances to catch a real kernel.
        samples_by_idx: dict[int, list[dict]] = {}
        static_by_idx: dict[int, dict] = {}

        for _ in range(N_SAMPLES):
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    parsed = _parse_smi_line(line, len(samples_by_idx))
                    if parsed is None:
                        continue
                    samples_by_idx.setdefault(parsed["index"], []).append(parsed)
                    static_by_idx[parsed["index"]] = parsed
            time.sleep(SAMPLE_DELAY_S)

        if not samples_by_idx:
            return None

        gpus = []
        for idx, samples in samples_by_idx.items():
            # util = max(compute util, memory util) across all samples —
            # catches both kernel bursts and memory-bound data-loader phases.
            util_max = max(max(s["util_pct"], s["mem_util_pct"]) for s in samples)
            # Pick the latest sample for static fields (name, total mem, temp).
            latest = samples[-1]
            # Power draw is fetched separately since mixing it into the multi-
            # sample query would slow the loop. One call is fine — power
            # doesn't oscillate as wildly as utilization.
            try:
                pr = subprocess.run(
                    ["nvidia-smi",
                     f"--query-gpu=power.draw", "--format=csv,noheader,nounits",
                     f"-i={idx}"],
                    capture_output=True, text=True, timeout=5,
                )
                power_w = float(pr.stdout.strip()) if pr.returncode == 0 else 0.0
            except Exception:
                power_w = 0.0

            gpus.append({
                "index": idx,
                "name": latest["name"],
                "util_pct": util_max,
                "mem_used_mb": latest["mem_used_mb"],
                "mem_total_mb": latest["mem_total_mb"],
                "temp_c": latest["temp_c"],
                "power_w": power_w,
            })
        return gpus
    except Exception as e:
        print(f"[gpu-monitor] nvidia-smi error: {e}")
        return None

def push(data):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        payload = json.dumps(data).encode()
        req = urllib.request.Request(API_URL, data=payload, method="PUT")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[gpu-monitor] push error: {e}")
        return False

def main():
    print(f"[gpu-monitor] Starting — API: {API_URL}, interval: {INTERVAL}s, hostname: {HOSTNAME}")
    while True:
        gpus = get_gpu_stats()
        if gpus:
            avg_util = sum(g["util_pct"] for g in gpus) / len(gpus)
            payload = {
                "gpu_util_pct": round(avg_util, 1),
                "hostname": HOSTNAME,
                "gpus": gpus,
            }
            if push(payload):
                print(f"[gpu-monitor] Pushed: {avg_util:.1f}% avg across {len(gpus)} GPUs")
            else:
                print("[gpu-monitor] Push failed")
        else:
            print("[gpu-monitor] No GPU data (nvidia-smi not available?)")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()