#!/usr/bin/env python3
"""Push nvidia-smi data to medimage-ui API from inside Ray cluster."""
import urllib.request, ssl, json, subprocess, socket, os, time

API_URL = os.getenv("GPU_UTIL_API_URL", "https://100.68.53.118:8083/api/gpu-util")
INTERVAL = int(os.getenv("GPU_UTIL_INTERVAL", "10"))

# Sample nvidia-smi multiple times per poll. `utilization.gpu` is a 1-second
# rolling window that lands on 0 between kernels (data loader, sync, CPU
# prep). Taking max(compute, memory-controller) across multiple samples
# catches both kernel bursts and memory-bound phases.
N_SAMPLES = int(os.getenv("GPU_UTIL_SAMPLES", "4"))
SAMPLE_DELAY_S = float(os.getenv("GPU_UTIL_SAMPLE_DELAY", "0.25"))


def _parse_line(line):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 7 or not p[0].isdigit() or not p[2].isdigit() or not p[3].isdigit() or not p[4].isdigit():
        return None
    return {
        "index": int(p[0]),
        "name": p[1],
        "util_pct": int(p[2]),
        "mem_util_pct": int(p[3]),
        "mem_used_mb": int(p[4]),
        "mem_total_mb": int(p[5]),
        "temp_c": int(p[6]) if p[6].isdigit() else 0,
    }


def push():
    hostname = socket.gethostname()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        samples_by_idx: dict[int, list[dict]] = {}
        for _ in range(N_SAMPLES):
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    parsed = _parse_line(line)
                    if parsed is not None:
                        samples_by_idx.setdefault(parsed["index"], []).append(parsed)
            time.sleep(SAMPLE_DELAY_S)

        if not samples_by_idx:
            print("[gpu-monitor] No GPU data (nvidia-smi not available?)")
            return False

        # Power draw is fetched once per GPU (mixing it into the multi-sample
        # loop would slow the loop down — power doesn't oscillate as wildly).
        power_by_idx: dict[int, float] = {}
        try:
            pr = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if pr.returncode == 0:
                for line in pr.stdout.strip().splitlines():
                    pp = [x.strip() for x in line.split(",")]
                    if len(pp) >= 2 and pp[0].isdigit():
                        try:
                            power_by_idx[int(pp[0])] = float(pp[1])
                        except ValueError:
                            pass
        except Exception:
            pass

        gpus = []
        for idx, samples in samples_by_idx.items():
            # util = max(compute, memory-controller) over all samples —
            # catches both kernel bursts and memory-bound data-loader phases.
            util_max = max(max(s["util_pct"], s["mem_util_pct"]) for s in samples)
            latest = samples[-1]
            gpus.append({
                "index": idx,
                "name": latest["name"],
                "util_pct": util_max,
                "mem_used_mb": latest["mem_used_mb"],
                "mem_total_mb": latest["mem_total_mb"],
                "temp_c": latest["temp_c"],
                "power_w": power_by_idx.get(idx, 0.0),
            })

        avg = sum(g["util_pct"] for g in gpus) / len(gpus) if gpus else 0
        data = json.dumps({"gpu_util_pct": round(avg, 1), "hostname": hostname, "gpus": gpus}).encode()
        req = urllib.request.Request(API_URL, data=data, method="PUT")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            print(f"[gpu-monitor] Pushed {len(gpus)} GPUs from {hostname}, avg={avg:.1f}%, status={resp.status}")
            return True
    except Exception as e:
        print(f"[gpu-monitor] Error: {e}")
        return False

if __name__ == "__main__":
    once = os.getenv("GPU_UTIL_ONCE", "")
    if once:
        push()
    else:
        print(f"[gpu-monitor] Starting — API: {API_URL}, interval: {INTERVAL}s")
        while True:
            push()
            time.sleep(INTERVAL)