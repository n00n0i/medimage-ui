#!/bin/bash
# GPU utilization updater for medimage-ui
# Runs nvidia-smi on GPU worker nodes via SSH and pushes per-node, per-GPU
# data to the API.
#
# nvidia-smi's `utilization.gpu` is a 1-second rolling window that lands on 0
# between kernels (data loader, sync). We sample N times per poll and take
# the max of (compute, memory-controller) util to capture peak activity.
#
# Usage:
#   1. Install on a machine with SSH access to GPU worker nodes
#   2. Set environment variables:
#      GPU_UTIL_API_URL   - API endpoint URL (default: https://localhost:8083/api/gpu-util)
#      GPU_NODES          - comma-separated worker node IPs/hostnames
#      GPU_UTIL_SAMPLES   - nvidia-smi samples per poll (default: 4)
#      GPU_UTIL_SAMPLE_DELAY - delay between samples, seconds (default: 0.25)
#   3. Run as cron: * * * * * /path/to/gpu-util-updater.sh
#   Or in a loop: while true; do /path/to/gpu-util-updater.sh; sleep 10; done

API_URL="${GPU_UTIL_API_URL:-https://localhost:8083/api/gpu-util}"
GPU_NODES="${GPU_NODES:-172.16.60.119,100.68.245.202,100.68.162.147}"
N_SAMPLES="${GPU_UTIL_SAMPLES:-4}"
SAMPLE_DELAY="${GPU_UTIL_SAMPLE_DELAY:-0.25}"

# Build a small awk program that, for each GPU index, tracks the max of
# (utilization.gpu, utilization.memory) across N_SAMPLES input rows.
# Each input row is: idx, util_gpu, util_mem
awk_max_util=$(cat <<'AWK'
{
    idx = $1
    g = $2 + 0
    m = $3 + 0
    v = (g > m ? g : m)
    if (v > max[idx]) max[idx] = v
}
END {
    for (i in max) print i, max[i]
}
AWK
)

cluster_total=0
cluster_count=0
gpus_json=""

IFS=',' read -ra NODES <<< "$GPU_NODES"
for node in "${NODES[@]}"; do
    # Collect N samples. Each sample is 3 columns: idx, util.gpu, util.memory.
    # We SSH once per sample, parse the CSV, feed the awk program.
    samples=$(for ((s=0; s<N_SAMPLES; s++)); do
        ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$node" \
            "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv,noheader,nounits" 2>/dev/null
        sleep "$SAMPLE_DELAY"
    done)

    if [ -z "$samples" ]; then
        continue
    fi

    # Per-node util: average of each GPU's peak across samples.
    peak_per_gpu=$(echo "$samples" | awk "$awk_max_util")
    if [ -z "$peak_per_gpu" ]; then
        continue
    fi

    # Latest static fields (name, mem totals, temp) for per-GPU detail.
    latest=$(echo "$samples" | tail -n 1)
    # power draw is fetched once (mixing it into the loop is slow + power
    # doesn't oscillate as wildly as utilization).
    power=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$node" \
        "nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits" 2>/dev/null)

    # Build a JSON array of per-GPU objects for this node.
    node_gpus="["
    first=1
    while read -r idx peak; do
        # Find static fields for this idx from the latest sample.
        # The latest sample only has idx, util.gpu, util.mem — we need the
        # full row, so we use the *first* sample's static fields instead.
        first_line=$(echo "$samples" | awk -F',' -v want="$idx" '$1 == want {print; exit}' | head -1)
        if [ -z "$first_line" ]; then continue; fi
        mem_used=$(echo "$first_line" | awk -F',' '{print $4}' | tr -d '[:space:]')
        mem_total=$(echo "$first_line" | awk -F',' '{print $5}' | tr -d '[:space:]')
        name=$(echo "$first_line" | awk -F',' '{print $2}' | sed 's/^ *//; s/ *$//')
        pwr=$(echo "$power" | awk -F',' -v want="$idx" '$1 == want {print $2}' | tr -d '[:space:]')
        [ -z "$pwr" ] && pwr="0"

        if [ $first -eq 0 ]; then node_gpus+=","; fi
        first=0
        node_gpus+=$(printf '{"index":%s,"name":"%s","util_pct":%s,"mem_used_mb":%s,"mem_total_mb":%s,"temp_c":0,"power_w":%s}' \
            "$idx" "$name" "$peak" "$mem_used" "$mem_total" "$pwr")

        cluster_total=$((cluster_total + peak))
        cluster_count=$((cluster_count + 1))
    done <<< "$peak_per_gpu"
    node_gpus+="]"

    # Push this node's data to the API (preserves per-node attribution).
    curl -sk -X PUT "$API_URL" -H "Content-Type: application/json" \
        -d "$(printf '{"hostname":"%s","gpu_util_pct":%s,"gpus":%s}' \
            "$node" "$(echo "$peak_per_gpu" | awk '{s+=$2; c++} END {if(c>0) printf "%.1f", s/c; else print 0}')" \
            "$node_gpus")" > /dev/null 2>&1
done

# Cluster-wide average fallback for legacy consumers.
if [ $cluster_count -gt 0 ]; then
    avg_util=$(echo "scale=1; $cluster_total / $cluster_count" | bc 2>/dev/null || echo "0")
    curl -sk -X PUT "$API_URL" -H "Content-Type: application/json" \
        -d "{\"gpu_util_pct\": $avg_util}" > /dev/null 2>&1
fi
