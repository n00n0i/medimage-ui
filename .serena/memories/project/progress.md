# medimage-ui Project Progress

## Architecture
- Backend: FastAPI (Python) at port 8000
- Frontend: React/TypeScript + Vite
- Docker: medimage-api, medimage-ui (nginx), minio, label-studio, jupyter
- Ray cluster: 100.68.53.118:8265 (3 nodes, 4×H200 GPUs, head=2 GPUs, 2 workers=1 GPU each)
- nginx proxies: `/api/` → backend, `/api/ray/` → Ray dashboard, `/api/ls/` → Label Studio, `/api/minio/` → MinIO

## Key Bug Fixes Completed
1. **`/api/deploy` endpoint** — added missing route
2. **`num_gpus` field** — added to TrainRequest, DB, INSERT, Ray payload
3. **VALUES placeholder count** — fixed 28→26
4. **Bulk state persistence** — module-level + localStorage
5. **`moduleRuns` persistence** — localStorage for terminal runs
6. **Deploy with jobId** — deployModel() passes jobId to /api/deploy
7. **Training logs** — final.log appended to logLines
8. **GpuMonitor** — uses /api/ray/gpu-stats
9. **Dashboard** — GPU stats fetch
10. **RayCluster** — GPU stats, graph colors (CPU=#60a5fa, GPU=#f97316, MEM=#a78bfa)
11. **GPU Util input** — removed disabled, default '1'
12. **Test-all stream legend** — train ✓, deploying, deployed ✓, error
13. **Public prefixes** — /api/ray/gpu-stats, /api/ray/api/, /api/ray/nodes

## GPU Stats Architecture (Final)
- `/api/ray/gpu-stats` endpoint parses **Ray Dashboard API** (not nvidia-smi)
  - `cluster_status` → GPU allocation (4/4 GPUs used)
  - `nodes?view=summary` → per-node GPU workers, CPU%, memory
  - Returns: `{gpus_used, gpus_total, nodes: [{hostname, ip, cpu_pct, mem, gpus_allocated}], cluster: {gpu_utilization_pct}}`
- nvidia-smi approach abandoned because:
  - Head node has no NVML access
  - `entrypoint_num_gpus` jobs stay PENDING (all 4 GPUs allocated)
  - Without GPU requirement, job runs on head node (no GPUs)
- nginx config: `location = /api/ray/gpu-stats` routes to backend (before `/api/ray/` catch-all)

## Current Issue
- GPU utilization shows 100% (allocation-based: 4/4 GPUs allocated)
- Actual compute utilization unknown without nvidia-smi on GPU nodes
- Ray `nodes?view=summary` returns `gpus:[]` — no per-GPU data

## RayCluster.tsx Node Cards
- GPU data injected from /api/ray/gpu-stats into nodes
- Each node shows `gpus_allocated` GPUs with "H200" label
- GPU utilization bar uses cluster-level `gpu_utilization_pct`

## GpuMonitor.tsx
- Creates virtual GPU entries from node data (gpus_allocated per node)
- H200 GPU, 80GB VRAM assumed
- Shows cluster-level utilization percentage

## Files Modified
- `backend/main.py`: /api/deploy, /api/ray/gpu-stats, VALUES fix, num_gpus, public prefixes
- `src/pages/TestAllModels.tsx`: Bulk state, localStorage, stream legend, deployModel with jobId
- `src/pages/RayCluster.tsx`: GPU stats, metric graph, node card GPU injection
- `src/pages/Dashboard.tsx`: GPU stats fetch
- `src/components/GpuMonitor.tsx`: Rewritten for new API format
- `vite.config.ts`: Proxy config
- `nginx.conf`: GPU stats location before Ray catch-all