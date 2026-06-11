import { useState, useEffect, useCallback, useRef, useMemo } from 'react'

// ─── Module-level background store ────────────────────────────────────────────
//
// The Test All page can spend 30+ minutes running 30+ training jobs
// sequentially. We don't want to:
//
//   • stop polling when the user navigates to /jobs or /models
//   • lose the run state if the user reloads the page
//   • duplicate a poll loop every time the component remounts
//
// So we keep the canonical state and the poll loop at module scope.
// Components subscribe to changes and render the latest snapshot.

interface ActiveJobRef {
  key: string
  jobId: string
  startedAt: number
}

const LS_ACTIVE_JOBS = 'medimage.testAll.activeJobs.v1'

const activeJobs = new Map<string, ActiveJobRef>()           // jobId → ref
const subscribers = new Set<() => void>()
let pollInterval: number | null = null

function loadPersistedJobs() {
  try {
    const raw = localStorage.getItem(LS_ACTIVE_JOBS)
    if (!raw) return
    const list: ActiveJobRef[] = JSON.parse(raw)
    for (const r of list) activeJobs.set(r.jobId, r)
  } catch { /* ignore */ }
}
function persistJobs() {
  try {
    localStorage.setItem(LS_ACTIVE_JOBS, JSON.stringify([...activeJobs.values()]))
  } catch { /* ignore */ }
}
loadPersistedJobs()

function notify() { for (const cb of subscribers) cb() }

function trackJob(key: string, jobId: string) {
  activeJobs.set(jobId, { key, jobId, startedAt: Date.now() / 1000 })
  persistJobs()
  ensurePolling()
  notify()
}

function untrackJob(jobId: string) {
  activeJobs.delete(jobId)
  persistJobs()
  if (activeJobs.size === 0 && pollInterval !== null) {
    window.clearInterval(pollInterval)
    pollInterval = null
  }
  notify()
}

function isJobTracked(jobId: string): boolean {
  return activeJobs.has(jobId)
}
void isJobTracked  // reserved for future "is running elsewhere" badge

function getTrackedJobs(): ActiveJobRef[] {
  return [...activeJobs.values()]
}

async function pollOnce() {
  const refs = getTrackedJobs()
  if (refs.length === 0) return
  await Promise.all(refs.map(async ref => {
    try {
      const res = await fetch(`/api/jobs/${ref.jobId}`)
      if (!res.ok) return
      const j = await res.json()
      const status: TestRun['status'] =
        j.status === 'completed' ? 'completed' :
        j.status === 'error'     ? 'error' :
        j.status === 'running'   ? 'running' : 'queued'
      const finishedAt = (j.status === 'completed' || j.status === 'error')
        ? (j.finished_at ?? Date.now() / 1000) : null
      updateRunFromBackend(ref.key, {
        jobId:      ref.jobId,
        status,
        error:      j.status === 'error' ? (j.error || 'Unknown error') : null,
        finishedAt,
        log:        j.log,
      })
      if (status === 'completed' || status === 'error') {
        untrackJob(ref.jobId)
      }
    } catch { /* ignore */ }
  }))
}

function ensurePolling() {
  if (pollInterval !== null) return
  pollInterval = window.setInterval(pollOnce, 3000)
  // Kick off an immediate poll so the UI updates without waiting 3s
  void pollOnce()
}
import {
  FlaskConical, Play, RefreshCw, Loader2, CheckCircle2, XCircle, Clock,
  Terminal, ChevronDown, ChevronUp, Search, Cpu, MessageSquare, Eye, EyeOff,
  Database, AlertCircle, StopCircle, Filter, Copy, Check, Download, X,
  Server, Cloud, Rocket,
} from 'lucide-react'
import { TRAINING_MATRIX } from './TrainModel'

// ─── Types ────────────────────────────────────────────────────────────────────

type TrainingType = 'classification' | 'detection' | 'segmentation' | 'vlm-finetune' | 'self-supervised' | 'llm-text'

interface TrainOption {
  value: string
  label: string
  engine: string
  model: string
  hardware: string
  description: string
  compatible: boolean
  zeroShot?: boolean
}

interface ModelRow {
  key: string
  trainingType: TrainingType
  option: TrainOption
}

interface LSProject {
  id: number
  title: string
  task_number: number
  finished_task_number: number
}

interface TextDataset {
  id: string
  name: string
  format: string
  row_count: number
}

interface TestRun {
  key: string
  jobId: string | null
  status: 'idle' | 'submitting' | 'queued' | 'running' | 'completed' | 'error'
  error: string | null
  startedAt: number | null
  finishedAt: number | null
  log: string
  datasetLabel: string
}

// Bulk run report — a snapshot of the last completed "Run all compatible"
// pass so the user can see at a glance which models failed and why,
// then export the failure list for follow-up fixes.
type BulkStage = 'train' | 'deploy'

interface BulkFailure {
  key:          string         // row key, e.g. 'classification:medr1-3b'
  label:        string         // human label, e.g. 'Med-R1-3B (GRPO Medical VLM)'
  trainingType: TrainingType
  engine:       string
  model:        string
  error:        string         // short error from backend (last 500 chars)
  jobId:        string | null
  stage:        BulkStage       // which step failed
}

interface BulkDeployed {
  key:        string
  label:      string
  provider:   'ray' | 'modal'
  url:        string           // endpoint URL returned by /api/deploy
  elapsedSec: number
}

interface BulkReport {
  startedAt:     number          // unix seconds
  finishedAt:    number          // unix seconds
  total:         number          // models attempted
  ok:            number          // training completed
  deployedCount: number          // successfully deployed after training
  failed:        number          // error (training OR deploy)
  stopped:       boolean         // user stopped early
  provider:      'ray' | 'modal' | null   // null if user didn't pick
  deployEnabled: boolean         // whether train → deploy chain was active
  failures:      BulkFailure[]
  deployed:      BulkDeployed[]  // successfully deployed
  log:           string          // human-readable full run log (line per model)
}

// Module-level runs map. Components subscribe to changes via
// useStoreSnapshot(). Kept outside React state so the polling loop
// at module scope can mutate it without going through setState.
const moduleRuns: Record<string, TestRun> = {}

function getRun(key: string): TestRun | undefined {
  return moduleRuns[key]
}
void getRun  // reserved for external introspection (jobs page badge)

function setRunLocal(key: string, run: TestRun) {
  moduleRuns[key] = run
  notify()
}

function updateRunFromBackend(key: string, patch: Partial<TestRun> & { jobId: string }) {
  const cur = moduleRuns[key]
  // Strip the key/jobId from the spread so the explicit fields below
  // aren't silently overwritten.
  const { jobId, ...rest } = patch
  if (!cur) {
    moduleRuns[key] = {
      key,
      jobId,
      status:     'submitting',
      error:      null,
      startedAt:  null,
      finishedAt: null,
      log:        '',
      datasetLabel: '',
      ...rest,
    }
  } else {
    moduleRuns[key] = { ...cur, jobId, ...rest }
  }
  notify()
}

function useStoreSnapshot(): Record<string, TestRun> {
  const [, force] = useState(0)
  useEffect(() => {
    const cb = () => force(n => n + 1)
    subscribers.add(cb)
    // Also re-hydrate any persisted jobs the first time we mount.
    ensurePolling()
    return () => { subscribers.delete(cb) }
  }, [])
  return moduleRuns
}

const TYPE_LABELS: Record<TrainingType, string> = {
  'classification':    'Classification',
  'detection':         'Detection',
  'segmentation':      'Segmentation',
  'llm-text':          'LLM',
  'vlm-finetune':      'VLM',
  'self-supervised':   'Self-Sup',
}

const TYPE_COLORS: Record<TrainingType, string> = {
  'classification':    '#6366f1',
  'detection':         '#f59e0b',
  'segmentation':      '#10b981',
  'llm-text':          '#8b5cf6',
  'vlm-finetune':      '#3b82f6',
  'self-supervised':   '#14b8a6',
}

const TYPE_ORDER: TrainingType[] = ['classification', 'detection', 'segmentation', 'llm-text', 'vlm-finetune', 'self-supervised']

const LS_TOKEN = 'medimage-ls-token-2026'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function isLlmType(t: TrainingType)   { return t === 'llm-text' || t === 'vlm-finetune' }
function needsTextDataset(t: TrainingType) { return t === 'llm-text' }

function formatElapsed(start: number | null, end: number | null): string {
  if (!start) return '—'
  const ms = ((end ?? Date.now() / 1000) - start) * 1000
  if (ms < 1000) return `${Math.round(ms)}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  const r = Math.floor(s % 60)
  return `${m}m ${r}s`
}

// ─── Main component ──────────────────────────────────────────────────────────

export default function TestAllModels() {
  // Flatten TRAINING_MATRIX into rows. No useMemo: Vite HMR replaces the
  // TRAINING_MATRIX const but doesn't bump a React-tracked reference, so a
  // useMemo([]) would freeze the row list at first mount and the new
  // medical VLMs added in dev would never appear. The matrix is tiny
  // (~50 entries) — recomputing on every render is free.
  const allRows: ModelRow[] = []
  for (const t of TYPE_ORDER) {
    const opts = (TRAINING_MATRIX[t] || []) as TrainOption[]
    for (const opt of opts) allRows.push({ key: `${t}:${opt.value}`, trainingType: t, option: opt })
  }

  // Datasets
  const [lsProjects, setLsProjects]   = useState<LSProject[]>([])
  const [textDatasets, setTextDatasets] = useState<TextDataset[]>([])
  const [datasetsLoading, setDatasetsLoading] = useState(true)

  // User overrides per model row
  const [overrides, setOverrides] = useState<Record<string, number | string | null>>({})
  const [useDefault, setUseDefault] = useState<Record<string, boolean>>({})

  // Filter
  const [typeFilter, setTypeFilter] = useState<'all' | TrainingType>('all')
  const [search, setSearch]         = useState('')
  // Show incompatible by default — users typically want to see every
  // catalog model in the bulk-test view. The "Run N Compatible" button
  // already uses the same gpuFreeGb-aware isCompatible function below.
  const [showIncompatible, setShowIncompatible] = useState(true)

  // Mirror TrainModel's dynamic GPU check: pull available system RAM
  // from the Ray cluster summary and use it as a proxy for free GPU
  // memory. With 4×H200 @ 50% util (~282 GB free), this lets every
  // catalog model with `hardware ≤ 282 GB` pass without a stale static
  // `compatible: false` blocking it.
  const [gpuFreeGb, setGpuFreeGb] = useState<number | null>(null)
  useEffect(() => {
    fetch('/api/ray/nodes?view=summary')
      .then(r => r.json())
      .then(data => {
        const nodes: any[] = data?.data?.summary ?? []
        let maxFreeGb = 0
        for (const node of nodes) {
          if (Array.isArray(node.mem) && node.mem.length >= 2) {
            const availBytes = Number(node.mem[1] ?? 0)
            if (availBytes > 0) {
              const availGb = availBytes / (1024 ** 3)
              if (availGb > maxFreeGb) maxFreeGb = availGb
            }
          }
        }
        if (maxFreeGb > 0) setGpuFreeGb(Math.floor(maxFreeGb))
        else setGpuFreeGb(64)  // généreus fallback
      })
      .catch(() => setGpuFreeGb(64))
  }, [])

  function parseRequiredGb(hardware: string): number {
    const m = hardware.match(/(\d+)\s*GB/i)
    return m ? parseInt(m[1]) : 0
  }

  function isCompatible(opt: TrainOption): boolean {
    if (gpuFreeGb !== null) return parseRequiredGb(opt.hardware) <= gpuFreeGb
    return opt.compatible
  }

  // Test runs (keyed by model row key) — backed by module store so the
  // poll loop keeps working even when the user navigates away.
  const runs = useStoreSnapshot()
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  // Bulk run control — also module-scoped so the bulk loop survives
  // navigation. We mirror into useState for re-render.
  const [bulkRunning, setBulkRunning] = useState(false)
  const [bulkStopRequested, setBulkStopRequested] = useState(false)
  const [bulkProgress, setBulkProgress] = useState<{ current: number; total: number; label: string; stage: 'train' | 'deploy' } | null>(null)
  // Snapshot of the last completed bulk run — surfaces failed models
  // + a copyable/exportable log so the user can follow up on each error.
  const [bulkReport, setBulkReport] = useState<BulkReport | null>(null)
  // Live stream of failures + statuses while the run is in progress, so
  // the user sees what's failing in real time instead of waiting for the
  // whole pass to finish.
  const [bulkStream, setBulkStream] = useState<{ label: string; status: 'ok' | 'err' | 'deploy' | 'deployed'; error?: string }[]>([])

  // Pre-run config — picked once before the bulk loop starts
  const [bulkProvider, setBulkProvider] = useState<'ray' | 'modal'>('ray')
  const [bulkDeployEnabled, setBulkDeployEnabled] = useState(true)

  // Keep latest runs in a ref so the bulk loop always reads the current
  // state instead of the snapshot from when runAllCompatible was called.
  const runsRef = useRef(runs)
  useEffect(() => { runsRef.current = runs }, [runs])

  // ── Load datasets (image + text) and ensure a sample text dataset
  //    exists for LLM/VLM smoke testing ──
  useEffect(() => {
    let cancelled = false
    async function load() {
      setDatasetsLoading(true)
      try {
        // Fetch current state of both
        let projects: LSProject[] = []
        let textList: TextDataset[] = []
        const [pRes, tRes] = await Promise.all([
          fetch('/api/ls/projects/?page_size=1000', { headers: { Authorization: `Token ${LS_TOKEN}` } }).then(r => r.json()),
          fetch('/api/text-datasets').then(r => r.json()),
        ])
        projects = Array.isArray(pRes) ? pRes : (pRes?.results ?? [])
        textList = tRes?.datasets ?? []

        // If there is no text dataset at all, auto-create an alpaca-format
        // sample so LLM smoke tests can run without manual setup.
        if (textList.length === 0) {
          try {
            const created = await fetch('/api/text-datasets/create-sample', {
              method:  'POST',
              headers: { 'Content-Type': 'application/json' },
              body:    JSON.stringify({ format: 'alpaca', name: 'sample-alpaca', rows: 20 }),
            })
            if (created.ok) {
              const d = await created.json()
              textList = [{
                id: d.id, name: d.name, format: d.format, row_count: d.row_count,
              }]
            }
          } catch { /* ignore — user can create manually */ }
        }

        if (cancelled) return
        setLsProjects(projects)
        setTextDatasets(textList)
      } catch (e) {
        // ignore
      } finally {
        if (!cancelled) setDatasetsLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  // ── Auto-suggest dataset per row ──
  const suggestedDataset = useCallback((row: ModelRow): number | string | null => {
    if (needsTextDataset(row.trainingType)) {
      // Prefer an alpaca-format dataset (works with all LLM models).
      // Fall back to sharegpt, then any text dataset.
      const alpaca = textDatasets.find(d => d.format === 'alpaca')
      if (alpaca) return alpaca.id
      const sharegpt = textDatasets.find(d => d.format === 'sharegpt')
      if (sharegpt) return sharegpt.id
      return textDatasets[0]?.id ?? null
    }
    // Image: pick the project with the most labeled tasks so the smoke
    // test has real data to iterate on.
    if (lsProjects.length === 0) return null
    const ranked = [...lsProjects].sort((a, b) => b.finished_task_number - a.finished_task_number)
    return ranked[0]?.id ?? lsProjects[0]?.id ?? null
  }, [lsProjects, textDatasets])

  // ── Filtered rows ──
  const filteredRows = useMemo(() => {
    let rows = allRows
    if (typeFilter !== 'all') rows = rows.filter(r => r.trainingType === typeFilter)
    if (!showIncompatible) rows = rows.filter(r => isCompatible(r.option))
    if (search.trim()) {
      const q = search.toLowerCase()
      rows = rows.filter(r =>
        r.option.label.toLowerCase().includes(q) ||
        r.option.model.toLowerCase().includes(q) ||
        r.option.engine.toLowerCase().includes(q)
      )
    }
    return rows
  }, [allRows, typeFilter, search, showIncompatible, gpuFreeGb])

  // ── Effective dataset for a row ──
  const effectiveDataset = (row: ModelRow): { id: number | string | null; label: string; source: 'default' | 'user' | 'none' } => {
    if (useDefault[row.key] !== false) {
      const s = suggestedDataset(row)
      if (s == null) return { id: null, label: '—', source: 'none' }
      if (needsTextDataset(row.trainingType)) {
        const td = textDatasets.find(d => d.id === s)
        return { id: s, label: td ? `${td.name} (${td.row_count} rows)` : String(s), source: 'default' }
      }
      const p = lsProjects.find(p => p.id === s)
      return { id: s, label: p ? `${p.title} (${p.finished_task_number} labeled)` : String(s), source: 'default' }
    }
    const v = overrides[row.key]
    if (v == null || v === '') return { id: null, label: '—', source: 'none' }
    if (needsTextDataset(row.trainingType)) {
      const td = textDatasets.find(d => d.id === v)
      return { id: v, label: td ? `${td.name} (${td.row_count} rows)` : String(v), source: 'user' }
    }
    const p = lsProjects.find(p => p.id === v)
    return { id: v, label: p ? `${p.title} (${p.finished_task_number} labeled)` : String(v), source: 'user' }
  }

  // ── Pre-flight validation: surface client-side errors before the
  //    job hits the cluster. Catches "no image dataset", "no labeled
  //    tasks", and "no text dataset" without burning a Ray slot. ──
  const validateBeforeSubmit = useCallback((row: ModelRow, ds: { id: number | string | null; label: string; source: 'default' | 'user' | 'none' }): string | null => {
    if (ds.id == null) {
      if (needsTextDataset(row.trainingType)) {
        return 'No text dataset — go to Datasets → upload a .jsonl file (or use auto-created sample)'
      }
      return 'No image dataset — create a Label Studio project and label some images first'
    }
    if (needsTextDataset(row.trainingType)) {
      const td = textDatasets.find(d => d.id === ds.id)
      if (!td) return 'Text dataset not found'
      if (td.row_count < 1) return 'Text dataset is empty'
      return null
    }
    // Image project
    const p = lsProjects.find(p => p.id === ds.id)
    if (!p) return 'Label Studio project not found'
    if (p.finished_task_number < 1) {
      return `Project "${p.title}" has 0 labeled tasks — annotate at least 1 image before training`
    }
    if (row.trainingType === 'detection' && row.option.engine === 'Ultralytics' && p.finished_task_number < 5) {
      return `Project "${p.title}" only has ${p.finished_task_number} labeled tasks — detection needs ≥5 to be useful`
    }
    return null
  }, [lsProjects, textDatasets])

  // ── Submit single test (uses `bulkProvider` so the bulk loop can
  //    steer Ray vs Modal from one place). ───────────────────────────────
  const submitTest = useCallback(async (row: ModelRow, provider: 'ray' | 'modal' = bulkProvider) => {
    const ds = effectiveDataset(row)
    const preErr = validateBeforeSubmit(row, ds)
    if (preErr) {
      setRunLocal(row.key, {
        key: row.key, jobId: null, status: 'error', error: preErr,
        startedAt: null, finishedAt: Date.now() / 1000, log: '', datasetLabel: ds.label,
      })
      return
    }
    setRunLocal(row.key, {
      key: row.key, jobId: null, status: 'submitting', error: null,
      startedAt: null, finishedAt: null, log: '', datasetLabel: ds.label,
    })
    try {
      const body = {
        training_type: row.trainingType,
        model_name:    row.option.model,
        engine:        row.option.engine,
        epochs:        1,
        batch_size:    8,
        learning_rate: 0.001,
        optimizer:     'adamw',
        imgsz:         640,
        notes:         `test-all run · ${row.option.label}`,
        cluster:       provider,
        text_dataset:  needsTextDataset(row.trainingType) ? String(ds.id) : '',
      }
      const r = await fetch(`/api/train/${ds.id}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      })
      if (!r.ok) {
        let detail = `HTTP ${r.status}`
        try { const d = await r.json(); detail = d?.detail || detail } catch {}
        throw new Error(detail)
      }
      const { job_id } = await r.json()
      const cur = moduleRuns[row.key]
      setRunLocal(row.key, {
        ...cur, jobId: job_id, status: 'queued', startedAt: Date.now() / 1000,
      })
      // Track at module scope so the background poll loop will pick
      // it up even if the user navigates away.
      trackJob(row.key, job_id)
    } catch (e: any) {
      const cur = moduleRuns[row.key]
      setRunLocal(row.key, {
        ...cur, status: 'error', error: e.message, finishedAt: Date.now() / 1000,
      })
    }
  }, [effectiveDataset, validateBeforeSubmit, bulkProvider])

  // ── Deploy a freshly-trained model. Used by the bulk loop right after
  //    a successful training job completes (per-model train → deploy →
  //    next flow). Mirrors the DeployModels page's POST to /api/deploy. ─
  const deployModel = useCallback(async (row: ModelRow, provider: 'ray' | 'modal'): Promise<{ ok: boolean; url?: string; error?: string; elapsedSec: number }> => {
    const t0 = Date.now() / 1000
    try {
      const r = await fetch('/api/deploy', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          model_id:   row.option.value,
          model_name: row.option.model,
          provider,
        }),
      })
      if (!r.ok) {
        let detail = `HTTP ${r.status}`
        try { const d = await r.json(); detail = d?.detail || detail } catch {}
        return { ok: false, error: detail, elapsedSec: Math.round(Date.now() / 1000 - t0) }
      }
      const data = await r.json()
      return {
        ok: true,
        url: data.url ?? data.endpoint ?? `${provider}://${row.option.value}`,
        elapsedSec: Math.round(Date.now() / 1000 - t0),
      }
    } catch (e: any) {
      return { ok: false, error: e.message, elapsedSec: Math.round(Date.now() / 1000 - t0) }
    }
  }, [])

  // ── Polling is now handled by the module-level ensurePolling() ──
  //    The background poll loop fires every 3s while any job is tracked
  //    and updates the store regardless of whether this component is
  //    mounted. The useStoreSnapshot() hook above re-renders this view
  //    whenever the store changes.

  // ── Wait until a given run reaches a terminal state (OK or error).
  //    Polls the job endpoint every 2s; resolves as soon as the run is
  //    no longer queued/running/submitting. Used by runAllCompatible so
  //    we never train two models in parallel. ──
  const waitForCompletion = useCallback(async (key: string, timeoutMs = 600_000): Promise<void> => {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (bulkStopRequestedRef.current) return
      const cur = moduleRuns[key]
      if (cur && (cur.status === 'completed' || cur.status === 'error')) return
      if (cur && cur.status === 'idle' && cur.error) return
      await new Promise(r => setTimeout(r, 2000))
    }
  }, [])

  // Keep latest bulkStopRequested in a ref so waitForCompletion sees it
  const bulkStopRequestedRef = useRef(bulkStopRequested)
  useEffect(() => { bulkStopRequestedRef.current = bulkStopRequested }, [bulkStopRequested])

  // ── Run all compatible — strictly sequential: per-model TRAIN →
  //    DEPLOY → NEXT. The user picks Ray vs Modal once before the loop
  //    starts; every model goes to the same provider. ──────────────────
  const runAllCompatible = async () => {
    if (bulkRunning) return
    setBulkRunning(true)
    setBulkStopRequested(false)
    bulkStopRequestedRef.current = false
    setBulkStream([])
    setBulkReport(null)

    const targets = filteredRows.filter(r => isCompatible(r.option))
    const startedAt = Date.now() / 1000
    const failures: BulkFailure[] = []
    const deployed: BulkDeployed[] = []
    let ok = 0
    let deployedOk = 0
    let stopped = false
    const logLines: string[] = []

    let idx = 0
    for (const row of targets) {
      if (bulkStopRequestedRef.current) { stopped = true; break }
      idx += 1
      setBulkProgress({ current: idx, total: targets.length, label: row.option.label, stage: 'train' })

      // Skip rows that already finished in a previous run, but re-run
      // anything that errored so the user can retry without resetting
      // the whole table.
      const existing = moduleRuns[row.key]
      if (existing && (existing.status === 'queued' || existing.status === 'running' || existing.status === 'submitting')) {
        // Wait for the in-flight one first
        await waitForCompletion(row.key)
      } else if (existing && existing.status === 'completed') {
        logLines.push(`[${idx}/${targets.length}]  ✓  ${row.option.label}  (cached)`)
        ok += 1
        setBulkStream(s => [...s, { label: row.option.label, status: 'ok' }])
        continue
      } else {
        await submitTest(row, bulkProvider)
        await waitForCompletion(row.key)
      }

      // Read the final state from the module store (the polling loop
      // has already attributed the job's status to this row key).
      const final = moduleRuns[row.key]
      if (final?.status !== 'completed') {
        const errMsg = (final?.error ?? 'Unknown error').slice(0, 500)
        failures.push({
          key:          row.key,
          label:        row.option.label,
          trainingType: row.trainingType,
          engine:       row.option.engine,
          model:        row.option.model,
          error:        errMsg,
          jobId:        final?.jobId ?? null,
          stage:        'train',
        })
        logLines.push(`[${idx}/${targets.length}]  ✗  TRAIN  ${row.option.label}  —  ${errMsg.slice(0, 120)}`)
        setBulkStream(s => [...s, { label: row.option.label, status: 'err', error: `TRAIN: ${errMsg}` }])
        continue
      }
      ok += 1
      logLines.push(`[${idx}/${targets.length}]  ✓  TRAIN  ${row.option.label}`)
      setBulkStream(s => [...s, { label: row.option.label, status: 'ok' }])

      // TRAIN → DEPLOY step (skip if user disabled it)
      if (bulkDeployEnabled) {
        if (bulkStopRequestedRef.current) { stopped = true; break }
        setBulkProgress({ current: idx, total: targets.length, label: row.option.label, stage: 'deploy' })
        setBulkStream(s => [...s, { label: row.option.label, status: 'deploy' }])
        const dep = await deployModel(row, bulkProvider)
        if (dep.ok) {
          deployedOk += 1
          deployed.push({
            key:        row.key,
            label:      row.option.label,
            provider:   bulkProvider,
            url:        dep.url ?? '',
            elapsedSec: dep.elapsedSec,
          })
          logLines.push(`[${idx}/${targets.length}]  ✓  DEPLOY  ${row.option.label}  →  ${dep.url}  (${dep.elapsedSec}s)`)
          setBulkStream(s => [...s, { label: row.option.label, status: 'deployed' }])
        } else {
          const errMsg = (dep.error ?? 'Unknown deploy error').slice(0, 500)
          failures.push({
            key:          row.key,
            label:        row.option.label,
            trainingType: row.trainingType,
            engine:       row.option.engine,
            model:        row.option.model,
            error:        errMsg,
            jobId:        final?.jobId ?? null,
            stage:        'deploy',
          })
          logLines.push(`[${idx}/${targets.length}]  ✗  DEPLOY  ${row.option.label}  —  ${errMsg.slice(0, 120)}`)
          setBulkStream(s => [...s, { label: row.option.label, status: 'err', error: `DEPLOY: ${errMsg}` }])
        }
      }
    }

    const report: BulkReport = {
      startedAt,
      finishedAt:  Date.now() / 1000,
      total:       targets.length,
      ok,
      deployedCount: deployedOk,
      failed:      failures.length,
      stopped,
      provider:    bulkDeployEnabled ? bulkProvider : null,
      deployEnabled: bulkDeployEnabled,
      failures,
      deployed,
      log:         logLines.join('\n'),
    }
    setBulkReport(report)
    setBulkProgress(null)
    setBulkRunning(false)
    setBulkStopRequested(false)
    bulkStopRequestedRef.current = false
  }

  // ── Aggregate stats ──
  const stats = useMemo(() => {
    const list = Object.values(runs)
    return {
      total:     list.length,
      pending:   list.filter(r => r.status === 'submitting' || r.status === 'queued' || r.status === 'running').length,
      ok:        list.filter(r => r.status === 'completed').length,
      err:       list.filter(r => r.status === 'error').length,
    }
  }, [runs])

  const toggleExpand = (key: string) => {
    setExpanded(prev => {
      const s = new Set(prev)
      s.has(key) ? s.delete(key) : s.add(key)
      return s
    })
  }

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <FlaskConical size={22} color="var(--primary)" />
            <h1 style={{ fontWeight: 700, fontSize: 22, color: 'var(--text-primary)' }}>Test All Models</h1>
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
            Smoke test every model in Train · 1 epoch · 8 batch · auto-paired with the best matching dataset
          </p>
          {bulkProgress && (
            <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Loader2 size={13} color="var(--primary)" className="animate-spin" />
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Running <strong style={{ color: 'var(--text-primary)' }}>{bulkProgress.label}</strong>
                <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>
                  ({bulkProgress.current} / {bulkProgress.total})
                </span>
              </span>
              <div style={{ flex: 1, maxWidth: 220, height: 4, background: 'var(--bg-elevated)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{
                  width: `${(bulkProgress.current / bulkProgress.total) * 100}%`,
                  height: '100%',
                  background: 'var(--primary)',
                  transition: 'width 0.3s ease',
                }} />
              </div>
            </div>
          )}

          {/* Live stream — surfaces each model's status as it completes
              so the user sees failures in real time during long runs. */}
          {bulkStream.length > 0 && bulkRunning && (
            <div style={{ marginTop: 10, maxHeight: 140, overflowY: 'auto', padding: '8px 10px', borderRadius: 6, background: 'var(--bg-elevated)', fontSize: 11.5, fontFamily: 'var(--font-mono)' }}>
              {bulkStream.slice(-20).map((s, i) => (
                <div key={`${s.label}-${i}`} style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '1px 0' }}>
                  {s.status === 'ok'
                    ? <CheckCircle2 size={11} color="var(--success)" />
                    : <XCircle      size={11} color="var(--danger)"  />}
                  <span style={{ color: s.status === 'ok' ? 'var(--text-secondary)' : 'var(--danger)', flex: 1 }}>{s.label}</span>
                  {s.error && (
                    <span style={{ color: 'var(--text-muted)', fontSize: 10.5, maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {s.error.slice(0, 80)}
                    </span>
                  )}
                </div>
              ))}
              {bulkStream.length > 20 && (
                <div style={{ color: 'var(--text-muted)', fontSize: 10, marginTop: 4 }}>
                  … +{bulkStream.length - 20} earlier
                </div>
              )}
            </div>
          )}

          {/* Post-run report — summary card + failed-list + export. */}
          {bulkReport && !bulkRunning && <BulkReportPanel report={bulkReport} setExpanded={setExpanded} onClose={() => setBulkReport(null)} />}
        </div>

        {/* Pre-run config — pick provider + toggle train→deploy chain. */}
        {!bulkRunning && !bulkReport && (
          <div style={{
            marginTop: 14, padding: 12, borderRadius: 8,
            background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
            display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
              <Server size={12} />
              <span style={{ fontWeight: 600 }}>Provider:</span>
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              {([
                { val: 'ray',   label: 'Ray Serve', icon: Server },
                { val: 'modal', label: 'Modal',     icon: Cloud },
              ] as const).map(({ val, label, icon: Icon }) => {
                const active = bulkProvider === val
                return (
                  <button
                    key={val}
                    onClick={() => setBulkProvider(val as 'ray' | 'modal')}
                    style={{
                      padding: '5px 10px', borderRadius: 6,
                      border: `1px solid ${active ? 'var(--primary)' : 'var(--border-default)'}`,
                      background: active ? 'var(--primary-dim)' : 'transparent',
                      color: active ? 'var(--primary-hover)' : 'var(--text-secondary)',
                      cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5,
                      fontSize: 11.5, fontWeight: 500,
                    }}
                  >
                    <Icon size={11} /> {label}
                  </button>
                )
              })}
            </div>
            <div style={{ width: 1, height: 22, background: 'var(--border-subtle)' }} />
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)', cursor: 'pointer', userSelect: 'none' }}>
              <input
                type="checkbox"
                checked={bulkDeployEnabled}
                onChange={e => setBulkDeployEnabled(e.target.checked)}
                style={{ width: 13, height: 13, cursor: 'pointer', accentColor: 'var(--primary)' }}
              />
              <span style={{ fontWeight: 500 }}>Train → Deploy → Next</span>
              <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>(auto-deploy after each train OK)</span>
            </label>
          </div>
        )}

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="btn btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={() => {
              for (const k of Object.keys(moduleRuns)) delete moduleRuns[k]
              for (const jobId of [...activeJobs.keys()]) untrackJob(jobId)
              notify()
            }}
            disabled={bulkRunning}
            title="Clear all results"
          >
            <RefreshCw size={14} /> Reset
          </button>
          {bulkRunning ? (
            <button
              className="btn btn-sm"
              style={{ background: 'var(--danger)', color: '#fff', border: 'none', display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={() => setBulkStopRequested(true)}
            >
              <StopCircle size={14} /> Stop Bulk
            </button>
          ) : (
            <button
              className="btn btn-primary"
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={runAllCompatible}
              disabled={datasetsLoading || filteredRows.length === 0}
            >
              <Play size={14} /> Run {filteredRows.filter(r => isCompatible(r.option)).length} Compatible
            </button>
          )}
        </div>
      </div>

      {/* Stats strip */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 18,
      }}>
        {[
          { label: 'Tested',     value: stats.total,                            color: 'var(--text-primary)', icon: FlaskConical },
          { label: 'Pending',    value: stats.pending,                          color: 'var(--primary)',      icon: Loader2 },
          { label: 'Succeeded',  value: stats.ok,                               color: 'var(--success)',      icon: CheckCircle2 },
          { label: 'Failed',     value: stats.err,                              color: 'var(--danger)',       icon: XCircle },
        ].map(({ label, value, color, icon: Icon }) => (
          <div key={label} className="card" style={{ padding: '12px 16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <Icon size={14} color={color} />
              <span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>{label}</span>
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="card" style={{ padding: 12, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: '1 1 240px', minWidth: 220 }}>
          <Search size={14} color="var(--text-muted)" />
          <input
            placeholder="Search model / engine…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text-primary)', fontSize: 13, padding: '4px 0',
            }}
          />
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <Filter size={13} color="var(--text-muted)" />
          {(['all', ...TYPE_ORDER] as const).map(t => (
            <button
              key={t}
              onClick={() => setTypeFilter(t as any)}
              style={{
                padding: '4px 10px', borderRadius: 14, fontSize: 11.5, fontWeight: 500,
                border: '1px solid', cursor: 'pointer', transition: 'all .15s',
                borderColor: typeFilter === t ? (t === 'all' ? 'var(--primary)' : TYPE_COLORS[t as TrainingType]) : 'var(--border-default)',
                background: typeFilter === t ? (t === 'all' ? 'var(--primary)' : TYPE_COLORS[t as TrainingType] + '20') : 'var(--bg-surface)',
                color: typeFilter === t ? (t === 'all' ? '#fff' : TYPE_COLORS[t as TrainingType]) : 'var(--text-secondary)',
              }}
            >
              {t === 'all' ? 'All' : TYPE_LABELS[t as TrainingType]}
            </button>
          ))}
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)', cursor: 'pointer' }}>
          <input type="checkbox" checked={showIncompatible} onChange={e => setShowIncompatible(e.target.checked)} />
          Show incompatible ({allRows.length - allRows.filter(r => isCompatible(r.option)).length})
        </label>
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '32px 1.6fr 1fr 2fr 130px 110px 90px',
          padding: '10px 16px',
          borderBottom: '1px solid var(--border-subtle)',
          background: 'var(--bg-elevated)',
          fontSize: 10.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em',
          color: 'var(--text-muted)',
        }}>
          <div></div>
          <div>Model</div>
          <div>Type</div>
          <div>Dataset</div>
          <div>Status</div>
          <div>Elapsed</div>
          <div style={{ textAlign: 'right' }}>Action</div>
        </div>

        {datasetsLoading && (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
            <Loader2 size={14} className="animate-spin" /> Loading datasets…
          </div>
        )}

        {!datasetsLoading && filteredRows.length === 0 && (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            ไม่มี model ที่ตรง filter
          </div>
        )}

        {filteredRows.map(row => {
          const run = runs[row.key]
          const ds  = effectiveDataset(row)
          const isExpanded = expanded.has(row.key)
          const typeColor = TYPE_COLORS[row.trainingType]
          const isIncompatible = !isCompatible(row.option)

          return (
            <div key={row.key} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: '32px 1.6fr 1fr 2fr 130px 110px 90px',
                padding: '12px 16px',
                alignItems: 'center',
                background: isIncompatible ? 'var(--bg-surface)' : 'transparent',
                opacity: isIncompatible ? 0.6 : 1,
              }}>
                {/* Expand chevron */}
                <button
                  onClick={() => toggleExpand(row.key)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 0, display: 'flex' }}
                >
                  {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>

                {/* Model */}
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {row.option.label}
                  </div>
                  <div style={{ fontSize: 10.5, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {row.option.model}
                  </div>
                </div>

                {/* Type badge */}
                <div>
                  <span style={{
                    fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 10,
                    background: typeColor + '20', color: typeColor,
                    textTransform: 'uppercase', letterSpacing: '0.04em',
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                  }}>
                    {isLlmType(row.trainingType) && <MessageSquare size={9} />}
                    {!isLlmType(row.trainingType) && <Cpu size={9} />}
                    {TYPE_LABELS[row.trainingType]}
                  </span>
                  {isIncompatible && (
                    <div style={{ fontSize: 9.5, color: 'var(--text-muted)', marginTop: 2 }}>
                      {row.option.hardware}
                    </div>
                  )}
                </div>

                {/* Dataset */}
                <DatasetCell
                  row={row}
                  ds={ds}
                  lsProjects={lsProjects}
                  textDatasets={textDatasets}
                  useDefault={useDefault[row.key] !== false}
                  validationError={validateBeforeSubmit(row, ds)}
                  onToggleDefault={(useD) => {
                    setUseDefault(prev => ({ ...prev, [row.key]: useD }))
                    // When the user first switches into override mode,
                    // seed the override with the current suggested dataset
                    // so the dropdown opens on a valid selection instead
                    // of starting blank.
                    if (!useD && overrides[row.key] == null) {
                      const s = suggestedDataset(row)
                      if (s != null) setOverrides(prev => ({ ...prev, [row.key]: s }))
                    }
                  }}
                  onChangeOverride={(v) => setOverrides(prev => ({ ...prev, [row.key]: v }))}
                />

                {/* Status */}
                <StatusCell run={run} />

                {/* Elapsed */}
                <div style={{ fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {formatElapsed(run?.startedAt ?? null, run?.finishedAt ?? null)}
                </div>

                {/* Action */}
                <div style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                  <button
                    className="btn btn-sm"
                    style={{
                      background: 'var(--primary)', color: '#fff', border: 'none',
                      display: 'flex', alignItems: 'center', gap: 4,
                      opacity: (run?.status === 'submitting' || run?.status === 'queued' || run?.status === 'running') ? 0.5 : 1,
                    }}
                    onClick={() => submitTest(row)}
                    disabled={isIncompatible || (run?.status === 'submitting' || run?.status === 'queued' || run?.status === 'running')}
                    title={isIncompatible ? 'Incompatible with current hardware' : `Test ${row.option.label}`}
                  >
                    {run?.status === 'submitting' ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />}
                    Test
                  </button>
                </div>
              </div>

              {/* Expanded: log + error */}
              {isExpanded && (
                <div style={{ padding: '0 16px 14px 48px', background: 'var(--bg-base)' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 12 }}>
                    <div>
                      <div style={{ color: 'var(--text-muted)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: 4 }}>Engine</div>
                      <div style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{row.option.engine}</div>
                    </div>
                    <div>
                      <div style={{ color: 'var(--text-muted)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: 4 }}>Hardware</div>
                      <div style={{ color: 'var(--text-secondary)' }}>{row.option.hardware}</div>
                    </div>
                    <div style={{ gridColumn: '1 / -1' }}>
                      <div style={{ color: 'var(--text-muted)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: 4 }}>Description</div>
                      <div style={{ color: 'var(--text-secondary)' }}>{row.option.description}</div>
                    </div>
                    {run?.error && (
                      <div style={{ gridColumn: '1 / -1' }}>
                        <div style={{ color: 'var(--danger)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                          <AlertCircle size={11} /> Error
                        </div>
                        <pre style={{ background: 'var(--danger-dim)', border: '1px solid var(--danger)', borderRadius: 6, padding: 8, fontSize: 11, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 160, overflowY: 'auto', margin: 0 }}>
                          {run.error}
                        </pre>
                      </div>
                    )}
                    {run?.log && (
                      <div style={{ gridColumn: '1 / -1' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                          <Terminal size={11} /> Training Log
                        </div>
                        <pre style={{ background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', borderRadius: 6, padding: 8, fontSize: 10.5, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 200, overflowY: 'auto', margin: 0 }}>
                          {run.log.slice(-2000) || '(empty)'}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function DatasetCell({ row, ds, lsProjects, textDatasets, useDefault, validationError, onToggleDefault, onChangeOverride }: {
  row: ModelRow
  ds: { id: number | string | null; label: string; source: 'default' | 'user' | 'none' }
  lsProjects: LSProject[]
  textDatasets: TextDataset[]
  useDefault: boolean
  validationError: string | null
  onToggleDefault: (useDefault: boolean) => void
  onChangeOverride: (v: number | string | null) => void
}) {
  const isLlm = needsTextDataset(row.trainingType)
  const options = isLlm ? textDatasets : lsProjects

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
      {ds.source === 'none' ? (
        <span style={{ fontSize: 11, color: 'var(--danger)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          <Database size={11} /> ไม่มี {isLlm ? 'text' : 'image'} dataset
        </span>
      ) : useDefault ? (
        <>
          <Database size={11} color="var(--text-muted)" style={{ flexShrink: 0 }} />
          <span
            style={{
              fontSize: 11.5, color: 'var(--text-secondary)', overflow: 'hidden',
              textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0,
            }}
            title={ds.label}
          >
            {ds.label}
          </span>
          <button
            onClick={() => onToggleDefault(false)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer', padding: 2,
              color: 'var(--text-muted)', display: 'flex', flexShrink: 0,
            }}
            title="Override dataset"
          >
            <Eye size={11} />
          </button>
          {validationError && (
            <span
              title={validationError}
              style={{ display: 'inline-flex', alignItems: 'center', color: 'var(--warning)', flexShrink: 0, cursor: 'help' }}
            >
              <AlertCircle size={12} />
            </span>
          )}
        </>
      ) : (
        <>
          <select
            value={ds.id == null ? '' : String(ds.id)}
            onChange={e => {
              const v = e.target.value
              if (!v) { onChangeOverride(null); return }
              onChangeOverride(isLlm ? v : Number(v))
            }}
            style={{
              flex: 1, minWidth: 0, padding: '3px 6px', fontSize: 11,
              background: 'var(--bg-input)', border: '1px solid var(--border-default)',
              borderRadius: 5, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)',
            }}
          >
            {options.length === 0 && <option value="">-- no dataset --</option>}
            {options.map(o => {
              const id = isLlm ? o.id : (o as LSProject).id
              const label = isLlm
                ? `${(o as TextDataset).name} (${(o as TextDataset).row_count})`
                : `${(o as LSProject).title} (${(o as LSProject).finished_task_number})`
              return <option key={String(id)} value={String(id)}>{label}</option>
            })}
          </select>
          <button
            onClick={() => onToggleDefault(true)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer', padding: 2,
              color: 'var(--text-muted)', display: 'flex', flexShrink: 0,
            }}
            title="Use auto-suggested dataset"
          >
            <EyeOff size={11} />
          </button>
          {validationError && (
            <span
              title={validationError}
              style={{ display: 'inline-flex', alignItems: 'center', color: 'var(--warning)', flexShrink: 0, cursor: 'help' }}
            >
              <AlertCircle size={12} />
            </span>
          )}
        </>
      )}
    </div>
  )
}

function StatusCell({ run }: { run: TestRun | undefined }) {
  if (!run) {
    return <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>—</span>
  }
  const map: Record<TestRun['status'], { icon: any; color: string; label: string }> = {
    idle:       { icon: Clock,         color: 'var(--text-muted)',     label: 'Idle' },
    submitting: { icon: Loader2,       color: 'var(--primary)',        label: 'Submitting' },
    queued:     { icon: Clock,         color: 'var(--warning)',        label: 'Queued' },
    running:    { icon: Loader2,       color: 'var(--primary)',        label: 'Running' },
    completed:  { icon: CheckCircle2,  color: 'var(--success)',        label: 'OK' },
    error:      { icon: XCircle,       color: 'var(--danger)',         label: 'Error' },
  }
  const c = map[run.status]
  const Icon = c.icon
  const isAnim = run.status === 'submitting' || run.status === 'running'
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      fontSize: 11, fontWeight: 600, color: c.color,
    }}>
      <Icon size={12} className={isAnim ? 'animate-spin' : ''} />
      {c.label}
    </span>
  )
}

// ─── Bulk report panel — surfaces per-model failures + a copyable log
//      after a "Run all compatible" pass. Lets the user see what broke
//      and export the failure list for follow-up fixes. ────────────────
function BulkReportPanel({ report, setExpanded, onClose }: {
  report:     BulkReport
  setExpanded: React.Dispatch<React.SetStateAction<Set<string>>>
  onClose:    () => void
}) {
  const [copied, setCopied] = useState(false)

  // Build a single-line summary of each failed model for copy/export
  const failureLines = report.failures.map(f =>
    `[${f.trainingType}] ${f.label}  (${f.engine}/${f.model})  →  ${f.error}` +
    (f.jobId ? `  [jobId=${f.jobId}]` : '')
  )

  function copyToClipboard() {
    const dur = `${Math.round(report.finishedAt - report.startedAt)}s`
    const header = report.deployEnabled
      ? `Bulk test run: ${report.ok} OK, ${report.deployedCount} deployed, ${report.failed} failed, ${report.total} total (provider=${report.provider ?? 'n/a'}, ${dur})`
      : `Bulk test run: ${report.ok} OK, ${report.failed} failed, ${report.total} total (${dur})`
    const deployedBlock = report.deployed.length > 0
      ? '\n\nDeployed:\n' + report.deployed.map(d => `  ✓ ${d.label}  →  ${d.url}  (${d.elapsedSec}s)`).join('\n')
      : ''
    const failureBlock = failureLines.length > 0
      ? '\n\nFailures:\n' + failureLines.join('\n')
      : ''
    const text = report.failures.length === 0 && report.deployed.length === 0
      ? `Bulk test run completed with no failures.\n${report.ok}/${report.total} models OK in ${dur}`
      : header + deployedBlock + failureBlock
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    }).catch(() => { /* clipboard blocked — fall back to textarea selection */ })
  }

  function downloadJson() {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `bulk-test-${new Date(report.startedAt * 1000).toISOString().slice(0, 19).replace(/:/g, '-')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const elapsedSec = Math.round(report.finishedAt - report.startedAt)
  const okPct = report.total > 0 ? Math.round((report.ok / report.total) * 100) : 0

  return (
    <div style={{
      marginTop: 14, padding: 16, borderRadius: 10,
      border: `1px solid ${report.failed > 0 ? 'rgba(239,68,68,0.35)' : 'rgba(16,185,129,0.35)'}`,
      background: report.failed > 0 ? 'rgba(239,68,68,0.04)' : 'rgba(16,185,129,0.04)',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
            {report.failed > 0
              ? <XCircle size={18} color="var(--danger)" />
              : report.deployEnabled && report.deployedCount > 0
                ? <Rocket size={18} color="var(--success)" />
                : <CheckCircle2 size={18} color="var(--success)" />}
            <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>
              {report.stopped
                ? `Bulk run stopped · ${report.ok} OK${report.deployEnabled ? `, ${report.deployedCount} deployed` : ''}, ${report.failed} failed (out of ${report.total} started)`
                : report.failed > 0
                  ? `Bulk run finished · ${report.failed} of ${report.total} failed`
                  : report.deployEnabled
                    ? `Bulk run finished · all ${report.total} trained + ${report.deployedCount} deployed to ${report.provider}`
                    : `Bulk run finished · all ${report.ok} models OK`}
            </span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              ({elapsedSec}s)
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, fontSize: 12, color: 'var(--text-secondary)' }}>
            <span><strong style={{ color: 'var(--success)' }}>{report.ok}</strong> OK</span>
            {report.deployEnabled && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <Rocket size={11} color="var(--info, #3b82f6)" />
                <strong style={{ color: 'var(--info, #3b82f6)' }}>{report.deployedCount}</strong> deployed
              </span>
            )}
            <span><strong style={{ color: 'var(--danger)'  }}>{report.failed}</strong> failed</span>
            <span style={{ color: 'var(--text-muted)' }}>{okPct}% success</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
          <button
            className="btn btn-secondary btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11.5 }}
            onClick={copyToClipboard}
            title="Copy failure list to clipboard (or full summary if no failures)"
          >
            {copied ? <><Check size={12} /> Copied</> : <><Copy size={12} /> Copy</>}
          </button>
          <button
            className="btn btn-secondary btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11.5 }}
            onClick={downloadJson}
            title="Download full report as JSON"
          >
            <Download size={12} /> JSON
          </button>
          <button
            className="btn btn-secondary btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11.5, padding: '4px 8px' }}
            onClick={onClose}
            title="Dismiss report"
          >
            <X size={12} />
          </button>
        </div>
      </div>

      {/* 2-column body: failed list (left) + run log (right).
          Width matches the 4-card stat strip below (full container width
          inside the maxWidth:1200 parent). */}
      <div style={{
        marginTop: 14,
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 12,
      }}>
        {/* LEFT — failed models list */}
        <div style={{
          border: '1px solid var(--border-subtle)', borderRadius: 6,
          background: 'var(--bg-base)',
          display: 'flex', flexDirection: 'column',
          minHeight: 0, maxHeight: 380,
        }}>
          <div style={{
            padding: '8px 12px', borderBottom: '1px solid var(--border-subtle)',
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: 11, fontWeight: 600, color: 'var(--text-muted)',
            textTransform: 'uppercase', letterSpacing: '0.05em', flexShrink: 0,
          }}>
            <XCircle size={11} color={report.failures.length > 0 ? 'var(--danger)' : 'var(--text-muted)'} />
            Failed models
            <span style={{ color: report.failures.length > 0 ? 'var(--danger)' : 'var(--text-muted)', fontWeight: 700 }}>
              ({report.failures.length})
            </span>
          </div>
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {report.failures.length === 0
              ? <div style={{ padding: '20px 14px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                  <CheckCircle2 size={20} color="var(--success)" style={{ marginBottom: 6 }} />
                  <div>No failures — every compatible model completed.</div>
                </div>
              : report.failures.map((f, i) => (
                <div key={f.key} style={{
                  padding: '10px 12px', borderBottom: i < report.failures.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                  display: 'flex', alignItems: 'flex-start', gap: 10,
                }}>
                  <XCircle size={14} color="var(--danger)" style={{ marginTop: 1, flexShrink: 0 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                      <span style={{ fontWeight: 600, fontSize: 12.5, color: 'var(--text-primary)' }}>{f.label}</span>
                      <span style={{
                        fontSize: 9.5, fontWeight: 700, padding: '1px 6px', borderRadius: 4,
                        background: f.stage === 'train' ? 'rgba(245,158,11,0.15)' : 'rgba(59,130,246,0.15)',
                        color:      f.stage === 'train' ? '#d97706'                 : '#3b82f6',
                        textTransform: 'uppercase', letterSpacing: '0.04em',
                      }}>
                        {f.stage === 'train' ? '✗ TRAIN' : '✗ DEPLOY'}
                      </span>
                      <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: 'var(--bg-elevated)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{f.trainingType}</span>
                      {f.jobId && (
                        <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>job={f.jobId.slice(0, 10)}…</span>
                      )}
                    </div>
                    <div style={{ fontSize: 10.5, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {f.engine} · {f.model}
                    </div>
                    <div style={{
                      fontSize: 11, color: 'var(--danger)', marginTop: 4, fontFamily: 'var(--font-mono)',
                      whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 72, overflowY: 'auto',
                      padding: 6, borderRadius: 4, background: 'rgba(239,68,68,0.06)',
                    }}>
                      {f.error}
                    </div>
                  </div>
                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ fontSize: 11, padding: '3px 8px', flexShrink: 0 }}
                    onClick={() => setExpanded(prev => new Set(prev).add(f.key))}
                    title="Open the row in the table below to see the full training log"
                  >
                    View log
                  </button>
                </div>
              ))
            }
          </div>
        </div>

        {/* RIGHT — full run log (always visible) */}
        <div style={{
          border: '1px solid var(--border-subtle)', borderRadius: 6,
          background: 'var(--bg-base)',
          display: 'flex', flexDirection: 'column',
          minHeight: 0, maxHeight: 380,
        }}>
          <div style={{
            padding: '8px 12px', borderBottom: '1px solid var(--border-subtle)',
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: 11, fontWeight: 600, color: 'var(--text-muted)',
            textTransform: 'uppercase', letterSpacing: '0.05em', flexShrink: 0,
          }}>
            <Terminal size={11} />
            Run log
            <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>({report.log.split('\n').filter(Boolean).length} lines)</span>
          </div>
          <pre style={{
            margin: 0, padding: 10, flex: 1, overflowY: 'auto',
            fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)',
            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}>{report.log || '(no log lines)'}</pre>
        </div>
      </div>
    </div>
  )
}
