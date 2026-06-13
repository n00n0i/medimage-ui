import { useState, useEffect, useCallback, useRef, useMemo } from 'react'

// ─── Backend-authoritative state (SQLite + WebSocket) ─────────────────────────
//
// All Test-All run state lives on the server in `bulk_runs` and
// `bulk_run_jobs` tables. The /ws/testall WebSocket pushes a fresh
// snapshot every 2s. There is no localStorage, no module-level map, and
// no client-side polling — the page is a pure renderer of the server
// state. Any tab, any browser, any reload sees the same data.

import keycloak from '../keycloak'
import { getPref, setPref, deletePref, subscribePrefs, refreshPrefs } from '../lib/userPrefs'

function wsTestAllUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const token = keycloak.token || ''
  const base = `${proto}//${window.location.host}/ws/testall`
  return token ? `${base}?token=${encodeURIComponent(token)}` : base
}

async function _refreshKeycloakToken(): Promise<void> {
  // Keycloak access tokens expire (~5min default). keycloak.token holds
  // the raw value — it stays as the expired value until updateToken()
  // is called. Without this, a long-lived tab's WS reconnects would
  // reuse the stale token, the backend would reject the handshake
  // (4001), and the user would silently stop receiving snapshot
  // updates. Refresh before each (re)connect so the URL always
  // carries a valid token.
  if (!keycloak.token) return
  try { await keycloak.updateToken(30) } catch { /* refresh failed — try with what we have */ }
}

// In-memory mirror of the server state. Components subscribe to changes
// and re-render. The store is filled by the WebSocket; mutations go back
// to the server via REST.
interface BulkJobRow {
  row_key: string
  job_id: string | null
  status: string
  error: string | null
  elapsed_sec: number
  deployed: number
  deploy_url: string | null
}
interface BulkRunRow {
  id: string
  started_at: number
  finished_at: number | null
  stopped: number
  provider: string
  deploy_enabled: number
  total: number
  ok: number
  failed: number
  deployed_count: number
}

interface BulkSnapshot {
  runs: BulkRunRow[]
  jobs_by_run: Record<string, BulkJobRow[]>
}

const _snapshot: BulkSnapshot = { runs: [], jobs_by_run: {} }
const _subs = new Set<() => void>()
let _ws: WebSocket | null = null
let _wsRetry: ReturnType<typeof setTimeout> | null = null
let _currentRunId: string | null = null
let _connected = false

function _applySnapshot(snap: BulkSnapshot) {
  Object.assign(_snapshot, snap)
  for (const cb of _subs) cb()
}

function _connect() {
  if (_ws && _ws.readyState <= 1) return
  // Refresh the Keycloak token first so the URL query param is valid
  // for the handshake (the access token expires ~5min after issue).
  void _refreshKeycloakToken().then(() => {
    const ws = new WebSocket(wsTestAllUrl())
    _ws = ws
    ws.onopen  = () => { _connected = true; for (const cb of _subs) cb() }
    ws.onclose = () => {
      _connected = false
      for (const cb of _subs) cb()
      if (_wsRetry) clearTimeout(_wsRetry)
      _wsRetry = setTimeout(_connect, 3000)
    }
    ws.onerror = () => ws.close()
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'bulk_snapshot' && msg.data) {
          _applySnapshot(msg.data)
        }
      } catch { /* ignore */ }
    }
  })
}

_connect()

function isTestAllConnected(): boolean { return _connected }
void isTestAllConnected  // reserved for future connection-status badge

function getSnapshot(): BulkSnapshot { return _snapshot }

function subscribeBulk(cb: () => void): () => void {
  _subs.add(cb)
  return () => { _subs.delete(cb) }
}

// ── REST helpers ──────────────────────────────────────────────────────────────
async function apiCreateBulkRun(provider: string, deployEnabled: boolean, total: number): Promise<string> {
  const r = await fetch('/api/bulk-runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, deploy_enabled: deployEnabled, total }),
  })
  if (!r.ok) throw new Error(`create bulk run HTTP ${r.status}`)
  const d = await r.json()
  return d.id
}

async function apiUpdateBulkRun(rid: string, patch: Record<string, any>) {
  const r = await fetch(`/api/bulk-runs/${rid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!r.ok) throw new Error(`update bulk run HTTP ${r.status}`)
}

async function apiStopBulkRun(rid: string): Promise<{ updated: number }> {
  // Mark the bulk run as stopped on the server. Idempotent — safe to
  // call repeatedly and safe when the run is already finished.
  const r = await fetch(`/api/bulk-runs/${rid}/stop`, { method: 'POST' })
  if (!r.ok) throw new Error(`stop bulk run HTTP ${r.status}`)
  return r.json()
}

async function apiUpsertBulkJob(rid: string, row_key: string, patch: Record<string, any>) {
  const r = await fetch(`/api/bulk-runs/${rid}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ row_key, ...patch }),
  })
  if (!r.ok) throw new Error(`upsert bulk job HTTP ${r.status}`)
}

async function _apiFetchJob(jobId: string): Promise<{ status: string; error: string | null; finished_at: number | null; log: string } | null> {
  try {
    const r = await fetch(`/api/jobs/${jobId}`)
    if (!r.ok) return null
    const j = await r.json()
    return { status: j.status, error: j.error, finished_at: j.finished_at, log: j.log || '' }
  } catch { return null }
}
import {
  FlaskConical, Play, RefreshCw, Loader2, CheckCircle2, XCircle, Clock,
  Terminal, ChevronDown, ChevronUp, Search, Cpu, MessageSquare, Eye, EyeOff,
  Database, AlertCircle, StopCircle, Filter, Copy, Check, Download, X,
  Server, Cloud, Rocket,
} from 'lucide-react'
import { TRAINING_MATRIX } from './TrainModel'

// ─── Types ────────────────────────────────────────────────────────────────────

type TrainingType = 'classification' | 'detection' | 'segmentation' | 'vlm-finetune' | 'self-supervised' | 'llm-text' | 'anomaly-detection'

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
  status: 'idle' | 'submitting' | 'queued' | 'running' | 'deploying' | 'deployed' | 'completed' | 'error'
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
// useStoreSnapshot(). The map is a derived view of the WebSocket
// snapshot — pure renderer state, no writes from React.
const moduleRuns: Record<string, TestRun> = {}

// Reset-in-flight counter. While > 0, both _rebuildModuleRunsFromSnapshot
// and _hydrateBulkState treat incoming snapshots as stale (because the
// server's DELETE for the reset is still in flight and a WS push may
// still carry the old rows). Without this guard, the Reset button
// would clear the UI, then a 2s-late WS push with the old snapshot
// would resurrect the report and the model statuses for one render
// before the next (empty) push finally cleared them — visible flicker.
let _resetInFlight = 0

function _rebuildModuleRunsFromSnapshot(snap: BulkSnapshot) {
  // While a Reset is in flight, ignore any non-empty snapshot push —
  // it's almost certainly the pre-DELETE data, and merging it would
  // resurrect rows the user just cleared. The post-DELETE empty
  // snapshot (handled below) still works and finishes the clear.
  if (_resetInFlight > 0 && snap.runs.length > 0) return
  const latest = snap.runs[0]
  if (!latest) {
    // Empty snapshot — the server has no bulk_runs for this user
    // (e.g. fresh page load, or a Reset just deleted them). Clear
    // ALL local rows including terminal-but-with-jobId states like
    // "deployed" that the older "completed/error/no-jobId" filter
    // would have leaked through.
    for (const k of Object.keys(moduleRuns)) delete moduleRuns[k]
    return
  }
  // Merge jobs from ALL runs in the snapshot, not just runs[0]. A single
  // Test gets attached to whatever _currentRunId was at submit time —
  // if a "Run all compatible" started afterwards, runs[0] is the bulk
  // run and the single Test's row sits in an older run. We need to see
  // both. The latest run still wins for the "current" pointer.
  for (const rid of Object.keys(snap.jobs_by_run)) {
    const runRow = snap.runs.find(r => r.id === rid)
    const runStarted  = runRow?.started_at  ?? latest.started_at
    const runFinished = runRow?.finished_at ?? null
    for (const j of snap.jobs_by_run[rid] || []) {
      const status = (
        j.status === 'completed' ? 'completed' :
        j.status === 'error'     ? 'error' :
        j.status === 'running'   ? 'running' :
        j.status === 'deploying' ? 'deploying' :
        j.status === 'deployed'  ? 'deployed' :
        j.status === 'queued'    ? 'queued' :
        j.status === 'submitting' ? 'submitting' : 'idle'
      ) as TestRun['status']
      moduleRuns[j.row_key] = {
        key:          j.row_key,
        jobId:        j.job_id,
        status,
        error:        j.error,
        // Per-job times, not the bulk-run times — the Elapsed column
        // should track the individual model's training duration, not
        // how long ago the whole bulk run started. The backend overlays
        // the live started_at/finished_at from the `jobs` table.
        startedAt:    (j as any).started_at ?? runStarted,
        finishedAt:   (status === 'completed' || status === 'error' || status === 'deployed') ? ((j as any).finished_at ?? runFinished) : null,
        log:          '',
        datasetLabel: '',
      }
    }
  }
  _currentRunId = latest.id
}

function getRun(key: string): TestRun | undefined {
  return moduleRuns[key]
}
void getRun  // reserved for external introspection (jobs page badge)

function setRunLocal(key: string, run: TestRun) {
  moduleRuns[key] = run
  // Mirror to the server so the snapshot reflects the new state. If
  // we don't have a current bulk_run yet (single-model Test, not a
  // "Run all" pass), lazily create one so the row state still syncs
  // and survives reloads / other tabs.
  if (_currentRunId) {
    void _pushRowToServer(_currentRunId, key, run)
  } else {
    void _ensureSingleRunId().then(rid => {
      if (rid) void _pushRowToServer(rid, key, run)
    })
  }
}

let _ensuringRun: Promise<string | null> | null = null
async function _ensureSingleRunId(): Promise<string | null> {
  // setRunLocal is called twice in quick succession for a single Test
  // (status='submitting' before the POST returns, then status='queued'
  // with jobId after). Both calls would otherwise race to create
  // separate bulk_runs and leave the job attached to one of them. Cache
  // the in-flight promise so the second call awaits the first.
  if (_ensuringRun) return _ensuringRun
  _ensuringRun = (async () => {
    try {
      const rid = await apiCreateBulkRun('ray', false, 1)
      _currentRunId = rid
      return rid
    } catch { return null }
    finally { _ensuringRun = null }
  })()
  return _ensuringRun
}

function _pushRowToServer(rid: string, key: string, run: TestRun) {
  return apiUpsertBulkJob(rid, key, {
    job_id:     run.jobId,
    status:     run.status,
    error:      run.error,
    elapsed_sec: 0,
    deployed:   false,
    deploy_url: null,
  })
}

function useStoreSnapshot(): Record<string, TestRun> {
  const [, force] = useState(0)
  useEffect(() => {
    _rebuildModuleRunsFromSnapshot(getSnapshot())
    return subscribeBulk(() => {
      _rebuildModuleRunsFromSnapshot(getSnapshot())
      force(n => n + 1)
    })
  }, [])
  return moduleRuns
}

// ── Bulk state (progress + report) — derived from server snapshot ─────────
interface BulkStreamEntry { label: string; status: 'ok' | 'err' | 'deploy' | 'deployed'; error?: string }
const _bulkState = {
  running: false,
  stopRequested: false,
  progress: null as { current: number; total: number; label: string; stage: 'train' | 'deploy' } | null,
  report: null as BulkReport | null,
  stream: [] as BulkStreamEntry[],
  // When the user dismisses the "Bulk run stopped/finished" panel we
  // remember the started_at of the run they dismissed so a follow-up
  // WS push doesn't immediately re-create the report. Reset when a NEW
  // run starts (different started_at). Persisted to user_prefs (server-
  // side, per-user) so a page refresh — in any browser, any tab, any
  // device — doesn't resurrect the dismissed report.
  dismissedRunId: _readDismissedRunId(),
}
const _bulkSubscribers = new Set<() => void>()
function _notifyBulk() { for (const cb of _bulkSubscribers) cb() }

// Persisted "user dismissed this run's report" flag. Stored server-side
// in the user_prefs table (per-user) so the dismissal survives a page
// refresh AND any browser/tab/device the same user logs in from. Read
// synchronously from the in-memory mirror — once the mirror is loaded
// the value is stable for this session.
const _BULK_DISMISS_KEY = 'medimage.bulkReport.dismissedRunId'
// Set true the first time the prefs mirror populates (or first write
// that creates the entry). Until then, _hydrateBulkState() must NOT
// build a report — otherwise the report would flash on screen and then
// get hidden once prefs arrive.
let _prefsReady = false
function _readDismissedRunId(): string | null {
  const v = getPref(_BULK_DISMISS_KEY)
  return v ?? null
}
function _writeDismissedRunId(id: string | null) {
  _prefsReady = true
  if (id == null) void deletePref(_BULK_DISMISS_KEY)
  else void setPref(_BULK_DISMISS_KEY, id)
}

function _hydrateBulkState() {
  const snap = getSnapshot()
  const latest = snap.runs[0]
  if (!latest) return
  // While a Reset is in flight, ignore non-empty snapshots so the
  // just-cleared report doesn't get resurrected from a stale WS push.
  // The local `report = null` written by the Reset handler stays in
  // place until the post-DELETE empty snapshot (handled by the early
  // return above) is received.
  if (_resetInFlight > 0) {
    _bulkState.report = null
    return
  }
  const jobs = snap.jobs_by_run[latest.id] || []
  const done = jobs.filter(j => j.status === 'completed' || j.status === 'error').length
  _bulkState.running = latest.finished_at == null
  // Match dismissal by started_at (stable unix timestamp). This is more
  // robust than the run UUID because the timestamp is identical between
  // dismiss time and the next page load, regardless of snapshot state.
  const latestKey = String(latest.started_at)
  // Re-read the dismissed flag from the prefs mirror every hydrate —
  // the mirror loads asynchronously after module init, so the value
  // set in the _bulkState initializer may be stale. This is the
  // server-side source of truth (per-user, survives any client storage
  // being cleared).
  const fromPrefs = _readDismissedRunId()
  if (fromPrefs !== _bulkState.dismissedRunId) {
    _bulkState.dismissedRunId = fromPrefs
  }
  // When a new run starts (different started_at), clear the dismissed
  // flag so the user can see its report when it finishes.
  if (_bulkState.dismissedRunId && _bulkState.dismissedRunId !== latestKey) {
    _bulkState.dismissedRunId = null
    _writeDismissedRunId(null)
  }
  if (_bulkState.running) {
    _bulkState.progress = {
      current: Math.min(done + 1, latest.total || 0),
      total:   latest.total || 0,
      label:   '',
      stage:   'train',
    }
  }
  // Only (re)build the report if the run finished AND the user hasn't
  // dismissed this run's report yet. If the user *has* dismissed it,
  // make sure any stale report is cleared.
  //
  // Wait for the prefs mirror to load before deciding — without this
  // guard, the report would flash on every page load (briefly shown
  // while dismissedRunId is still null, then hidden once prefs arrive).
  if (!_prefsReady) return
  if (latest.finished_at != null && _bulkState.dismissedRunId === latestKey) {
    _bulkState.report = null
  } else if (latest.finished_at != null) {
    _bulkState.report = {
      startedAt:     latest.started_at,
      finishedAt:    latest.finished_at,
      total:         latest.total,
      ok:            latest.ok,
      failed:        latest.failed,
      deployedCount: latest.deployed_count,
      stopped:       !!latest.stopped,
      provider:      latest.provider as 'ray' | 'modal' | null,
      deployEnabled: !!latest.deploy_enabled,
      failures: jobs.filter(j => j.status === 'error').map(j => ({
        key: j.row_key, label: j.row_key, trainingType: 'classification' as TrainingType,
        engine: '', model: '', error: j.error || 'unknown', jobId: j.job_id, stage: 'train' as const,
      })),
      deployed: jobs.filter(j => j.deployed).map(j => ({
        key: j.row_key, label: j.row_key, provider: 'ray' as const,
        url: j.deploy_url || '', elapsedSec: j.elapsed_sec,
      })),
      log: '',
    }
  }
}

function useBulkState() {
  const [, force] = useState(0)
  useEffect(() => {
    // userPrefs doesn't auto-load on import (would race keycloak init
    // and 401). Trigger the load now — this useEffect runs after the
    // component mounts, which is after keycloak is ready, so the
    // patched fetch can inject the Bearer token. Once the mirror is
    // populated, subscribePrefs will fire and re-hydrate the bulk state
    // with the correct dismissed flag (and flip _prefsReady so the
    // hydrate stops guarding on the uninitialised mirror).
    void refreshPrefs()
    _hydrateBulkState()
    const offSnap = subscribeBulk(() => {
      _hydrateBulkState()
      _notifyBulk()
      force(n => n + 1)
    })
    // Also re-hydrate when the user-prefs mirror updates. The dismissed
    // flag is read from the mirror, so when prefs load asynchronously
    // we need to re-check the dismissed flag and (if it now matches the
    // latest run) clear the report.
    const offPrefs = subscribePrefs(() => {
      _prefsReady = true
      _hydrateBulkState()
      _notifyBulk()
      force(n => n + 1)
    })
    return () => { offSnap(); offPrefs() }
  }, [])
  return _bulkState
}

const TYPE_LABELS: Record<TrainingType, string> = {
  'classification':    'Classification',
  'detection':         'Detection',
  'segmentation':      'Segmentation',
  'llm-text':          'LLM',
  'vlm-finetune':      'VLM',
  'self-supervised':   'Self-Sup',
  'anomaly-detection': 'Anomaly',
}

const TYPE_COLORS: Record<TrainingType, string> = {
  'classification':    '#6366f1',
  'detection':         '#f59e0b',
  'segmentation':      '#10b981',
  'llm-text':          '#8b5cf6',
  'vlm-finetune':      '#3b82f6',
  'self-supervised':   '#14b8a6',
  'anomaly-detection': '#ef4444',
}

const TYPE_ORDER: TrainingType[] = ['classification', 'detection', 'segmentation', 'anomaly-detection', 'llm-text', 'vlm-finetune', 'self-supervised']

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

  // Pull actual GPU free memory from Ray's per-node `gpus` array
  // (`memoryTotal - memoryUsed`, in MiB → GiB). This is the CORRECT
  // signal for "can a model fit on a single GPU" — `node.mem` is
  // system RAM and can be wildly different (e.g. 140 GB H200 HBM3e vs
  // 132 GB shared CPU). We track the max across all GPUs because Ray
  // schedules onto the most-free node, plus the per-model hardware
  // value in the catalog is the per-GPU requirement.
  const [gpuFreeGb, setGpuFreeGb] = useState<number | null>(null)
  const [gpuTotalGb, setGpuTotalGb] = useState<number | null>(null)
  useEffect(() => {
    fetch('/api/ray/nodes?view=summary')
      .then(r => r.json())
      .then(data => {
        const nodes: any[] = data?.data?.summary ?? []
        let maxFreeGiB = 0
        let maxTotalGiB = 0
        for (const node of nodes) {
          for (const g of (node.gpus || []) as any[]) {
            const total = Number(g.memoryTotal ?? 0)  // MiB
            const used  = Number(g.memoryUsed  ?? 0)  // MiB
            if (total <= 0) continue
            const freeGiB = (total - used) / 1024
            if (freeGiB > maxFreeGiB) maxFreeGiB = freeGiB
            const totalGiB = total / 1024
            if (totalGiB > maxTotalGiB) maxTotalGiB = totalGiB
          }
        }
        if (maxFreeGiB > 0)  setGpuFreeGb(Math.floor(maxFreeGiB))
        else                  setGpuFreeGb(null)
        if (maxTotalGiB > 0) setGpuTotalGb(Math.floor(maxTotalGiB))
      })
      .catch(() => {})
  }, [])

  function parseRequiredGb(hardware: string): number {
    const m = hardware.match(/(\d+)\s*GB/i)
    return m ? parseInt(m[1]) : 0
  }

  function isCompatible(opt: TrainOption): boolean {
    // BOTH conditions must be true:
    //   - opt.compatible: catalog says this model is supported on this
    //     backend (e.g. covidnet-cxr-3 is flagged compatible:false because
    //     it needs TF 1.15 which is EOL; the hardware check alone would
    //     wrongly include it whenever the cluster has enough VRAM)
    //   - hardware fits in current free GPU memory
    if (!opt.compatible) return false
    if (gpuFreeGb !== null) return parseRequiredGb(opt.hardware) <= gpuFreeGb
    return true
  }

  // Test runs (keyed by model row key) — backed by module store so the
  // poll loop keeps working even when the user navigates away.
  const runs = useStoreSnapshot()
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  // Per-row "copied" feedback so the Copy icon next to the training log
  // can flash a check for 1.5s after the user clicks it.
  const [copiedRowLog, setCopiedRowLog] = useState<Set<string>>(new Set())
  // Per-row training log fetched on-demand. The bulk-run WS snapshot
  // intentionally omits the log (bandwidth — 78 jobs × multi-KB each
  // every 2s would explode). When the user expands a row, we poll
  // /api/jobs/{jobId} every 2s and cache the result here so the
  // expanded panel can render the latest output.
  const [rowLogs, setRowLogs] = useState<Record<string, string>>({})

  // Bulk run control — derived from the WebSocket snapshot (running,
  // progress, report). stopRequested is local-only (it lives for the
  // current browser session and is cleared when the loop ends).
  const bulk = useBulkState()
  // Local-only — not persisted
  const setBulkStopRequested = (v: boolean) => { _bulkState.stopRequested = v; _notifyBulk() }
  const setBulkStream     = (v: BulkStreamEntry[] | ((prev: BulkStreamEntry[]) => BulkStreamEntry[])) => {
    bulk.stream = typeof v === 'function' ? v(bulk.stream) : v; _notifyBulk()
  }
  const bulkRunning       = bulk.running
  const bulkProgress      = bulk.progress
  const bulkReport        = bulk.report
  const bulkStream        = bulk.stream

  // Poll /api/jobs/{jobId} for every expanded row so the Training Log
  // panel shows live output for running jobs. The bulk-run WS snapshot
  // doesn't carry logs (intentional — would be too much data per push),
  // so we fetch on demand only for rows the user is actually looking at.
  useEffect(() => {
    const timers: ReturnType<typeof setInterval>[] = []
    for (const key of expanded) {
      const jobId = moduleRuns[key]?.jobId
      if (!jobId) continue
      const poll = async () => {
        const job = await _apiFetchJob(jobId)
        if (job && job.log != null) {
          setRowLogs(prev => prev[key] === job.log ? prev : { ...prev, [key]: job.log })
        }
      }
      void poll()
      timers.push(setInterval(poll, 2000))
    }
    return () => { for (const t of timers) clearInterval(t) }
  }, [expanded])

  // Pre-run config — picked once before the bulk loop starts
  const [bulkProvider, setBulkProvider] = useState<'ray' | 'modal'>('ray')
  const [bulkDeployEnabled, setBulkDeployEnabled] = useState(true)
  // GPUs per training job — e.g. 1 = sequential, 2-4 = data-parallel within
  // a single job (less jobs fit on the cluster at once but each trains
  // faster). Capped at 4 (cluster size on 4×H200 deployments).
  const [bulkGpusPerJob, setBulkGpusPerJob] = useState<number>(1)
  // Stop the loop the moment any step fails so the user can debug
  // the broken model before running the rest.
  const [bulkStopOnError, setBulkStopOnError] = useState(false)

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
    // Anomaly detection only needs 1+ normal image; labels are optional
    if (row.trainingType === 'anomaly-detection' && p.finished_task_number < 1) {
      return `Project "${p.title}" has 0 images — anomaly detection needs ≥1 normal image`
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
        num_gpus:      bulkGpusPerJob,
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
      // The bulk-run snapshot is updated by setRunLocal (which calls
      // apiUpsertBulkJob). The /ws/testall WebSocket will push the
      // updated state back within 2s.
    } catch (e: any) {
      const cur = moduleRuns[row.key]
      setRunLocal(row.key, {
        ...cur, status: 'error', error: e.message, finishedAt: Date.now() / 1000,
      })
    }
  }, [effectiveDataset, validateBeforeSubmit, bulkProvider, bulkGpusPerJob])

  // ── Deploy a freshly-trained model. Used by the bulk loop right after
  //    a successful training job completes (per-model train → deploy →
  //    next flow). Mirrors the DeployModels page's POST to /api/deploy. ─
  const deployModel = useCallback(async (row: ModelRow, provider: 'ray' | 'modal', jobId?: string | null): Promise<{ ok: boolean; url?: string; error?: string; elapsedSec: number }> => {
    const t0 = Date.now() / 1000
    try {
      const r = await fetch('/api/deploy', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          job_id:     jobId ?? null,
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

  // ── Stop a previously-deployed model actor on the Ray cluster ──────────
  // Used by the bulk loop to free the GPU that a previous deploy is
  // holding — otherwise after 4 deploys (on a 4-GPU cluster) every
  // subsequent training job gets stuck in PENDING waiting for resources
  // and the reconcile watchdog has to kill it. See _ray_serve_state
  // cleanup in the backend (POST /api/jobs/{job_id}/deploy-ray/stop).
  const stopDeploy = useCallback(async (jobId: string): Promise<{ ok: boolean; error?: string }> => {
    try {
      const r = await fetch(`/api/jobs/${jobId}/deploy-ray/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      if (!r.ok) {
        let detail = `HTTP ${r.status}`
        try { const d = await r.json(); detail = d?.detail || detail } catch {}
        return { ok: false, error: detail }
      }
      return { ok: true }
    } catch (e: any) {
      return { ok: false, error: e.message }
    }
  }, [])

  // ── Real-time updates: handled by the /ws/testall WebSocket at the
  //    top of this file. The server pushes a fresh snapshot every 2s
  //    and useStoreSnapshot() re-renders this view whenever it changes.

  // ── Wait until a given run reaches a terminal state (OK or error).
  //    Polls the job endpoint every 2s; resolves as soon as the run is
  //    no longer queued/running/submitting. Used by runAllCompatible so
  //    we never train two models in parallel. ──
  const waitForCompletion = useCallback(async (key: string, timeoutMs = 600_000): Promise<void> => {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (bulk.stopRequested) return
      const cur = moduleRuns[key]
      if (cur && (cur.status === 'completed' || cur.status === 'error')) return
      if (cur && cur.status === 'idle' && cur.error) return
      await new Promise(r => setTimeout(r, 2000))
    }
  }, [])

  // ── Run all compatible — strictly sequential: per-model TRAIN →
  //    DEPLOY → NEXT. The user picks Ray vs Modal once before the loop
  //    starts; every model goes to the same provider. ──────────────────
  const runAllCompatible = async () => {
    if (bulkRunning) return
    setBulkStopRequested(false)
    setBulkStream([])

    // Create the bulk_runs row first so the server knows we're starting
    // a new run. From this point on, every per-model state change goes
    // to /api/bulk-runs/{id}/jobs and the WebSocket will broadcast.
    let runId: string
    try {
      const targets = filteredRows.filter(r => isCompatible(r.option))
      runId = await apiCreateBulkRun(bulkProvider, bulkDeployEnabled, targets.length)
      _currentRunId = runId
    } catch (e: any) {
      alert(`Could not start bulk run: ${e.message}`)
      return
    }

    const targets = filteredRows.filter(r => isCompatible(r.option))
    const failures: BulkFailure[] = []
    const deployed: BulkDeployed[] = []
    let ok = 0
    let deployedOk = 0
    let stopped = false
    const logLines: string[] = []

    // Track the previous iteration's deployed job_id so we can stop its
    // Ray Serve actor before the next training job starts.
    let prevDeployedJobId: string | null = null

    // Pre-flight: stop any deploys left over from a previous bulk run
    if (bulkDeployEnabled) {
      try {
        const r = await fetch('/api/ray/serve/stop-all', { method: 'POST' })
        const d = r.ok ? await r.json() : null
        if (d) {
          logLines.push(`[pre-flight] Stopped ${d.stopped ?? 0} leftover deploys (${d.failed ?? 0} failed)`)
        }
      } catch (e: any) {
        logLines.push(`[pre-flight] stop-all failed: ${e.message}`)
      }
    }

    let idx = 0
    for (const row of targets) {
      if (bulk.stopRequested) { stopped = true; break }
      idx += 1
      _bulkState.progress = { current: idx, total: targets.length, label: row.option.label, stage: 'train' }
      _notifyBulk()

      // Free the GPU that the previous deploy is holding
      if (bulkDeployEnabled && prevDeployedJobId) {
        const stop = await stopDeploy(prevDeployedJobId)
        logLines.push(`[${idx}/${targets.length}]  ↻  stopped prev deploy ${prevDeployedJobId.slice(0, 8)}… (${stop.ok ? 'ok' : 'fail'})`)
        prevDeployedJobId = null
      }

      const existing = moduleRuns[row.key]
      if (existing && (existing.status === 'queued' || existing.status === 'running' || existing.status === 'submitting')) {
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
        if (bulkStopOnError) { stopped = true; break }
        continue
      }
      ok += 1
      logLines.push(`[${idx}/${targets.length}]  ✓  TRAIN  ${row.option.label}`)
      setBulkStream(s => [...s, { label: row.option.label, status: 'ok' }])

      // Update server-side counts
      void apiUpdateBulkRun(runId, { ok, failed: failures.length })

      // TRAIN → DEPLOY step
      if (bulkDeployEnabled) {
        if (bulk.stopRequested) { stopped = true; break }
        _bulkState.progress = { current: idx, total: targets.length, label: row.option.label, stage: 'deploy' }
        _notifyBulk()
        setBulkStream(s => [...s, { label: row.option.label, status: 'deploy' }])
        // Mark the row as "deploying" so the StatusCell shows the
        // rocket/spinner state and the per-row badge matches the
        // bulk-progress header.
        const existingRun = moduleRuns[row.key]
        if (existingRun) {
          setRunLocal(row.key, { ...existingRun, status: 'deploying' })
        }
        void apiUpsertBulkJob(runId, row.key, {
          job_id:      final?.jobId ?? null,
          status:      'deploying',
          error:       null,
          elapsed_sec: 0,
          deployed:    0,
          deploy_url:  null,
        })
        const dep = await deployModel(row, bulkProvider, final?.jobId)
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
          // Persist the deploy result to the server with the new
          // 'deployed' status so subsequent WS pushes keep the row
          // in that state (not collapsed back to 'completed').
          const deployedRun = moduleRuns[row.key]
          if (deployedRun) {
            setRunLocal(row.key, { ...deployedRun, status: 'deployed' })
          }
          void apiUpsertBulkJob(runId, row.key, {
            job_id:      final?.jobId ?? null,
            status:      'deployed',
            error:       null,
            elapsed_sec: dep.elapsedSec,
            deployed:    1,
            deploy_url:  dep.url ?? null,
          })
          void apiUpdateBulkRun(runId, { deployed_count: deployedOk })
          if (final?.jobId) prevDeployedJobId = final.jobId
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
          const errRun = moduleRuns[row.key]
          if (errRun) {
            setRunLocal(row.key, { ...errRun, status: 'error', error: errMsg, finishedAt: Date.now() / 1000 })
          }
          void apiUpsertBulkJob(runId, row.key, {
            job_id:      final?.jobId ?? null,
            status:      'error',
            error:       errMsg,
            elapsed_sec: 0,
            deployed:    0,
            deploy_url:  null,
          })
          if (bulkStopOnError) { stopped = true; break }
        }
      }
    }

    // Persist final state to the server. The next WS push will rebuild
    // _bulkState from these authoritative rows.
    try {
      await apiUpdateBulkRun(runId, {
        finished_at: Date.now() / 1000,
        stopped:     stopped,
        total:       targets.length,
        ok,
        failed:      failures.length,
        deployed_count: deployedOk,
      })
    } catch { /* best-effort */ }
    _bulkState.progress = null
    _bulkState.running = false
    _bulkState.stopRequested = false
    _notifyBulk()
  }

  // ── Aggregate stats ──
  // NOTE: not wrapped in useMemo — `runs` is `moduleRuns` which is
  // mutated in place by _rebuildModuleRunsFromSnapshot(), so its object
  // reference never changes and useMemo([runs]) would never recompute.
  // The calculation is O(N) over the row count (≤~80 in practice), so
  // recomputing on every render is fine.
  const stats = (() => {
    const list = Object.values(runs)
    return {
      total:     list.length,
      pending:   list.filter(r => r.status === 'submitting' || r.status === 'queued' || r.status === 'running' || r.status === 'deploying').length,
      ok:        list.filter(r => r.status === 'completed' || r.status === 'deployed').length,
      err:       list.filter(r => r.status === 'error').length,
    }
  })()

  const toggleExpand = (key: string) => {
    setExpanded(prev => {
      const s = new Set(prev)
      s.has(key) ? s.delete(key) : s.add(key)
      return s
    })
  }

  return (
    <div>
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

          {/* Live stream — surfaces each model's status as it completes
              so the user sees failures in real time during long runs. */}
          {bulkStream.length > 0 && bulkRunning && (
            <div style={{ marginTop: 10, borderRadius: 6, background: 'var(--bg-elevated)' }}>
              <div style={{ display: 'flex', gap: 12, padding: '6px 10px', fontSize: 10.5, color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><CheckCircle2 size={10} color="var(--success)" /> train OK</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Rocket size={10} color="#3b82f6" /> deploying</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><CheckCircle2 size={10} color="#3b82f6" /> deployed</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><XCircle size={10} color="var(--danger)" /> error</span>
              </div>
              <div style={{ maxHeight: 120, overflowY: 'auto', padding: '6px 10px', fontSize: 11.5, fontFamily: 'var(--font-mono)' }}>
              {bulkStream.slice(-20).map((s, i) => (
                <div key={`${s.label}-${i}`} style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '1px 0' }}>
                  {s.status === 'ok'
                    ? <CheckCircle2 size={11} color="var(--success)" />
                    : s.status === 'deploy'
                    ? <Rocket size={11} color="#3b82f6" />
                    : s.status === 'deployed'
                    ? <CheckCircle2 size={11} color="#3b82f6" />
                    : <XCircle size={11} color="var(--danger)" />}
                  <span style={{ color: s.status === 'ok' ? 'var(--text-secondary)' : s.status === 'err' ? 'var(--danger)' : 'var(--text-primary)', flex: 1 }}>{s.label}</span>
                  {s.status === 'ok' && <span style={{ color: 'var(--success)', fontSize: 10 }}>train ✓</span>}
                  {s.status === 'deploy' && <span style={{ color: '#3b82f6', fontSize: 10 }}>deploying…</span>}
                  {s.status === 'deployed' && <span style={{ color: '#3b82f6', fontSize: 10 }}>deployed ✓</span>}
                  {s.status === 'err' && s.error && (
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
            </div>
          )}

          {/* Post-run report — summary card + failed-list + export. */}
          {bulkReport && !bulkRunning && <BulkReportPanel report={bulkReport} setExpanded={setExpanded} onClose={() => {
            // Remember which run this report belongs to so a follow-up
            // WS push doesn't re-hydrate it from the snapshot. Persist
            // to localStorage so the dismissal survives a page refresh.
            // Key by startedAt (unix seconds) — stable across snapshot
            // re-emissions and identical to what _hydrateBulkState matches
            // against on the next page load.
            _bulkState.dismissedRunId = String(bulkReport.startedAt)
            _writeDismissedRunId(_bulkState.dismissedRunId)
            _bulkState.report = null
            _notifyBulk()
          }} />}
        </div>

        {/* Always-visible toolbar (when not running) — just Reset.
              Split out from the pre-run config below so the button
              stays reachable when a report card is showing. */}
        {!bulkRunning && (
          <div style={{
            marginTop: 14, padding: 8, borderRadius: 8,
            background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
            display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 8,
          }}>
            <button
              className="btn btn-secondary"
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={async () => {
                // Race-safe reset:
                //   1. Bump _resetInFlight so the next 1-2 WS pushes
                //      (which may still carry the pre-DELETE rows)
                //      are ignored by both _rebuildModuleRunsFromSnapshot
                //      and _hydrateBulkState.
                //   2. Clear local state immediately for snappy feedback.
                //   3. DELETE the server-side rows. Once this resolves
                //      (≤200ms typical), the next WS push will be empty
                //      and the rebuild will land in the "empty snapshot"
                //      branch which clears everything.
                //   4. Drop the flag.
                _resetInFlight++
                try {
                  for (const k of Object.keys(moduleRuns)) delete moduleRuns[k]
                  _bulkState.report = null
                  _bulkState.progress = null
                  _bulkState.stream = []
                  _notifyBulk()
                  await fetch('/api/bulk-runs', { method: 'DELETE' })
                } catch { /* best-effort — local state is already cleared */ }
                finally { _resetInFlight-- }
              }}
              title="Clear all results"
            >
              <RefreshCw size={14} /> Reset
            </button>
          </div>
        )}

        {/* Pre-run config + Run button — hidden when a report is
              showing (user has to dismiss the report or reset first). */}
        {!bulkRunning && !bulkReport && (
          <div style={{
            marginTop: 14, padding: 12, borderRadius: 8,
            background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
            display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap',
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
            <div style={{ width: 1, height: 22, background: 'var(--border-subtle)' }} />

            {/* GPUs per job — 1 = sequential, 2-4 = data-parallel within job */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
              <Cpu size={12} />
              <span style={{ fontWeight: 600 }}>GPUs/job:</span>
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              {[1, 2, 4].map(n => {
                const active = bulkGpusPerJob === n
                return (
                  <button
                    key={n}
                    onClick={() => setBulkGpusPerJob(n)}
                    style={{
                      padding: '5px 10px', borderRadius: 6,
                      border: `1px solid ${active ? 'var(--primary)' : 'var(--border-default)'}`,
                      background: active ? 'var(--primary-dim)' : 'transparent',
                      color: active ? 'var(--primary-hover)' : 'var(--text-secondary)',
                      cursor: 'pointer',
                      fontSize: 11.5, fontWeight: 500,
                      fontFamily: 'var(--font-mono)', minWidth: 32,
                    }}
                    title={
                      n === 1 ? 'Sequential — one job at a time' :
                      n === 2 ? 'Data-parallel — 2 GPUs per job (half the jobs fit at once)' :
                      'Data-parallel — 4 GPUs per job (only ~1 job fits at a time)'
                    }
                  >
                    {n}
                  </button>
                )
              })}
            </div>
            <div style={{ width: 1, height: 22, background: 'var(--border-subtle)' }} />

            {/* Stop on first error — debug the broken model before running the rest */}
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)', cursor: 'pointer', userSelect: 'none' }}>
              <input
                type="checkbox"
                checked={bulkStopOnError}
                onChange={e => setBulkStopOnError(e.target.checked)}
                style={{ width: 13, height: 13, cursor: 'pointer', accentColor: 'var(--primary)' }}
              />
              <span style={{ fontWeight: 500 }}>Stop on first error</span>
            </label>

            {/* Push actions to the right */}
            <div style={{ flex: 1 }} />
            {gpuFreeGb !== null && (
              <div
                title={
                  gpuTotalGb
                    ? `Largest GPU on the Ray cluster has ${gpuFreeGb} GB free out of ${gpuTotalGb} GB total — drives the isCompatible() check`
                    : `Largest GPU on the Ray cluster has ${gpuFreeGb} GB free — drives the isCompatible() check`
                }
                style={{
                  display: 'flex', alignItems: 'center', gap: 4,
                  padding: '4px 9px', borderRadius: 14,
                  background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
                  fontSize: 11, fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                }}
              >
                <Cpu size={11} />
                <span><strong style={{ color: 'var(--text-primary)' }}>{gpuFreeGb}</strong> GB GPU free</span>
              </div>
            )}
            <button
              className="btn btn-primary"
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={runAllCompatible}
              disabled={datasetsLoading || filteredRows.length === 0}
            >
              <Play size={14} /> Run {filteredRows.filter(r => isCompatible(r.option)).length} Compatible
            </button>
          </div>
        )}

        {/* During-run banner: progress + Stop button. The pre-run config
            card hides itself while a run is in progress. */}
        {bulkRunning && (
          <div style={{
            marginTop: 14, padding: 12, borderRadius: 8,
            background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
            display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
          }}>
            {bulkProgress && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 220 }}>
                <Loader2 size={13} color="var(--primary)" className="animate-spin" />
                <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  {bulkProgress.stage === 'deploy' ? 'Deploying' : 'Training'}{' '}
                  <strong style={{ color: 'var(--text-primary)' }}>{bulkProgress.label}</strong>
                  <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>
                    ({bulkProgress.current} / {bulkProgress.total})
                  </span>
                </span>
                <div style={{ flex: 1, maxWidth: 220, height: 4, background: 'var(--bg-base)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{
                    width: `${(bulkProgress.current / bulkProgress.total) * 100}%`,
                    height: '100%', background: 'var(--primary)', transition: 'width 0.3s ease',
                  }} />
                </div>
              </div>
            )}
            <div style={{ flex: 1 }} />
            <button
              className="btn btn-sm"
              style={{ background: 'var(--danger)', color: '#fff', border: 'none', display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={async () => {
                // Three things, in order of responsiveness:
                //   1. Tell the server right now — survives a page
                //      refresh, tab close, or a crashed JS loop. The
                //      next WS push will reflect the stopped state and
                //      unstick the UI even if the in-browser loop is
                //      not running anymore.
                //   2. Flip the local stop flag — the in-browser
                //      runAllCompatible loop checks it every iteration
                //      and every 2s inside waitForCompletion, so it
                //      breaks within ~2s and finalises the report.
                //   3. Clear the progress bar so the user sees
                //      immediate feedback that the stop was registered
                //      (the next WS push replaces it with the
                //      finished-report card).
                if (_currentRunId) {
                  try { await apiStopBulkRun(_currentRunId) } catch { /* best-effort */ }
                }
                _bulkState.stopRequested = true
                _bulkState.progress = null
                _notifyBulk()
              }}
            >
              <StopCircle size={14} /> Stop Bulk
            </button>
          </div>
        )}
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
            <div key={row.key} data-row-key={row.key} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
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
                    {(run?.log || rowLogs[row.key]) && (
                      <div style={{ gridColumn: '1 / -1' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                          <Terminal size={11} /> Training Log
                          <button
                            className="btn-icon"
                            style={{ marginLeft: 'auto', padding: 2, color: 'var(--text-muted)' }}
                            title="Copy log to clipboard"
                            onClick={() => {
                              const logText = rowLogs[row.key] ?? run.log
                              navigator.clipboard.writeText(logText).then(() => {
                                setCopiedRowLog(prev => new Set(prev).add(row.key))
                                setTimeout(() => setCopiedRowLog(prev => { const n = new Set(prev); n.delete(row.key); return n }), 1500)
                              }).catch(() => { /* clipboard blocked */ })
                            }}
                          >
                            {copiedRowLog.has(row.key) ? <Check size={11} /> : <Copy size={11} />}
                          </button>
                        </div>
                        <pre style={{ background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', borderRadius: 6, padding: 8, fontSize: 10.5, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 200, overflowY: 'auto', margin: 0 }}>
                          {(rowLogs[row.key] ?? run.log ?? '').slice(-2000) || '(empty)'}
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
    deploying:  { icon: Rocket,        color: '#3b82f6',               label: 'Deploying' },
    deployed:   { icon: Rocket,        color: 'var(--success)',        label: 'Deployed' },
    completed:  { icon: CheckCircle2,  color: 'var(--success)',        label: 'OK' },
    error:      { icon: XCircle,       color: 'var(--danger)',         label: 'Error' },
  }
  const c = map[run.status]
  const Icon = c.icon
  const isAnim = run.status === 'submitting' || run.status === 'running' || run.status === 'deploying'
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
  const [copiedRunLog, setCopiedRunLog] = useState(false)

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
  // 3 disjoint categories that always sum to `total`:
  //   • Failed       — red
  //   • OK           — green (includes the deployed subset)
  //   • Not completed — gray (only non-zero when stopped early)
  // Deployed is a *subset* of OK, not a separate bar segment — shown
  // as a sub-label on the OK chip when > 0 so the legend never
  // disappears (e.g. "all OK, no deploy" or "all deployed" both show
  // an OK chip, just with a different sub-text).
  const notCompleted = Math.max(0, report.total - report.ok - report.failed)
  const seg = (n: number) => report.total > 0 ? (n / report.total) * 100 : 0

  return (
    <div style={{
      marginTop: 14, padding: 16, borderRadius: 10,
      background: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
    }}>
      {/* ── Stacked progress bar — 3 segments (Failed | OK | gray track).
            Deployed is a sub-state of OK and not a separate color band
            (would shrink the green bar visually for no informational
            gain — the count is in the legend). ── */}
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={report.total}
        aria-valuenow={report.ok + report.failed}
        aria-label={`Bulk run outcome: ${report.ok} trained, ${report.deployedCount} deployed, ${report.failed} failed, of ${report.total}`}
        style={{
          height: 8, borderRadius: 4, overflow: 'hidden',
          background: 'var(--bg-elevated)',
          display: 'flex', flexDirection: 'row',
        }}
      >
        {report.failed > 0 && (
          <div style={{ width: `${seg(report.failed)}%`, background: 'var(--danger)' }} />
        )}
        {report.ok > 0 && (
          <div style={{ width: `${seg(report.ok)}%`, background: 'var(--success)' }} />
        )}
        {/* The unfilled "not completed" portion is the bar's own background. */}
      </div>

      {/* ── Legend row — 3 category chips. Each is hidden if its count
            is 0 EXCEPT the OK chip, which always shows when the run
            produced any successes (so "all OK" never leaves an empty
            legend). Deployed count rides along as a sub-label on the
            OK chip. ── */}
      <div style={{
        marginTop: 10, display: 'flex', flexWrap: 'wrap', alignItems: 'center',
        gap: '4px 18px', fontSize: 11.5, color: 'var(--text-secondary)',
        fontVariantNumeric: 'tabular-nums',
      }}>
        {report.failed > 0 && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--danger)', flexShrink: 0 }} />
            <span style={{ color: 'var(--text-muted)' }}>Failed</span>
            <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{report.failed}</strong>
          </span>
        )}
        {report.ok > 0 && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--success)', flexShrink: 0 }} />
            <span style={{ color: 'var(--text-muted)' }}>OK</span>
            <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{report.ok}</strong>
            {report.deployedCount > 0 && (
              <span style={{ color: 'var(--text-muted)', fontSize: 10.5 }}>
                ({report.deployedCount} deployed)
              </span>
            )}
          </span>
        )}
        {notCompleted > 0 && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', flexShrink: 0 }} />
            <span style={{ color: 'var(--text-muted)' }}>Not completed</span>
            <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{notCompleted}</strong>
          </span>
        )}
        <span style={{ marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: 'var(--text-muted)' }}>Total</span>
          <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{report.total}</strong>
          <span style={{ color: 'var(--text-muted)' }}>·</span>
          <span style={{ color: 'var(--text-muted)' }}>{elapsedSec}s</span>
          <span style={{ color: 'var(--text-muted)' }}>·</span>
          <span style={{ color: okPct >= 80 ? 'var(--success)' : okPct >= 50 ? 'var(--warning)' : 'var(--danger)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {okPct}% success
          </span>
        </span>
      </div>

      {/* ── Header row: title + actions. ── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginTop: 14 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            {report.failed > 0
              ? <XCircle size={16} color="var(--danger)" />
              : report.deployEnabled && report.deployedCount > 0
                ? <Rocket size={16} color="var(--info)" />
                : <CheckCircle2 size={16} color="var(--success)" />}
            <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>
              {report.stopped
                ? `Bulk run stopped · ${report.ok} OK${report.deployEnabled ? `, ${report.deployedCount} deployed` : ''}, ${report.failed} failed (out of ${report.total} started)`
                : report.failed > 0
                  ? `Bulk run finished · ${report.failed} of ${report.total} failed`
                  : report.deployEnabled
                    ? `Bulk run finished · all ${report.total} trained + ${report.deployedCount} deployed to ${report.provider}`
                    : `Bulk run finished · all ${report.ok} models OK`}
            </span>
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
                    onClick={() => {
                      setExpanded(prev => new Set(prev).add(f.key))
                      // Scroll the matching row into view — the table is
                      // far below the report panel, so just expanding
                      // without scrolling leaves the user staring at
                      // "nothing happened".
                      setTimeout(() => {
                        const el = document.querySelector(`[data-row-key="${f.key}"]`)
                        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
                      }, 50)
                    }}
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
            <button
              className="btn-icon"
              style={{ marginLeft: 'auto', padding: 2, color: 'var(--text-muted)' }}
              title="Copy run log to clipboard"
              onClick={() => {
                navigator.clipboard.writeText(report.log).then(() => {
                  setCopiedRunLog(true)
                  setTimeout(() => setCopiedRunLog(false), 1500)
                }).catch(() => { /* clipboard blocked */ })
              }}
            >
              {copiedRunLog ? <Check size={11} /> : <Copy size={11} />}
            </button>
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
