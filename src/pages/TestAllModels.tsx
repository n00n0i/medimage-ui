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
  Database, AlertCircle, StopCircle, Filter,
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
  const [showIncompatible, setShowIncompatible] = useState(false)

  // Test runs (keyed by model row key) — backed by module store so the
  // poll loop keeps working even when the user navigates away.
  const runs = useStoreSnapshot()
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  // Bulk run control — also module-scoped so the bulk loop survives
  // navigation. We mirror into useState for re-render.
  const [bulkRunning, setBulkRunning] = useState(false)
  const [bulkStopRequested, setBulkStopRequested] = useState(false)
  const [bulkProgress, setBulkProgress] = useState<{ current: number; total: number; label: string } | null>(null)

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
    if (!showIncompatible) rows = rows.filter(r => r.option.compatible)
    if (search.trim()) {
      const q = search.toLowerCase()
      rows = rows.filter(r =>
        r.option.label.toLowerCase().includes(q) ||
        r.option.model.toLowerCase().includes(q) ||
        r.option.engine.toLowerCase().includes(q)
      )
    }
    return rows
  }, [allRows, typeFilter, search, showIncompatible])

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

  // ── Submit single test ──
  const submitTest = useCallback(async (row: ModelRow) => {
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
        cluster:       'ray',
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
  }, [effectiveDataset, validateBeforeSubmit])

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

  // ── Run all compatible — strictly sequential: submit, then wait for
  //    the job to finish (OK or error) before moving to the next model. ──
  const runAllCompatible = async () => {
    if (bulkRunning) return
    setBulkRunning(true)
    setBulkStopRequested(false)
    bulkStopRequestedRef.current = false
    const targets = filteredRows.filter(r => r.option.compatible)
    let idx = 0
    for (const row of targets) {
      if (bulkStopRequestedRef.current) break
      idx += 1
      setBulkProgress({ current: idx, total: targets.length, label: row.option.label })

      // Skip rows that already finished in a previous run, but re-run
      // anything that errored so the user can retry without resetting
      // the whole table.
      const existing = moduleRuns[row.key]
      if (existing && (existing.status === 'queued' || existing.status === 'running' || existing.status === 'submitting')) {
        // Wait for the in-flight one first
        await waitForCompletion(row.key)
        continue
      }
      if (existing && existing.status === 'completed') continue

      await submitTest(row)
      // After submit, wait until this specific job reaches a terminal
      // state before submitting the next one. This keeps the Ray
      // cluster fed with one training job at a time and ensures error
      // messages are attributed to the model that produced them.
      await waitForCompletion(row.key)
    }
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
        </div>
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
              <Play size={14} /> Run {filteredRows.filter(r => r.option.compatible).length} Compatible
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
          Show incompatible ({allRows.length - allRows.filter(r => r.option.compatible).length})
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
          const isIncompatible = !row.option.compatible

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
