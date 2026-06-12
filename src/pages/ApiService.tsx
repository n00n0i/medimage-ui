import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import {
  Zap, Key, Copy, Trash2, Plus, RefreshCw, CheckCircle2, Code2,
  Globe, Lock, Eye, EyeOff, ExternalLink, BookOpen, Wifi, WifiOff,
} from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────

interface ApiKey {
  id: string
  name: string
  key: string
  created: string
  lastUsed: string | null
  calls: number
}

interface DeployedModel {
  id: string
  name: string
  model: string
  training_type: string
  engine: string
  created_at: number
  ray_serve_url: string
  modal_url: string
  inference_provider: string
}

const TT_LABELS: Record<string, string> = {
  classification:   'Classification',
  detection:        'Detection',
  segmentation:     'Segmentation',
  'vlm-finetune':   'VLM Fine-tune',
  'export-edge':    'Edge Export',
  'llm-text':       'LLM / Text',
  'self-supervised':'Self-supervised',
}

const TT_COLORS: Record<string, string> = {
  classification:   '#6366f1',
  detection:        '#0ea5e9',
  segmentation:     '#10b981',
  'vlm-finetune':   '#8b5cf6',
  'export-edge':    '#f59e0b',
  'llm-text':       '#ec4899',
  'self-supervised':'#14b8a6',
}

const METHOD_STYLE: Record<string, React.CSSProperties> = {
  GET:    { background: '#0ea5e920', color: '#0ea5e9' },
  POST:   { background: '#10b98120', color: '#10b981' },
  DELETE: { background: '#ef444420', color: '#ef4444' },
  PUT:    { background: '#f59e0b20', color: '#f59e0b' },
}

const API_REFERENCE = [
  {
    group: 'Inference',
    endpoints: [
      { method: 'POST', path: '/api/inference',             desc: 'Run inference on a trained model (image upload or prompt)' },
    ],
  },
  {
    group: 'Models',
    endpoints: [
      { method: 'POST', path: '/api/models/import',         desc: 'Import a pretrained model (HuggingFace, URL, built-in)' },
    ],
  },
  {
    group: 'Jobs',
    endpoints: [
      { method: 'GET',    path: '/api/jobs',                desc: 'List all training / import jobs' },
      { method: 'GET',    path: '/api/jobs/{job_id}',       desc: 'Get job status, progress, and logs' },
      { method: 'DELETE', path: '/api/jobs/{job_id}',       desc: 'Cancel or hide a job' },
    ],
  },
  {
    group: 'Training',
    endpoints: [
      { method: 'POST', path: '/api/train/{project_id}',    desc: 'Start a training run for a project' },
    ],
  },
  {
    group: 'Datasets',
    endpoints: [
      { method: 'POST', path: '/api/datasets/import-hf',           desc: 'Import a HuggingFace dataset into MinIO bucket' },
      { method: 'GET',  path: '/api/datasets/import-hf/{job_id}',  desc: 'Poll HF import job status' },
      { method: 'GET',  path: '/api/datasets/buckets',             desc: 'List MinIO buckets' },
      { method: 'POST', path: '/api/text-datasets/upload',         desc: 'Upload text / JSONL dataset file' },
      { method: 'GET',  path: '/api/text-datasets',                desc: 'List uploaded text datasets' },
      { method: 'DELETE', path: '/api/text-datasets/{ds_id}',      desc: 'Delete a text dataset' },
    ],
  },
  {
    group: 'Projects',
    endpoints: [
      { method: 'GET',  path: '/api/projects',              desc: 'List Label Studio projects' },
    ],
  },
  {
    group: 'Settings',
    endpoints: [
      { method: 'GET',  path: '/api/settings/{key}',        desc: 'Get a platform setting value' },
      { method: 'POST', path: '/api/settings/{key}',        desc: 'Set a platform setting value' },
    ],
  },
  {
    group: 'System',
    endpoints: [
      { method: 'GET',  path: '/api/modal/status',          desc: 'Modal.com GPU cluster status' },
      { method: 'POST', path: '/api/modal/start',           desc: 'Start Modal GPU cluster' },
      { method: 'POST', path: '/api/modal/stop',            desc: 'Stop Modal GPU cluster' },
      { method: 'GET',  path: '/healthz',                   desc: 'Backend health check' },
    ],
  },
]

// ── Helpers ───────────────────────────────────────────────────────────────

function genKey(): string {
  const arr = new Uint8Array(24)
  crypto.getRandomValues(arr)
  return 'mia_' + Array.from(arr).map(b => b.toString(16).padStart(2, '0')).join('')
}

function loadKeys(): ApiKey[] {
  try { return JSON.parse(localStorage.getItem('mia_api_keys') || '[]') } catch { return [] }
}

function saveKeys(keys: ApiKey[]) {
  localStorage.setItem('mia_api_keys', JSON.stringify(keys))
}

function fmtDate(ts: number) {
  return new Date(ts * 1000).toLocaleDateString('th-TH', { day: '2-digit', month: 'short', year: 'numeric' })
}

// ── Model Card sub-component ───────────────────────────────────────────────

function ModelCard({ m, dimmed, serveStatus, onSelect }: {
  m: DeployedModel
  dimmed: boolean
  serveStatus: Record<string, boolean | null>
  onSelect: (m: DeployedModel) => void
}) {
  const color = TT_COLORS[m.training_type] ?? '#6366f1'
  const hasRay   = m.inference_provider === 'ray'   && !!m.ray_serve_url
  const hasModal = m.inference_provider === 'modal' && !!m.modal_url
  const hasAny   = hasRay || hasModal
  const online = hasAny ? serveStatus[m.id] : undefined
  const isOnline = online === true
  const isOffline = hasAny && online === false
  const isLoadingStatus = hasAny && online === null
  const clickable = !dimmed && (!hasAny || isOnline)

  return (
    <button
      onClick={() => clickable && onSelect(m)}
      style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
        borderRadius: 12, padding: '18px 18px',
        cursor: clickable ? 'pointer' : 'not-allowed',
        opacity: dimmed ? 0.45 : isOffline ? 0.55 : 1,
        textAlign: 'left',
        transition: 'border-color 0.15s, box-shadow 0.15s, opacity 0.15s',
        display: 'flex', flexDirection: 'column', gap: 12,
      }}
      onMouseEnter={e => { if (clickable) { (e.currentTarget as HTMLElement).style.borderColor = color; (e.currentTarget as HTMLElement).style.boxShadow = `0 0 0 3px ${color}18` } }}
      onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-default)'; (e.currentTarget as HTMLElement).style.boxShadow = 'none' }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ width: 42, height: 42, borderRadius: 10, background: color + '18', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <Zap size={20} color={color} />
        </div>
        {hasRay ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, fontWeight: 700, padding: '3px 7px', borderRadius: 20, flexShrink: 0, background: '#f59e0b20', color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Ray</div>
        ) : hasModal ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, fontWeight: 700, padding: '3px 7px', borderRadius: 20, flexShrink: 0, background: '#8b5cf620', color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Modal</div>
        ) : null}
        {hasAny && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, fontWeight: 600, padding: '3px 7px', borderRadius: 20, flexShrink: 0, background: isLoadingStatus ? 'var(--bg-elevated)' : isOnline ? '#10b98120' : '#ef444418', color: isLoadingStatus ? 'var(--text-muted)' : isOnline ? '#10b981' : '#ef4444' }}>
            {isLoadingStatus ? (
              <RefreshCw size={9} style={{ animation: 'spin 1s linear infinite' }} />
            ) : isOnline ? (
              <Wifi size={9} />
            ) : (
              <WifiOff size={9} />
            )}
            {isLoadingStatus ? '…' : isOnline ? 'Online' : 'Offline'}
          </div>
        )}
      </div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.name}</div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.model}</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 'auto' }}>
        <span style={{ fontSize: 10, padding: '2px 7px', borderRadius: 4, background: color + '22', color, fontWeight: 600 }}>
          {TT_LABELS[m.training_type] ?? m.training_type}
        </span>
        <code style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 4 }}>
          {m.id}
        </code>
      </div>
    </button>
  )
}

// ── Main Page ──────────────────────────────────────────────────────────────

export default function ApiService() {
  const [models, setModels] = useState<DeployedModel[]>([])
  const [serveStatus, setServeStatus] = useState<Record<string, boolean | null>>({})
  const [loading, setLoading] = useState(true)
  const [apiKeys, setApiKeys] = useState<ApiKey[]>(loadKeys)
  const [newKeyName, setNewKeyName] = useState('')
  const [creating, setCreating] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [revealId, setRevealId] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<DeployedModel | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const statusIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 3000)
  }

  const fetchModels = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/jobs?view=models')
      const data = await res.json()
      const completed = (data.jobs ?? []) as DeployedModel[]
      setModels(completed)
      // init status as null (loading) for models with a real deployment
      const initial: Record<string, boolean | null> = {}
      completed.forEach(m => {
        if (m.inference_provider === 'ray'   && m.ray_serve_url) initial[m.id] = null
        if (m.inference_provider === 'modal' && m.modal_url)    initial[m.id] = null
      })
      setServeStatus(initial)
    } catch { setModels([]) }
    setLoading(false)
  }, [])

  const checkServeStatus = useCallback(async (modelList: DeployedModel[]) => {
    const rayModels   = modelList.filter(m => m.inference_provider === 'ray'   && m.ray_serve_url)
    const modalModels = modelList.filter(m => m.inference_provider === 'modal' && m.modal_url)
    if (rayModels.length === 0 && modalModels.length === 0) return
    const results = await Promise.allSettled([
      ...rayModels.map(m   => fetch(`/api/jobs/${m.id}/ray-status`)  .then(r => r.json()).then(d => ({ id: m.id, online: d.online as boolean, kind: 'ray'   as const }))),
      ...modalModels.map(m => fetch(`/api/jobs/${m.id}/modal-status`).then(r => r.ok ? r.json() : { online: false }).then(d => ({ id: m.id, online: d.online as boolean, kind: 'modal' as const }))),
    ])
    setServeStatus(prev => {
      const next = { ...prev }
      results.forEach(r => { if (r.status === 'fulfilled') next[r.value.id] = r.value.online })
      return next
    })
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  // poll status every 30s when models are loaded
  useEffect(() => {
    if (models.length === 0) return
    checkServeStatus(models)
    statusIntervalRef.current = setInterval(() => checkServeStatus(models), 30_000)
    return () => { if (statusIntervalRef.current) clearInterval(statusIntervalRef.current) }
  }, [models, checkServeStatus])

  const createKey = () => {
    if (!newKeyName.trim()) return
    const newKey: ApiKey = {
      id: crypto.randomUUID(),
      name: newKeyName.trim(),
      key: genKey(),
      created: new Date().toISOString(),
      lastUsed: null,
      calls: 0,
    }
    const updated = [...apiKeys, newKey]
    setApiKeys(updated)
    saveKeys(updated)
    setNewKeyName('')
    setCreating(false)
    setRevealId(newKey.id)
    showToast('API key created')
  }

  const deleteKey = (id: string) => {
    const key = apiKeys.find(k => k.id === id)
    if (!confirm(`Revoke API key "${key?.name ?? id}"?\n\nKey นี้จะใช้งานไม่ได้ทันที`)) return
    const updated = apiKeys.filter(k => k.id !== id)
    setApiKeys(updated)
    saveKeys(updated)
    showToast('Key revoked')
  }

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 2000)
    })
  }

  const baseUrl = window.location.origin

  const { onlineModels, offlineModels, undeployedModels } = useMemo(() => {
    const isOnline = (m: DeployedModel) => {
      const hasRay = m.inference_provider === 'ray' && !!m.ray_serve_url
      const hasModal = m.inference_provider === 'modal' && !!m.modal_url
      const hasAny = hasRay || hasModal
      return hasAny && serveStatus[m.id] === true
    }
    const hasDeployment = (m: DeployedModel) => {
      const hasRay = m.inference_provider === 'ray' && !!m.ray_serve_url
      const hasModal = m.inference_provider === 'modal' && !!m.modal_url
      return hasRay || hasModal
    }
    return {
      onlineModels: models.filter(m => isOnline(m)),
      offlineModels: models.filter(m => hasDeployment(m) && !isOnline(m)),
      undeployedModels: models.filter(m => !hasDeployment(m)),
    }
  }, [models, serveStatus])

  return (
    <div className="max-w-5xl mx-auto">
      {/* Toast */}
      {toast && (
        <div style={{
          position: 'fixed', top: 24, right: 24, zIndex: 9999,
          background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
          borderRadius: 10, padding: '10px 18px', fontSize: 13, color: 'var(--text-primary)',
          boxShadow: '0 4px 16px #0004', display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <CheckCircle2 size={14} color="#10b981" /> {toast}
        </div>
      )}

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>API as a Service</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
            Call your trained models via REST API — classify images, run detection, inference from any app
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <a
            href="/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: 6, textDecoration: 'none' }}
          >
            <BookOpen size={14} /> API Reference
          </a>
          <button className="btn btn-secondary btn-sm" onClick={fetchModels}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* ── API Keys section ──────────────────────────────────────────── */}
      <section style={{ marginBottom: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Key size={15} color="var(--primary)" />
            <h2 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>API Keys</h2>
            <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 20, background: 'var(--bg-elevated)', color: 'var(--text-muted)' }}>
              {apiKeys.length}
            </span>
          </div>
          {!creating && (
            <button
              className="btn btn-secondary btn-sm"
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={() => setCreating(true)}
            >
              <Plus size={13} /> New Key
            </button>
          )}
        </div>

        {/* Create key form */}
        {creating && (
          <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderRadius: 10, padding: 16, marginBottom: 12, display: 'flex', gap: 10, alignItems: 'center' }}>
            <input
              value={newKeyName}
              onChange={e => setNewKeyName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && createKey()}
              placeholder="Key name, e.g. My App, Production"
              autoFocus
              style={{ flex: 1, padding: '7px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-base)', color: 'var(--text-primary)' }}
            />
            <button className="btn btn-primary btn-sm" onClick={createKey} disabled={!newKeyName.trim()}>Create</button>
            <button className="btn btn-secondary btn-sm" onClick={() => { setCreating(false); setNewKeyName('') }}>Cancel</button>
          </div>
        )}

        {/* Key list */}
        {apiKeys.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)', fontSize: 13, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: 10 }}>
            <Lock size={28} style={{ margin: '0 auto 10px', opacity: 0.3, display: 'block' }} />
            No API keys yet — create one to start calling your models
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {apiKeys.map(k => {
              const revealed = revealId === k.id
              const masked = k.key.slice(0, 8) + '••••••••••••••••••••••••••••••••••••••••' + k.key.slice(-4)
              return (
                <div key={k.id} style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderRadius: 10, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--bg-elevated)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <Key size={14} color="var(--primary)" />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{k.name}</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                      <code style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: revealed ? '0.02em' : undefined }}>
                        {revealed ? k.key : masked}
                      </code>
                      <button
                        title={revealed ? 'Hide key' : 'Reveal key'}
                        onClick={() => setRevealId(revealed ? null : k.id)}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 2, lineHeight: 1 }}
                      >
                        {revealed ? <EyeOff size={12} /> : <Eye size={12} />}
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
                      Created {new Date(k.created).toLocaleDateString('th-TH')}
                      {k.lastUsed && ` · Last used ${new Date(k.lastUsed).toLocaleDateString('th-TH')}`}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      title={copiedId === k.id ? 'Copied!' : 'Copy key'}
                      className="btn btn-secondary btn-sm"
                      style={{ padding: '5px 10px' }}
                      onClick={() => copyToClipboard(k.key, k.id)}
                    >
                      {copiedId === k.id ? <CheckCircle2 size={13} color="#10b981" /> : <Copy size={13} />}
                    </button>
                    <button
                      title="Revoke key"
                      className="btn btn-secondary btn-sm"
                      style={{ padding: '5px 10px' }}
                      onClick={() => deleteKey(k.id)}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* ── Endpoints section ─────────────────────────────────────────── */}
      <section>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
          <Globe size={15} color="var(--primary)" />
          <h2 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Model Endpoints</h2>
          <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 20, background: 'var(--bg-elevated)', color: 'var(--text-muted)' }}>
            {models.length}
          </span>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)', fontSize: 13 }}>
            <RefreshCw size={22} style={{ margin: '0 auto 10px', display: 'block', opacity: 0.4, animation: 'spin 1s linear infinite' }} />
            Loading models…
          </div>
        ) : models.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)', fontSize: 13, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: 10 }}>
            <Zap size={28} style={{ margin: '0 auto 10px', opacity: 0.3, display: 'block' }} />
            No trained models yet — complete a training job to expose an endpoint
          </div>
        ) : (
          <div>
            {onlineModels.length > 0 && (
              <div style={{ marginBottom: offlineModels.length + undeployedModels.length > 0 ? 24 : 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                  <Wifi size={13} color="#10b981" />
                  <span style={{ fontSize: 12, fontWeight: 600, color: '#10b981' }}>Online — Ready to use</span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>({onlineModels.length})</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
                  {onlineModels.map(m => <ModelCard key={m.id} m={m} dimmed={false} serveStatus={serveStatus} onSelect={setSelectedModel} />)}
                </div>
              </div>
            )}
            {offlineModels.length > 0 && (
              <div style={{ marginBottom: undeployedModels.length > 0 ? 24 : 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                  <WifiOff size={13} color="#ef4444" />
                  <span style={{ fontSize: 12, fontWeight: 600, color: '#ef4444' }}>Offline</span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>({offlineModels.length}) — endpoint unavailable until redeployed</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
                  {offlineModels.map(m => <ModelCard key={m.id} m={m} dimmed={false} serveStatus={serveStatus} onSelect={setSelectedModel} />)}
                </div>
              </div>
            )}
            {undeployedModels.length > 0 && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                  <Zap size={13} color="var(--text-muted)" />
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>Not deployed</span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>({undeployedModels.length}) — deploy first from the Models page</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
                  {undeployedModels.map(m => <ModelCard key={m.id} m={m} dimmed={true} serveStatus={serveStatus} onSelect={setSelectedModel} />)}
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Model endpoint modal */}
      {selectedModel && (
        <EndpointModal
          model={selectedModel}
          baseUrl={baseUrl}
          copiedId={copiedId}
          onCopy={copyToClipboard}
          onClose={() => setSelectedModel(null)}
        />
      )}

      {/* ── Quick API Reference ───────────────────────────────────────── */}
      <section style={{ marginTop: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BookOpen size={15} color="var(--primary)" />
            <h2 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Quick Reference</h2>
          </div>
          <a
            href="/docs"
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontSize: 12, color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: 4, textDecoration: 'none' }}
          >
            Full Swagger UI <ExternalLink size={11} />
          </a>
        </div>

        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderRadius: 12, overflow: 'hidden' }}>
          {API_REFERENCE.map((group, gi) => (
            <div key={group.group}>
              {/* Group header */}
              <div style={{
                padding: '8px 16px',
                background: 'var(--bg-elevated)',
                borderBottom: '1px solid var(--border-subtle)',
                borderTop: gi > 0 ? '1px solid var(--border-default)' : undefined,
                fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em',
              }}>
                {group.group}
              </div>
              {group.endpoints.map((ep, i) => (
                <div
                  key={ep.path + ep.method}
                  style={{
                    display: 'flex', alignItems: 'flex-start', gap: 14, padding: '11px 16px',
                    borderBottom: i < group.endpoints.length - 1 ? '1px solid var(--border-subtle)' : undefined,
                    flexWrap: 'wrap',
                  }}
                >
                  <span style={{
                    fontSize: 11, fontWeight: 700, borderRadius: 4, padding: '2px 7px', flexShrink: 0,
                    minWidth: 46, textAlign: 'center',
                    ...METHOD_STYLE[ep.method],
                  }}>
                    {ep.method}
                  </span>
                  <code style={{ fontSize: 12, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', flex: 1, minWidth: 180 }}>
                    {ep.path}
                  </code>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)', flex: 2, minWidth: 160 }}>{ep.desc}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

// ── Endpoint Modal ────────────────────────────────────────────────────────

function EndpointModal({ model: m, baseUrl, copiedId, onCopy, onClose }: {
  model: DeployedModel
  baseUrl: string
  copiedId: string | null
  onCopy: (text: string, id: string) => void
  onClose: () => void
}) {
  const color = TT_COLORS[m.training_type] ?? '#6366f1'
  const endpoint = `${baseUrl}/api/inference`

  const curlExample = `curl -X POST "${endpoint}" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -F "model_id=${m.id}" \\
  -F "image=@image.jpg"`

  const pythonExample = `import requests

resp = requests.post(
    "${endpoint}",
    headers={"X-API-Key": "YOUR_API_KEY"},
    data={"model_id": "${m.id}"},
    files={"image": open("image.jpg", "rb")},
)
print(resp.json())`

  const jsExample = `const form = new FormData()
form.append('model_id', '${m.id}')
form.append('image', imageFile)

const res = await fetch('${endpoint}', {
  method: 'POST',
  headers: { 'X-API-Key': 'YOUR_API_KEY' },
  body: form,
})
const data = await res.json()
console.log(data)`

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        style={{ maxWidth: 680, width: '95vw', padding: 28, maxHeight: '90vh', overflowY: 'auto' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 42, height: 42, borderRadius: 11, background: color + '18', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <Zap size={20} color={color} />
            </div>
            <div>
              <h3 style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-primary)', margin: 0 }}>{m.name}</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <span style={{ fontSize: 11, padding: '1px 7px', borderRadius: 4, background: color + '22', color, fontWeight: 600 }}>
                  {TT_LABELS[m.training_type] ?? m.training_type}
                </span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{m.model}</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>·</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{fmtDate(m.created_at)}</span>
              </div>
            </div>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={onClose}>✕</button>
        </div>

        {/* Endpoint URL */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20, padding: '10px 14px', background: 'var(--bg-elevated)', borderRadius: 9 }}>
          <span style={{ fontSize: 11, fontWeight: 700, borderRadius: 4, padding: '2px 7px', background: '#10b98120', color: '#10b981', flexShrink: 0 }}>POST</span>
          <code style={{ fontSize: 12, flex: 1, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {endpoint}
          </code>
          <button
            title="Copy endpoint"
            onClick={() => onCopy(endpoint, 'ep_url_' + m.id)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: copiedId === 'ep_url_' + m.id ? '#10b981' : 'var(--text-muted)', lineHeight: 1, padding: 4, flexShrink: 0 }}
          >
            {copiedId === 'ep_url_' + m.id ? <CheckCircle2 size={13} /> : <Copy size={13} />}
          </button>
          <a href="/docs" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, textDecoration: 'none', flexShrink: 0 }}>
            Docs <ExternalLink size={10} />
          </a>
        </div>

        {/* Model ID row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20, padding: '8px 14px', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: 8 }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, flexShrink: 0 }}>model_id</span>
          <code style={{ fontSize: 12, flex: 1, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{m.id}</code>
          <button
            onClick={() => onCopy(m.id, 'mid_' + m.id)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: copiedId === 'mid_' + m.id ? '#10b981' : 'var(--text-muted)', lineHeight: 1, padding: 4, flexShrink: 0 }}
          >
            {copiedId === 'mid_' + m.id ? <CheckCircle2 size={13} /> : <Copy size={13} />}
          </button>
        </div>

        {/* Code examples */}
        <CodeTabs model={m} curlExample={curlExample} pythonExample={pythonExample} jsExample={jsExample} />
      </div>
    </div>
  )
}

// ── Code Tabs sub-component ────────────────────────────────────────────────

function CodeTabs({ curlExample, pythonExample, jsExample }: {
  model: DeployedModel
  curlExample: string
  pythonExample: string
  jsExample: string
}) {
  const [tab, setTab] = useState<'curl' | 'python' | 'js'>('curl')
  const [copied, setCopied] = useState(false)

  const code = tab === 'curl' ? curlExample : tab === 'python' ? pythonExample : jsExample

  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const tabs: { id: 'curl' | 'python' | 'js'; label: string }[] = [
    { id: 'curl',   label: 'cURL' },
    { id: 'python', label: 'Python' },
    { id: 'js',     label: 'JavaScript' },
  ]

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ display: 'flex', gap: 4, background: 'var(--bg-elevated)', borderRadius: 8, padding: 4 }}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding: '4px 12px', borderRadius: 6, fontSize: 12, fontWeight: 500, border: 'none', cursor: 'pointer',
                background: tab === t.id ? 'var(--bg-surface)' : 'transparent',
                color: tab === t.id ? 'var(--text-primary)' : 'var(--text-muted)',
                boxShadow: tab === t.id ? '0 1px 4px #0002' : 'none',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
        <button
          onClick={copy}
          style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, padding: '4px 10px', borderRadius: 6, border: '1px solid var(--border-subtle)', background: 'var(--bg-elevated)', color: copied ? '#10b981' : 'var(--text-muted)', cursor: 'pointer' }}
        >
          {copied ? <CheckCircle2 size={11} /> : <Copy size={11} />}
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <div style={{ position: 'relative', background: 'var(--bg-base)', border: '1px solid var(--border-subtle)', borderRadius: 8, overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-elevated)' }}>
          <Code2 size={12} color="var(--text-muted)" />
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {tab === 'curl' ? 'Terminal' : tab === 'python' ? 'Python 3' : 'JavaScript / TypeScript'}
          </span>
        </div>
        <pre style={{ margin: 0, padding: '14px 16px', fontSize: 12, lineHeight: 1.7, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', overflowX: 'auto', whiteSpace: 'pre' }}>
          {code}
        </pre>
      </div>
    </div>
  )
}
