import { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import {
  Database, RefreshCw, ExternalLink, CheckCircle2, XCircle,
  AlertCircle, Loader2, HardDrive, FolderSync, Upload, FileText, Trash2,
  Search, Package, Wand2, X, File, FileImage, FileSpreadsheet,
  ChevronRight, ChevronDown, Eye, Folder, BookText,
} from 'lucide-react'

interface HFDataset {
  id: string
  author?: string
  downloads?: number
  tags?: string[]
  gated?: boolean | string
}

interface HFDatasetDetail {
  id: string
  author?: string
  description?: string
  tags?: string[]
  downloads?: number
  likes?: number
  gated?: boolean | string
  private?: boolean
  lastModified?: string
  createdAt?: string
  cardData?: any
  license?: string
}

interface HFFile {
  path: string
  size: number
  type: 'blob' | 'tree'
}

function _hfStripHtml(html: string): string {
  // HF descriptions come as HTML with whitespace, <br> tags, code blocks,
  // and inline formatting. Strip tags, normalise line breaks, collapse
  // repeated blank lines, then return the plain text for display.
  if (!html) return ''
  let s = html
  s = s.replace(/<\s*br\s*\/?>/gi, '\n')
  s = s.replace(/<\/(p|div|h[1-6]|li)>/gi, '\n')
  s = s.replace(/<li[^>]*>/gi, '• ')
  s = s.replace(/<[^>]+>/g, '')
  s = s.replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'")
  s = s.replace(/[ \t]+/g, ' ').replace(/\n[ \t]+/g, '\n').replace(/[ \t]+\n/g, '\n')
  s = s.split('\n').map(l => l.trim()).filter(Boolean).join('\n')
  return s.trim()
}

function _hfFilePrio(p: string): number {
  const x = p.toLowerCase()
  if (x.endsWith('.parquet')) return 0
  if (x.endsWith('.jsonl') || x.endsWith('.json') || x.endsWith('.csv')) return 1
  if (x.endsWith('.jpg') || x.endsWith('.jpeg') || x.endsWith('.png') || x.endsWith('.webp')) return 2
  if (x.endsWith('.txt') || x.endsWith('.md')) return 3
  if (x.endsWith('.zip') || x.endsWith('.tar') || x.endsWith('.gz')) return 4
  return 9
}

function _hfFileIcon(p: string) {
  const x = p.toLowerCase()
  if (x.endsWith('.parquet') || x.endsWith('.csv') || x.endsWith('.jsonl') || x.endsWith('.json'))
    return FileSpreadsheet
  if (x.endsWith('.jpg') || x.endsWith('.jpeg') || x.endsWith('.png') || x.endsWith('.webp'))
    return FileImage
  // .md gets a dedicated book-style icon (markdown is documentation)
  if (x.endsWith('.md')) return BookText
  // .zip/.tar/.gz use the same MD-style icon per the design change —
  // distinguishes from a generic FileText, and matches the visual
  // language used for documentation in the file picker.
  if (x.endsWith('.zip') || x.endsWith('.tar') || x.endsWith('.gz')) return BookText
  if (x.endsWith('.txt')) return FileText
  return File
}

function _hfFmtSize(b: number): string {
  if (!b) return '—'
  if (b >= 1024 * 1024 * 1024) return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`
  if (b >= 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`
  if (b >= 1024) return `${(b / 1024).toFixed(0)} KB`
  return `${b} B`
}

const LS_API   = '/api/ls'
const LS_TOKEN = 'medimage-ls-token-2026'

interface LSProject {
  id: number
  title: string
  task_number: number
  finished_task_number: number
}

interface LSStorage {
  id: number
  title: string
  bucket: string
  status: string
  last_sync: string | null
  last_sync_count: number | null
}

interface Dataset {
  projectId: number
  projectTitle: string
  storages: LSStorage[]
  taskCount: number
  labeledCount: number
  lastSync: string | null
  syncStatus: string
}

interface TextDataset {
  id: string
  name: string
  format: string
  row_count: number
  size_bytes: number
  created_at: number
  preview?: any[]
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    completed: { cls: 'badge badge-success', icon: <CheckCircle2 size={11} />, label: 'Synced'  },
    failed:    { cls: 'badge badge-danger',  icon: <XCircle size={11} />,      label: 'Failed'  },
    started:   { cls: 'badge badge-warning', icon: <Loader2 size={11} className="animate-spin" />, label: 'Syncing' },
    unknown:   { cls: 'badge badge-neutral', icon: <AlertCircle size={11} />,  label: 'No sync' },
  }
  const c = map[status] ?? map.unknown
  return <span className={c.cls}>{c.icon}{c.label}</span>
}

// Module-level cache: persists data across navigations so no flicker on re-visit
let _datasetsCache: Dataset[] = []

export default function Datasets() {
  const [loading,  setLoading]  = useState(_datasetsCache.length === 0)
  const [datasets, setDatasets] = useState<Dataset[]>(_datasetsCache)
  const [syncing,  setSyncing]  = useState<Set<number>>(new Set())
  const [toast,    setToast]    = useState<{ msg: string; type: string } | null>(null)
  const [tab, setTab] = useState<'image' | 'text'>('image')
  const [textDatasets, setTextDatasets] = useState<TextDataset[]>([])
  const [textLoading, setTextLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [showHFImport, setShowHFImport] = useState(false)
  const [autoLabelProject, setAutoLabelProject] = useState<Dataset | null>(null)
  const [autoLabelEngine, setAutoLabelEngine] = useState('Ultralytics')
  const [autoLabelModel, setAutoLabelModel] = useState('yolov8s.pt')
  const [autoLabeling, setAutoLabeling] = useState(false)

  const showToast = (msg: string, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

  const fetchDatasets = useCallback(async () => {
    if (_datasetsCache.length === 0) setLoading(true)
    try {
      const projRes = await axios.get(`${LS_API}/projects/`, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
        params: { page_size: 100 },
      })
      const projects: LSProject[] = projRes.data?.results ?? (Array.isArray(projRes.data) ? projRes.data : [])

      const results = await Promise.all(projects.map(async (p): Promise<Dataset> => {
        try {
          const storRes = await axios.get(`${LS_API}/storages/s3`, {
            headers: { Authorization: `Token ${LS_TOKEN}` },
            params: { project: p.id },
          })
          const storages: LSStorage[] = Array.isArray(storRes.data) ? storRes.data : []
          const sorted   = [...storages].sort((a, b) => (b.last_sync ?? '').localeCompare(a.last_sync ?? ''))
          const statuses = storages.map(s => s.status)
          return {
            projectId:    p.id,
            projectTitle: p.title,
            storages,
            taskCount:    p.task_number           ?? 0,
            labeledCount: p.finished_task_number  ?? 0,
            lastSync:     sorted[0]?.last_sync    ?? null,
            syncStatus:   statuses.includes('started')  ? 'started'
                        : statuses.includes('failed')   ? 'failed'
                        : statuses.every(s => s === 'completed') && statuses.length > 0 ? 'completed'
                        : 'unknown',
          }
        } catch {
          return {
            projectId: p.id, projectTitle: p.title, storages: [],
            taskCount: p.task_number ?? 0, labeledCount: p.finished_task_number ?? 0,
            lastSync: null, syncStatus: 'unknown',
          }
        }
      }))

      _datasetsCache = results
      setDatasets(results)
    } catch (e: any) {
      showToast(`Error: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchDatasets() }, [fetchDatasets])

  const fetchTextDatasets = useCallback(async () => {
    setTextLoading(true)
    try {
      const res = await fetch('/api/text-datasets')
      const data = await res.json()
      setTextDatasets(data.datasets ?? [])
    } catch { /* ignore */ }
    setTextLoading(false)
  }, [])

  useEffect(() => {
    if (tab === 'text') fetchTextDatasets()
  }, [tab, fetchTextDatasets])

  const handleFileUpload = async (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (!['jsonl', 'json', 'txt', 'csv'].includes(ext ?? '')) {
      showToast('รองรับเฉพาะ .jsonl, .json, .txt, .csv', 'error')
      return
    }
    setUploading(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch('/api/text-datasets/upload', { method: 'POST', body: fd })
      if (!res.ok) throw new Error(await res.text())
      showToast(`✅ อัปโหลด "${file.name}" สำเร็จ`)
      fetchTextDatasets()
    } catch (e: any) {
      showToast(`❌ อัปโหลดล้มเหลว: ${e.message}`, 'error')
    }
    setUploading(false)
  }

  const deleteTextDataset = async (id: string, name: string) => {
    if (!confirm(`ลบ dataset "${name}"?`)) return
    try {
      await fetch(`/api/text-datasets/${id}`, { method: 'DELETE' })
      showToast(`ลบ "${name}" แล้ว`)
      fetchTextDatasets()
    } catch {
      showToast('ลบไม่สำเร็จ', 'error')
    }
  }

  const fmtBytes = (b: number) =>
    b < 1024 ? `${b} B` : b < 1024 * 1024 ? `${(b/1024).toFixed(1)} KB` : `${(b/1024/1024).toFixed(1)} MB`

  const openInLS = useCallback((projectId: number) => {
    // Backend handles LS login server-side, sets sessionid cookie, then redirects to LS
    window.open(`/api/ls-goto/${projectId}`, `ls-${projectId}-${Date.now()}`)
  }, [])

  const syncAll = async (ds: Dataset) => {
    if (ds.storages.length === 0) { showToast('No storage to sync', 'info'); return }
    setSyncing(prev => new Set([...prev, ds.projectId]))
    try {
      await Promise.all(ds.storages.map(s =>
        axios.post(`${LS_API}/storages/s3/${s.id}/sync`, {},
          { headers: { Authorization: `Token ${LS_TOKEN}` } }
        ).catch(() => {})
      ))
      showToast(`Syncing ${ds.storages.length} storage(s) for "${ds.projectTitle}"`)
      setTimeout(fetchDatasets, 3000)
    } finally {
      setSyncing(prev => { const s = new Set(prev); s.delete(ds.projectId); return s })
    }
  }

  const fmt = (d: string | null) =>
    d ? new Date(d).toLocaleString('th-TH', {
      day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
    }) : '—'

  const AUTO_LABEL_PRESETS = [
    { engine: 'Ultralytics', model: 'yolov8s.pt',                     label: 'YOLOv8-Small (General)'         },
    { engine: 'Ultralytics', model: 'yolov8n.pt',                     label: 'YOLOv8-Nano (Fast/Edge)'        },
    { engine: 'HuggingFace', model: 'facebook/detr-resnet-50',        label: 'DETR ResNet-50'                 },
    { engine: 'MedSAM',      model: 'medsam_vit_b',                   label: 'MedSAM (Medical Segmentation)'  },
    { engine: 'HuggingFace', model: 'microsoft/rad-dino',             label: 'RAD-DINO (Radiology)'           },
  ]

  const runAutoLabel = async () => {
    if (!autoLabelProject) return
    setAutoLabeling(true)
    try {
      const res = await axios.post(`/api/autolabel/${autoLabelProject.projectId}`, {
        project_id: autoLabelProject.projectId,
        model: autoLabelModel,
        engine: autoLabelEngine,
        task_ids: [],
      }, { withCredentials: true })
      showToast(`Auto-label เสร็จ: ${res.data.labeled}/${res.data.total} tasks`, 'success')
      setAutoLabelProject(null)
      fetchDatasets()
    } catch (e: any) {
      showToast(e?.response?.data?.detail || 'Auto-label failed', 'error')
    } finally {
      setAutoLabeling(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>Datasets</h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
            {tab === 'image' ? 'Label Studio annotation projects — datasets backed by MinIO S3' : 'Text & Instruction datasets for LLM/VLM fine-tuning'}
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button className="btn btn-secondary btn-sm" onClick={tab === 'image' ? fetchDatasets : fetchTextDatasets} disabled={loading || textLoading}>
            <RefreshCw size={14} />Refresh
          </button>
          <button
            className="btn btn-secondary btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={() => setShowHFImport(true)}
          >
            <Package size={14} /> Import from HuggingFace
          </button>
        </div>
      </div>

      {/* Tab switcher */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, padding: '4px', background: 'var(--bg-elevated)', borderRadius: 10, width: 'fit-content' }}>
        <button
          onClick={() => setTab('image')}
          style={{
            padding: '6px 16px', borderRadius: 8, fontSize: 13, fontWeight: 500, border: 'none', cursor: 'pointer',
            background: tab === 'image' ? 'var(--bg-surface)' : 'transparent',
            color: tab === 'image' ? 'var(--text-primary)' : 'var(--text-muted)',
            boxShadow: tab === 'image' ? '0 1px 3px rgba(0,0,0,0.15)' : 'none',
          }}
        >
          <Database size={13} style={{ display: 'inline', marginRight: 6 }} />
          Image Datasets
        </button>
        <button
          onClick={() => setTab('text')}
          style={{
            padding: '6px 16px', borderRadius: 8, fontSize: 13, fontWeight: 500, border: 'none', cursor: 'pointer',
            background: tab === 'text' ? 'var(--bg-surface)' : 'transparent',
            color: tab === 'text' ? 'var(--text-primary)' : 'var(--text-muted)',
            boxShadow: tab === 'text' ? '0 1px 3px rgba(0,0,0,0.15)' : 'none',
          }}
        >
          <FileText size={13} style={{ display: 'inline', marginRight: 6 }} />
          Text Datasets
        </button>
      </div>

      {toast && (
        <div className={`toast ${toast.type === 'error' ? 'toast-error' : 'toast-success'} mb-4`}>
          {toast.msg}
        </div>
      )}

      {/* ── Text Datasets Tab ── */}
      {tab === 'text' && (
        <div>
          {/* Upload zone */}
          <div
            className="card mb-6"
            style={{
              borderStyle: 'dashed', borderWidth: 2, cursor: 'pointer',
              borderColor: dragOver ? '#8b5cf6' : 'var(--border)',
              background: dragOver ? 'rgba(139,92,246,0.06)' : 'var(--bg-surface)',
              textAlign: 'center', padding: '32px 16px', transition: 'all 0.15s',
            }}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => {
              e.preventDefault()
              setDragOver(false)
              const file = e.dataTransfer.files[0]
              if (file) handleFileUpload(file)
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jsonl,.json,.txt,.csv"
              style={{ display: 'none' }}
              onChange={e => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            />
            {uploading ? (
              <><Loader2 size={28} className="animate-spin" style={{ color: '#8b5cf6', margin: '0 auto 8px' }} />
              <p style={{ fontSize: 14, color: 'var(--text-muted)' }}>กำลังอัปโหลด...</p></>
            ) : (
              <><Upload size={28} style={{ color: '#8b5cf6', margin: '0 auto 8px' }} />
              <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-primary)', marginBottom: 4 }}>คลิกหรือลาก .jsonl file มาวางที่นี่</p>
              <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>รองรับ .jsonl (Alpaca / ShareGPT), .json, .txt, .csv</p></>
            )}
          </div>

          {/* Format hints */}
          <div className="card mb-6" style={{ background: 'var(--bg-elevated)' }}>
            <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>รูปแบบ .jsonl ที่รองรับ</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Alpaca format</p>
                    <pre style={{ fontSize: 11, background: 'var(--bg-surface)', borderRadius: 6, padding: '8px', color: 'var(--text-secondary)', overflow: 'auto' }}>{`{"instruction": "Describe the defect",\n "input": "",\n "output": "Surface scratch on panel A"}`}</pre>
              </div>
              <div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>ShareGPT / ChatML format</p>
                <pre style={{ fontSize: 11, background: 'var(--bg-surface)', borderRadius: 6, padding: '8px', color: 'var(--text-secondary)', overflow: 'auto' }}>{`{"conversations": [\n  {"role": "user", "content": "..."},\n  {"role": "assistant", "content": "..."}\n]}`}</pre>
              </div>
            </div>
          </div>

          {/* Dataset list */}
          {textLoading ? (
            <div className="flex items-center justify-center h-32"><div className="loading-spinner" /></div>
          ) : textDatasets.length === 0 ? (
            <div className="card" style={{ textAlign: 'center', padding: '40px 16px' }}>
              <FileText size={36} style={{ color: 'var(--text-muted)', margin: '0 auto 12px' }} />
              <p style={{ fontSize: 14, color: 'var(--text-muted)' }}>ยังไม่มี text dataset</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {textDatasets.map(ds => (
                <div key={ds.id} className="card">
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 32, height: 32, borderRadius: 8, background: 'rgba(139,92,246,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        <FileText size={15} style={{ color: '#8b5cf6' }} />
                      </div>
                      <div>
                        <h3 style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)', margin: 0 }}>{ds.name}</h3>
                        <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                          <span style={{ fontSize: 11, background: 'rgba(139,92,246,0.15)', color: '#8b5cf6', borderRadius: 4, padding: '1px 6px', fontWeight: 600 }}>{ds.format}</span>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{ds.row_count.toLocaleString()} rows</span>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>·</span>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fmtBytes(ds.size_bytes)}</span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => deleteTextDataset(ds.id, ds.name)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 4 }}
                      title="ลบ dataset นี้"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  {/* Preview rows */}
                  {ds.preview && ds.preview.length > 0 && (
                    <div style={{ marginTop: 8, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Preview (3 rows)</p>
                      {ds.preview.slice(0, 3).map((row, i) => (
                        <pre key={i} style={{ fontSize: 10, background: 'var(--bg-elevated)', borderRadius: 4, padding: '4px 6px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '100%', marginBottom: 2 }}>
                          {JSON.stringify(row).slice(0, 100)}{JSON.stringify(row).length > 100 ? '...' : ''}
                        </pre>
                      ))}
                    </div>
                  )}
                  <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                    <a href="/train" className="btn btn-primary btn-sm" style={{ fontSize: 11 }}>Use in Training →</a>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>{fmt(new Date(ds.created_at * 1000).toISOString())}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Image Datasets Tab ── */}
      {tab === 'image' && (
        <>
        {loading && datasets.length === 0 ? (
          <div className="flex items-center justify-center h-64"><div className="loading-spinner" /></div>
        ) : datasets.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '64px 0', color: 'var(--text-muted)' }}>
            <Database size={40} style={{ margin: '0 auto 16px', opacity: 0.4, display: 'block' }} />
            <p style={{ fontSize: 14 }}>No datasets yet — create a project to get started</p>
            <a href="/projects" style={{ color: 'var(--primary-hover)', fontSize: 13, marginTop: 8, display: 'inline-block' }}>
              Go to Projects →
            </a>
          </div>
        ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {datasets.map((ds) => {
          const isSyncing = syncing.has(ds.projectId)
          return (
            <div key={ds.projectId} className="card" style={{ display: 'flex', flexDirection: 'column' }}>
              {/* Header */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: 1, marginRight: 10 }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <Database size={15} style={{ color: 'var(--primary)' }} />
                  </div>
                  <h3 style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', margin: 0 }}>
                    {ds.projectTitle}
                  </h3>
                </div>
                <StatusBadge status={ds.syncStatus} />
              </div>

              {/* Bucket chips — full width 2 column grid */}
              {ds.storages.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 10 }}>
                  {[...new Set(ds.storages.map(s => s.bucket))].map(b => (
                    <span key={b} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--primary-hover)', background: 'var(--primary-dim)', border: '1px solid var(--primary-border)', borderRadius: 4, padding: '3px 8px', overflow: 'hidden', minWidth: 0 }}>
                      <HardDrive size={9} style={{ flexShrink: 0 }} />
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{b}</span>
                    </span>
                  ))}
                </div>
              )}

              {/* Stats */}
              <div style={{ display: 'flex', gap: 20, paddingTop: 12, paddingBottom: 12, borderTop: '1px solid var(--border-subtle)', borderBottom: '1px solid var(--border-subtle)', marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>{ds.taskCount.toLocaleString()}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>Images</div>
                </div>
                <div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--success)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>{ds.labeledCount.toLocaleString()}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>Labeled</div>
                </div>
                {ds.taskCount > 0 && (
                  <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>
                      {Math.round((ds.labeledCount / ds.taskCount) * 100)}%
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>Complete</div>
                  </div>
                )}
              </div>

              {/* Last sync */}
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, flex: 1 }}>
                Last sync: {fmt(ds.lastSync)}
              </div>

              {/* Actions */}
              <div style={{ display: 'flex', gap: 8 }}>
                {ds.storages.length > 0 && (
                  <button
                    className="btn btn-primary btn-sm"
                    style={{ flex: 1 }}
                    onClick={() => syncAll(ds)}
                    disabled={isSyncing || ds.syncStatus === 'started'}
                  >
                    {isSyncing
                      ? <><Loader2 size={12} className="animate-spin" />Syncing…</>
                      : <><FolderSync size={12} />Sync</>
                    }
                  </button>
                )}
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ flex: 1 }}
                  onClick={() => openInLS(ds.projectId)}
                >
                  <ExternalLink size={12} />Open in LS
                </button>
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ flex: 1 }}
                  title="Auto-generate pre-annotations with a pre-trained model"
                  onClick={() => { setAutoLabelProject(ds); setAutoLabelEngine('Ultralytics'); setAutoLabelModel('yolov8s.pt') }}
                >
                  <Wand2 size={12} />Auto-label
                </button>
              </div>
            </div>
          )
        })}
      </div>
      )}
      </>
      )}

      {/* Auto-label modal */}
      {autoLabelProject && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="card" style={{ width: 440, padding: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Wand2 size={18} style={{ color: 'var(--primary)' }} />
                <h3 style={{ fontWeight: 600 }}>Auto-label</h3>
              </div>
              <button className="btn btn-secondary btn-sm" onClick={() => setAutoLabelProject(null)}><X size={14} /></button>
            </div>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
              Generate pre-annotations สำหรับ <strong>{autoLabelProject.projectTitle}</strong> ด้วย pre-trained model
            </p>
            <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>เลือก Model</label>
            <select
              className="input"
              style={{ marginBottom: 16 }}
              value={`${autoLabelEngine}|${autoLabelModel}`}
              onChange={e => {
                const [eng, mdl] = e.target.value.split('|')
                setAutoLabelEngine(eng)
                setAutoLabelModel(mdl)
              }}
            >
              {AUTO_LABEL_PRESETS.map(p => (
                <option key={`${p.engine}|${p.model}`} value={`${p.engine}|${p.model}`}>{p.label}</option>
              ))}
            </select>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16, padding: '8px 12px', background: 'var(--bg-elevated)', borderRadius: 6 }}>
              Engine: <strong>{autoLabelEngine}</strong> · Model: <code style={{ fontSize: 11 }}>{autoLabelModel}</code>
              <br />
              จะ generate pre-annotations สำหรับทุก task ที่ยังไม่มี annotation
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setAutoLabelProject(null)} disabled={autoLabeling}>Cancel</button>
              <button className="btn btn-primary" style={{ flex: 1 }} onClick={runAutoLabel} disabled={autoLabeling}>
                {autoLabeling ? <><Loader2 size={14} className="animate-spin" />Running…</> : <><Wand2 size={14} />Start Auto-label</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {showHFImport && (
        <HFDatasetImportModal
          onClose={() => setShowHFImport(false)}
          onDone={() => { setShowHFImport(false); fetchDatasets() }}
        />
      )}
    </div>
  )
}

// ─── HuggingFace Dataset Import Modal ────────────────────────────────────────

function HFDatasetImportModal({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [hfQuery, setHfQuery] = useState('')
  const [hfResults, setHfResults] = useState<HFDataset[]>([])
  const [hfSearching, setHfSearching] = useState(false)
  const [hfSearched, setHfSearched] = useState(false)
  const [selected, setSelected] = useState<HFDataset | null>(null)
  // ── File-selection step: fetched after a dataset is picked ───────────
  const [hfFiles, setHfFiles] = useState<HFFile[]>([])
  const [hfFilesLoading, setHfFilesLoading] = useState(false)
  const [hfFilesError, setHfFilesError] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [showFilePicker, setShowFilePicker] = useState(false)
  // ── Right-panel detail (description / tags / downloads) ───────────────
  const [hfDetail, setHfDetail] = useState<HFDatasetDetail | null>(null)
  const [hfDetailLoading, setHfDetailLoading] = useState(false)
  // ── Archive-inspect popover (eye icon on zip/tar files) ───────────────
  const [inspectFile, setInspectFile] = useState<string | null>(null)
  const [inspectData, setInspectData] = useState<any | null>(null)
  const [inspectLoading, setInspectLoading] = useState(false)
  const [inspectError, setInspectError] = useState('')
  const [bucketName, setBucketName] = useState('')
  const [bucketError, setBucketError] = useState('')
  const [maxFiles, setMaxFiles] = useState(20)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<any>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  const suggestBucket = (id: string) => {
    const base = id.split('/').pop() ?? id
    return 'hf-' + base.toLowerCase().replace(/[^a-z0-9]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '').slice(0, 27)
  }

  const validateBucket = (name: string): string => {
    if (!name) return 'Required'
    if (!/^[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]$/.test(name))
      return '3–63 chars, lowercase letters/numbers/hyphens only, no leading/trailing hyphens'
    return ''
  }

  const searchHF = async () => {
    if (!hfQuery.trim()) return
    setHfSearching(true)
    setHfResults([])
    try {
      const res = await fetch(
        `https://huggingface.co/api/datasets?search=${encodeURIComponent(hfQuery.trim())}&limit=8&sort=downloads&direction=-1`,
        { signal: AbortSignal.timeout(8000) }
      )
      const data = await res.json()
      setHfResults(Array.isArray(data) ? data : [])
    } catch { setHfResults([]) }
    setHfSearching(false)
    setHfSearched(true)
  }

  const selectDataset = (ds: HFDataset) => {
    setSelected(ds)
    setHfResults([])
    setHfSearched(false)
    setHfQuery('')
    setShowFilePicker(true)
    // Pre-fetch file tree so the picker renders immediately when the
    // user expands it. Gated repos require a token and HF will return
    // 401 — surface that as a clear error instead of failing silently.
    setHfFiles([])
    setSelectedFiles(new Set())
    setHfFilesError('')
    setHfFilesLoading(true)
    fetch(`https://huggingface.co/api/datasets/${ds.id.split('/').map(encodeURIComponent).join('/')}/tree/main?recursive=true`, { signal: AbortSignal.timeout(15000) })
      .then(async r => {
        if (!r.ok) {
          const body = await r.text().catch(() => '')
          throw new Error(`HTTP ${r.status} — ${body.slice(0, 120) || (r.status === 401 ? 'repo is gated or private (add a HF token in Profile)' : 'request failed')}`)
        }
        return r.json()
      })
      .then((entries: any[]) => {
        // HF tree API returns type='file' (or 'lfs' for LFS pointer files),
        // NOT 'blob' as the field name might suggest. The 'directory'
        // entries are folders — skip them.
        const blobs = (entries || [])
          .filter(e => (e?.type === 'file' || e?.type === 'lfs') && !String(e.path || '').toLowerCase().endsWith('.gitattributes'))
          .map(e => ({ path: String(e.path), size: Number(e.size || 0), type: 'blob' as const }))
        blobs.sort((a, b) => _hfFilePrio(a.path) - _hfFilePrio(b.path))
        setHfFiles(blobs)
        // Pre-check the first 20 (matches backend default). The user
        // can then check/uncheck before importing.
        setSelectedFiles(new Set(blobs.slice(0, 20).map(b => b.path)))
      })
      .catch((e: any) => setHfFilesError(e?.message || 'Could not load file tree'))
      .finally(() => setHfFilesLoading(false))

    // Pre-fetch dataset metadata (description, tags, downloads, etc.) for
    // the right-hand detail panel. Independent of the file tree fetch so a
    // tree-fetch error doesn't blank the description.
    setHfDetail(null)
    setHfDetailLoading(true)
    fetch(`https://huggingface.co/api/datasets/${ds.id.split('/').map(encodeURIComponent).join('/')}`, { signal: AbortSignal.timeout(15000) })
      .then(async r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d: any) => setHfDetail({
        id: d.id,
        author: d.author,
        description: d.description || '',
        tags: d.tags || [],
        downloads: d.downloads,
        likes: d.likes,
        gated: d.gated,
        private: d.private,
        lastModified: d.lastModified,
        createdAt: d.createdAt,
        cardData: d.cardData,
        license: d.cardData?.license || d.cardData?.licenses?.[0]?.name,
      }))
      .catch(() => setHfDetail(null))
      .finally(() => setHfDetailLoading(false))

    const suggested = suggestBucket(ds.id)
    setBucketName(suggested)
    setBucketError('')
  }

  const inspectArchive = async (filePath: string) => {
    if (!selected) return
    setInspectFile(filePath)
    setInspectData(null)
    setInspectError('')
    setInspectLoading(true)
    try {
      const url = `/api/datasets/inspect-hf-file?repo_id=${encodeURIComponent(selected.id)}&path=${encodeURIComponent(filePath)}`
      const r = await fetch(url, { signal: AbortSignal.timeout(180_000) })
      if (!r.ok) {
        const body = await r.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${r.status}`)
      }
      setInspectData(await r.json())
    } catch (e: any) {
      setInspectError(e?.message || 'Inspect failed')
    } finally {
      setInspectLoading(false)
    }
  }

  const closeInspect = () => {
    setInspectFile(null)
    setInspectData(null)
    setInspectError('')
    setInspectLoading(false)
  }

  const startImport = async () => {
    const err = validateBucket(bucketName)
    if (err) { setBucketError(err); return }
    if (!selected) return
    if (selectedFiles.size === 0) {
      setBucketError('Select at least one file to download')
      return
    }
    try {
      const res = await fetch('/api/datasets/import-hf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hf_dataset_id: selected.id,
          bucket_name: bucketName,
          max_files: maxFiles,
          // Pass the explicit file list. Backend honours it when
          // provided; falls back to "first N by priority" when not.
          selected_files: Array.from(selectedFiles),
        }),
      })
      const data = await res.json()
      if (!res.ok) { setBucketError(data.detail ?? 'Import failed'); return }
      setJobId(data.job_id)
      pollRef.current = setInterval(async () => {
        const r = await fetch(`/api/datasets/import-hf/${data.job_id}`)
        const d = await r.json()
        setJob(d)
        setTimeout(() => logRef.current?.scrollTo(0, logRef.current.scrollHeight), 30)
        if (d.status === 'completed' || d.status === 'error') {
          clearInterval(pollRef.current!)
        }
      }, 800)
    } catch (e: any) {
      setBucketError(e.message)
    }
  }

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const isRunning = job && (job.status === 'queued' || job.status === 'running')
  const isDone    = job?.status === 'completed'
  const isError   = job?.status === 'error'

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        style={{ maxWidth: selected ? 1240 : 600, width: '95vw', padding: 28, maxHeight: '92vh', display: 'flex', flexDirection: 'column' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 9, background: '#f9731620', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Package size={18} color="#f97316" />
            </div>
            <div>
              <h3 style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-primary)', margin: 0 }}>Import from HuggingFace</h3>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: 0 }}>Download dataset files → create MinIO bucket</p>
            </div>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={onClose} disabled={isRunning}>✕</button>
        </div>

        {/* ── Pre-import form ── */}
        {!jobId && (
          <div style={{ display: 'flex', gap: 22, flex: 1, minHeight: 0, overflow: 'hidden' }}>
            {/* Left column — form */}
            <div style={{ flex: selected ? '0 0 56%' : 1, minWidth: 0, minHeight: 0, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
            {/* Search */}
            <div style={{ marginBottom: 18 }}>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>
                Search HuggingFace Datasets
              </label>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  value={hfQuery}
                  onChange={e => setHfQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && searchHF()}
                  placeholder="e.g. imagenet, coco, squad, mnist..."
                  style={{ flex: 1, padding: '8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)' }}
                />
                <button
                  onClick={searchHF}
                  disabled={hfSearching || !hfQuery.trim()}
                  style={{ padding: '8px 14px', borderRadius: 7, fontSize: 12, fontWeight: 600, cursor: 'pointer', border: '1px solid var(--border-default)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6, opacity: hfSearching || !hfQuery.trim() ? 0.5 : 1 }}
                >
                  {hfSearching ? <Loader2 size={13} className="animate-spin" /> : <Search size={13} />}
                  Search
                </button>
              </div>

              {/* Results */}
              {hfResults.length > 0 && (
                <div style={{ marginTop: 8, border: '1px solid var(--border-default)', borderRadius: 8, overflow: 'hidden', maxHeight: 240, overflowY: 'auto' }}>
                  {hfResults.map((ds, i) => (
                    <button
                      key={ds.id}
                      onClick={() => selectDataset(ds)}
                      style={{
                        width: '100%', display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
                        border: 'none', borderBottom: i < hfResults.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                        background: i % 2 === 0 ? 'var(--bg-surface)' : 'var(--bg-base)',
                        cursor: 'pointer', textAlign: 'left',
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ds.id}</div>
                        <div style={{ display: 'flex', gap: 6, marginTop: 3, flexWrap: 'wrap' }}>
                          {ds.gated && <span style={{ fontSize: 10, background: '#f9731620', color: '#f97316', borderRadius: 4, padding: '1px 5px', fontWeight: 600 }}>Gated</span>}
                          {(ds.tags ?? []).slice(0, 3).map(t => (
                            <span key={t} style={{ fontSize: 10, background: 'var(--bg-elevated)', color: 'var(--text-muted)', borderRadius: 4, padding: '1px 5px' }}>{t}</span>
                          ))}
                        </div>
                      </div>
                      {ds.downloads !== undefined && (
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', flexShrink: 0, textAlign: 'right' }}>
                          ↓ {ds.downloads >= 1000 ? (ds.downloads / 1000).toFixed(0) + 'k' : ds.downloads}
                        </div>
                      )}
                      <span style={{ fontSize: 11, color: 'var(--primary)', fontWeight: 600, flexShrink: 0 }}>Select →</span>
                    </button>
                  ))}
                </div>
              )}
              {hfSearched && hfResults.length === 0 && !hfSearching && (
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>ไม่พบผลลัพธ์ ลองคำค้นอื่น</p>
              )}
            </div>

            {/* Selected dataset chip */}
            {selected && (
              <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderRadius: 8, background: '#f9731614', border: '1px solid #f9731640' }}>
                <Package size={14} color="#f97316" style={{ flexShrink: 0 }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>{selected.id}</span>
                {selected.gated && <span style={{ fontSize: 10, background: '#f9731620', color: '#f97316', borderRadius: 4, padding: '2px 6px', fontWeight: 700 }}>Gated ⚠</span>}
                <button onClick={() => { setSelected(null); setShowFilePicker(false); setHfFiles([]); setSelectedFiles(new Set()) }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 16, lineHeight: 1 }}>×</button>
              </div>
            )}

            {/* ── File-selection step (self-scrolling flex child) ── */}
            {selected && (
              <div style={{ marginBottom: 18, border: '1px solid var(--border-default)', borderRadius: 8, overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: '1 1 auto', minHeight: 0 }}>
                <button
                  onClick={() => setShowFilePicker(s => !s)}
                  style={{
                    width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                    padding: '10px 14px', background: 'var(--bg-surface)', border: 'none',
                    cursor: 'pointer', textAlign: 'left',
                  }}
                >
                  {showFilePicker
                    ? <ChevronDown size={14} color="var(--text-muted)" style={{ flexShrink: 0 }} />
                    : <ChevronRight size={14} color="var(--text-muted)" style={{ flexShrink: 0 }} />}
                  <FileText size={14} color="var(--text-secondary)" style={{ flexShrink: 0 }} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>
                    Select files to download
                  </span>
                  {!hfFilesLoading && hfFiles.length > 0 && (
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {selectedFiles.size}/{hfFiles.length} selected
                    </span>
                  )}
                </button>
                {showFilePicker && (
                  <div style={{ borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-base)', flex: '1 1 auto', minHeight: 0, overflowY: 'auto' }}>
                    {hfFilesLoading && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '14px', color: 'var(--text-muted)', fontSize: 12 }}>
                        <Loader2 size={13} className="animate-spin" /> Loading file tree from HuggingFace…
                      </div>
                    )}
                    {hfFilesError && (
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: '12px 14px', color: 'var(--danger)', fontSize: 12 }}>
                        <AlertCircle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                        <div style={{ flex: 1 }}>
                          {hfFilesError}
                          {hfFilesError.includes('401') || hfFilesError.includes('gated') ? (
                            <div style={{ marginTop: 4, color: 'var(--text-muted)', fontSize: 11 }}>
                              Add a token in <a href="/profile" style={{ color: 'var(--primary)' }}>Profile → HuggingFace Token</a> then retry
                            </div>
                          ) : null}
                        </div>
                      </div>
                    )}
                    {!hfFilesLoading && !hfFilesError && hfFiles.length === 0 && (
                      <div style={{ padding: '14px', color: 'var(--text-muted)', fontSize: 12 }}>
                        No downloadable files in this repo.
                      </div>
                    )}
                    {hfFiles.length > 0 && (
                      <>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px', borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}>
                          <button
                            onClick={() => setSelectedFiles(new Set(hfFiles.map(f => f.path)))}
                            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 11, color: 'var(--primary)', fontWeight: 600, padding: '2px 4px' }}
                          >
                            Select all ({hfFiles.length})
                          </button>
                          <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>·</span>
                          <button
                            onClick={() => setSelectedFiles(new Set())}
                            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, padding: '2px 4px' }}
                          >
                            Deselect all
                          </button>
                          <span style={{ flex: 1 }} />
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                            {hfFiles.filter(f => selectedFiles.has(f.path)).reduce((s, f) => s + f.size, 0) > 0
                              ? _hfFmtSize(hfFiles.filter(f => selectedFiles.has(f.path)).reduce((s, f) => s + f.size, 0))
                              : ''}
                          </span>
                        </div>
                        {hfFiles.map(f => {
                          const checked = selectedFiles.has(f.path)
                          const Icon = _hfFileIcon(f.path)
                          const lower = f.path.toLowerCase()
                          const isArchive = lower.endsWith('.zip') || lower.endsWith('.tar') || lower.endsWith('.tar.gz') || lower.endsWith('.tgz')
                          const isText = !isArchive && (
                            lower.endsWith('.md') || lower.endsWith('.txt') || lower.endsWith('.json') || lower.endsWith('.jsonl') ||
                            lower.endsWith('.csv') || lower.endsWith('.tsv') || lower.endsWith('.yaml') || lower.endsWith('.yml') ||
                            lower.endsWith('.log') || lower.endsWith('.xml') || lower.endsWith('.html') ||
                            lower.endsWith('.py') || lower.endsWith('.js') || lower.endsWith('.ts')
                          )
                          const canPreview = isArchive || isText
                          return (
                            <label
                              key={f.path}
                              style={{
                                display: 'flex', alignItems: 'center', gap: 10,
                                padding: '7px 14px',
                                borderBottom: '1px solid var(--border-subtle)',
                                cursor: 'pointer', background: checked ? '#f9731610' : 'transparent',
                              }}
                            >
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => {
                                  setSelectedFiles(prev => {
                                    const next = new Set(prev)
                                    if (next.has(f.path)) next.delete(f.path)
                                    else next.add(f.path)
                                    return next
                                  })
                                }}
                                style={{ accentColor: '#f97316', cursor: 'pointer' }}
                              />
                              <Icon size={13} color={checked ? '#f97316' : 'var(--text-muted)'} style={{ flexShrink: 0 }} />
                              <span style={{ fontSize: 12, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {f.path}
                              </span>
                              {canPreview && (
                                <button
                                  type="button"
                                  onClick={e => { e.preventDefault(); e.stopPropagation(); inspectArchive(f.path) }}
                                  title="Preview archive contents"
                                  style={{
                                    background: 'none', border: 'none', padding: 2,
                                    color: inspectFile === f.path ? '#f97316' : 'var(--text-muted)',
                                    cursor: 'pointer', display: 'flex', alignItems: 'center',
                                    borderRadius: 4,
                                  }}
                                >
                                  <Eye size={13} />
                                </button>
                              )}
                              <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', flexShrink: 0, minWidth: 60, textAlign: 'right' }}>
                                {_hfFmtSize(f.size)}
                              </span>
                            </label>
                          )
                        })}
                      </>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Bucket name */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                MinIO Bucket Name <span style={{ color: 'var(--danger)' }}>*</span>
              </label>
              <input
                value={bucketName}
                onChange={e => { setBucketName(e.target.value); setBucketError(validateBucket(e.target.value)) }}
                placeholder="e.g. hf-squad"
                style={{ width: '100%', padding: '8px 10px', borderRadius: 7, fontSize: 13, border: `1px solid ${bucketError ? 'var(--danger)' : 'var(--border-default)'}`, background: 'var(--bg-surface)', color: 'var(--text-primary)', boxSizing: 'border-box' }}
              />
              {bucketError
                ? <p style={{ fontSize: 11, color: 'var(--danger)', marginTop: 4 }}>{bucketError}</p>
                : <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>lowercase letters, numbers, hyphens · 3–63 chars</p>
              }
            </div>

            {/* Max files (cap on top of user's explicit selection) */}
            <div style={{ marginBottom: 22 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                Max files to download
                <span style={{ marginLeft: 8, fontWeight: 400, color: 'var(--text-muted)' }}>
                  (hard cap on top of the file picker — use {maxFiles} to limit huge selections)
                </span>
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <input
                  type="range" min={1} max={100} value={maxFiles}
                  onChange={e => setMaxFiles(Number(e.target.value))}
                  style={{ flex: 1, accentColor: '#f97316' }}
                />
                <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', minWidth: 30, textAlign: 'right' }}>{maxFiles}</span>
              </div>
            </div>
            </div>

            {/* Right column — HF detail (description / tags / downloads) */}
            {selected && (
              <div
                style={{
                  flex: '0 0 44%', minWidth: 0, minHeight: 0,
                  background: 'var(--bg-base)',
                  border: '1px solid var(--border-default)',
                  borderRadius: 9,
                  padding: 16,
                  overflowY: 'auto',
                  display: 'flex', flexDirection: 'column', gap: 12,
                }}
              >
                {hfDetailLoading && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 12 }}>
                    <Loader2 size={13} className="animate-spin" /> Loading details from HuggingFace…
                  </div>
                )}
                {!hfDetailLoading && hfDetail && (
                  <>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.3 }}>{hfDetail.id}</div>
                      {hfDetail.author && hfDetail.author !== hfDetail.id.split('/')[0] && (
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>by {hfDetail.author}</div>
                      )}
                    </div>
                    {/* Tags */}
                    {hfDetail.tags && hfDetail.tags.length > 0 && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {hfDetail.tags.slice(0, 12).map(t => (
                          <span key={t} style={{ fontSize: 10, background: 'var(--bg-elevated)', color: 'var(--text-muted)', borderRadius: 4, padding: '2px 6px' }}>{t}</span>
                        ))}
                      </div>
                    )}
                    {/* Stats */}
                    <div style={{ display: 'flex', gap: 14, fontSize: 11, color: 'var(--text-muted)' }}>
                      {hfDetail.downloads !== undefined && (
                        <span>↓ <strong style={{ color: 'var(--text-primary)' }}>{hfDetail.downloads >= 1000 ? `${(hfDetail.downloads/1000).toFixed(1)}k` : hfDetail.downloads}</strong> downloads</span>
                      )}
                      {hfDetail.likes !== undefined && (
                        <span>♥ <strong style={{ color: 'var(--text-primary)' }}>{hfDetail.likes}</strong></span>
                      )}
                      {hfDetail.license && (
                        <span>📄 {hfDetail.license}</span>
                      )}
                      {hfDetail.gated && (
                        <span style={{ color: '#f97316', fontWeight: 600 }}>⚠ Gated</span>
                      )}
                    </div>
                    {/* Description */}
                    {hfDetail.description && (
                      <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 10, marginTop: 2 }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
                          Description
                        </div>
                        <pre style={{
                          fontSize: 12, lineHeight: 1.55, color: 'var(--text-secondary)',
                          fontFamily: 'var(--font-sans, inherit)', whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word', margin: 0,
                          maxHeight: 360, overflowY: 'auto',
                          padding: '8px 10px', background: 'var(--bg-surface)', borderRadius: 6,
                          border: '1px solid var(--border-subtle)',
                        }}>
                          {_hfStripHtml(hfDetail.description).slice(0, 4000)}
                          {_hfStripHtml(hfDetail.description).length > 4000 ? '\n…(truncated)' : ''}
                        </pre>
                      </div>
                    )}
                    {!hfDetail.description && !hfDetailLoading && (
                      <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        No description provided in the dataset card.
                      </div>
                    )}
                  </>
                )}
                {!hfDetailLoading && !hfDetail && (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    Could not load details (repo may be private or rate-limited).
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Action buttons — outside the 2-column wrapper so the left
            column's flex: 1 doesn't squish them to zero width. */}
        {!jobId && (
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 16, flexShrink: 0 }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button
              className="btn btn-primary"
              style={{ background: '#f97316', borderColor: '#f97316', display: 'flex', alignItems: 'center', gap: 7, opacity: !selected || !bucketName ? 0.6 : 1 }}
              onClick={startImport}
              disabled={!selected || !bucketName}
            >
              <Package size={14} /> Import Dataset
            </button>
          </div>
        )}

        {/* ── Archive-inspect popover (eye icon) ── */}
        {inspectFile && (
          <div
            className="modal-overlay"
            style={{ zIndex: 1100, background: 'rgba(0,0,0,0.55)' }}
            onClick={closeInspect}
          >
            <div
              className="modal"
              style={{ maxWidth: 720, width: '92vw', maxHeight: '80vh', padding: 22, display: 'flex', flexDirection: 'column' }}
              onClick={e => e.stopPropagation()}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                <BookText size={18} color="#f97316" />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                    {(inspectData && (inspectData.kind === 'zip' || inspectData.kind === 'tar')) ? 'Archive contents' : 'File preview'}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {selected?.id}/{inspectFile}
                  </div>
                </div>
                <button className="btn btn-secondary btn-sm" onClick={closeInspect}>✕</button>
              </div>
              {inspectLoading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: 20, color: 'var(--text-muted)', fontSize: 12 }}>
                  <Loader2 size={14} className="animate-spin" /> Downloading archive and listing entries (up to 200 MB)…
                </div>
              )}
              {inspectError && (
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: 12, color: 'var(--danger)', fontSize: 12 }}>
                  <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} /> {inspectError}
                </div>
              )}
              {inspectData && inspectData.kind === 'unsupported' && (
                <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>
                  {inspectData.message}
                </div>
              )}
              {inspectData && inspectData.kind === 'too_large' && (
                <div style={{ padding: 16, color: 'var(--warning, #f59e0b)', fontSize: 12 }}>
                  {inspectData.message} ({_hfFmtSize(inspectData.size_bytes || 0)} downloaded so far)
                </div>
              )}
              {inspectData && (inspectData.kind === 'zip' || inspectData.kind === 'tar') && (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                    <span>{inspectData.entry_count} entries</span>
                    <span>·</span>
                    <span>{_hfFmtSize(inspectData.size_bytes || 0)} on disk</span>
                    {inspectData.truncated && (
                      <>
                        <span>·</span>
                        <span style={{ color: 'var(--warning, #f59e0b)' }}>showing first 500</span>
                      </>
                    )}
                  </div>
                  <div style={{ flex: 1, overflowY: 'auto', border: '1px solid var(--border-subtle)', borderRadius: 6, background: 'var(--bg-base)', maxHeight: 440 }}>
                    {inspectData.entries.map((e: any, i: number) => {
                      const isDir = e.is_dir
                      const I = isDir ? Folder : File
                      return (
                        <div
                          key={i}
                          style={{
                            display: 'flex', alignItems: 'center', gap: 8,
                            padding: '5px 12px',
                            borderBottom: '1px solid var(--border-subtle)',
                            fontSize: 11, fontFamily: 'var(--font-mono)',
                          }}
                        >
                          <I size={12} color={isDir ? 'var(--text-muted)' : 'var(--text-secondary)'} style={{ flexShrink: 0 }} />
                          <span style={{ flex: 1, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {e.path}
                          </span>
                          {!isDir && e.size > 0 && (
                            <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>{_hfFmtSize(e.size)}</span>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </>
              )}
              {inspectData && inspectData.kind === 'text' && (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                    <span>{_hfFmtSize(inspectData.preview_bytes || 0)} previewed</span>
                    {inspectData.size_bytes != null && (
                      <>
                        <span>·</span>
                        <span>file is {_hfFmtSize(inspectData.size_bytes)}</span>
                      </>
                    )}
                    {inspectData.truncated && (
                      <>
                        <span>·</span>
                        <span style={{ color: 'var(--warning, #f59e0b)' }}>truncated to 256 KB</span>
                      </>
                    )}
                  </div>
                  <pre
                    style={{
                      flex: 1, overflowY: 'auto', overflowX: 'auto',
                      border: '1px solid var(--border-subtle)', borderRadius: 6,
                      background: 'var(--bg-base)', maxHeight: 440, margin: 0,
                      padding: '12px 14px',
                      fontSize: 12, lineHeight: 1.5,
                      color: 'var(--text-secondary)',
                      fontFamily: 'var(--font-mono)',
                      whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                    }}
                  >
                    {inspectData.content}
                  </pre>
                </>
              )}
            </div>
          </div>
        )}

        {/* ── Progress ── */}
        {jobId && job && (
          <div>
            {/* Status row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <div style={{ width: 40, height: 40, borderRadius: 10, background: isDone ? '#10b98120' : isError ? '#ef444420' : '#f9731620', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                {isRunning ? <Loader2 size={20} color="#f97316" className="animate-spin" /> :
                 isDone    ? <CheckCircle2 size={20} color="#10b981" /> :
                             <XCircle size={20} color="#ef4444" />}
              </div>
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                  {isRunning ? `Importing ${job.hf_id}…` : isDone ? 'Import complete!' : 'Import failed'}
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0, marginTop: 2 }}>
                  Bucket: <code style={{ fontFamily: 'var(--font-mono)' }}>{job.bucket}</code>
                  {job.total_files > 0 && ` · ${job.done_files}/${job.total_files} files`}
                </p>
              </div>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{job.progress}%</span>
            </div>

            {/* Progress bar */}
            <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-elevated)', marginBottom: 16, overflow: 'hidden' }}>
              <div style={{ height: '100%', borderRadius: 3, transition: 'width 0.4s ease', width: `${job.progress}%`, background: isDone ? '#10b981' : isError ? '#ef4444' : '#f97316' }} />
            </div>

            {/* Log */}
            <pre ref={logRef} style={{
              background: 'var(--bg-base)', border: '1px solid var(--border-subtle)', borderRadius: 8,
              padding: '10px 14px', fontSize: 11, lineHeight: 1.6, color: 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)', maxHeight: '30vh', overflowY: 'auto',
              whiteSpace: 'pre-wrap', wordBreak: 'break-all', marginBottom: 20,
            }}>
              {job.log || 'Starting…'}
            </pre>

            {isError && (
              <div style={{ padding: '10px 14px', borderRadius: 8, background: '#ef444418', border: '1px solid #ef444440', marginBottom: 16, fontSize: 12, color: '#ef4444' }}>
                {job.error}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              {isDone && (
                <a href="http://localhost:9001" target="_blank" rel="noopener noreferrer" className="btn btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                  <ExternalLink size={12} /> MinIO Console
                </a>
              )}
              <button
                className="btn btn-primary"
                style={{ background: isDone ? '#10b981' : '#ef4444', borderColor: isDone ? '#10b981' : '#ef4444' }}
                onClick={isDone ? onDone : onClose}
                disabled={isRunning}
              >
                {isDone ? '✓ Done' : isRunning ? 'Importing…' : 'Close'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
