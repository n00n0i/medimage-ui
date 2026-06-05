import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { BrainCircuit, Terminal, Pencil, Trash2, CheckCircle2, RefreshCw, Loader, X, ChevronDown, ChevronUp, RotateCcw, CheckSquare, Square, Download, ExternalLink, Search, Cloud, Wifi, WifiOff, Save, Eye, EyeOff, Globe, StopCircle, Copy, Check } from 'lucide-react'

function CopyLogButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      title="Copy logs"
      onClick={() => { navigator.clipboard.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000) }) }}
      style={{ position: 'absolute', top: 4, right: 4, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: 4, padding: '2px 5px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3, color: copied ? '#10b981' : 'var(--text-muted)', fontSize: 10 }}
    >
      {copied ? <Check size={10} /> : <Copy size={10} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

interface HFModel {
  id: string
  author?: string
  pipeline_tag?: string
  downloads?: number
  likes?: number
  tags?: string[]
}

interface Model {
  id: string
  name: string
  training_type: string
  model: string
  engine: string
  status: string
  progress: number
  created_at: number
  started_at: number | null
  finished_at: number | null
  error: string | null
  dataset: string
  epochs: number
  batch_size: number
  source?: string
  s3_weights_path?: string
  modal_url?: string
  modal_api_key?: string
  ray_serve_url?: string
  inference_provider?: string
}

// ─── Pretrained presets ───────────────────────────────────────────────────────

const PRETRAINED: Record<string, Array<{ label: string; arch: string; engine: string; hf?: string; domain?: ModelDomain }>> = {
  classification: [
    { label: 'EfficientNet-B4',    arch: 'efficientnet-b4',              engine: 'PyTorch',  hf: 'google/efficientnet-b4',                                  domain: 'generic' },
    { label: 'EfficientNet-B2',    arch: 'efficientnet-b2',              engine: 'PyTorch',  hf: 'google/efficientnet-b2',                                  domain: 'generic' },
    { label: 'ResNet-50',          arch: 'resnet50',                     engine: 'PyTorch',  hf: 'microsoft/resnet-50',                                     domain: 'generic' },
    { label: 'ConvNeXt-Tiny',      arch: 'convnext_tiny',                engine: 'PyTorch',  hf: 'facebook/convnext-tiny-224',                              domain: 'generic' },
    { label: 'ViT-Base/16',        arch: 'vit-b-16',                     engine: 'PyTorch',  hf: 'google/vit-base-patch16-224',                             domain: 'generic' },
    { label: 'Swin-Tiny',          arch: 'swin-tiny-patch4',             engine: 'PyTorch',  hf: 'microsoft/swin-tiny-patch4-window7-224',                  domain: 'generic' },
    { label: 'DINOv2-Small',       arch: 'vit_small_patch14_dinov2',     engine: 'TIMM',     hf: 'facebook/dinov2-small',                                   domain: 'industrial' },
    { label: 'EfficientViT-M5',    arch: 'efficientvit_m5',              engine: 'TIMM',     hf: 'mit-han-lab/efficientvit-m5',                             domain: 'edge' },
  ],
  detection: [
    { label: 'YOLOv8n',            arch: 'yolov8n',                      engine: 'PyTorch',  hf: 'Ultralytics/assets',                                      domain: 'edge' },
    { label: 'YOLOv8s',            arch: 'yolov8s',                      engine: 'PyTorch',  hf: 'Ultralytics/assets',                                      domain: 'generic' },
    { label: 'YOLOv8m',            arch: 'yolov8m',                      engine: 'PyTorch',  hf: 'Ultralytics/assets',                                      domain: 'generic' },
    { label: 'YOLOv9c',            arch: 'yolov9c',                      engine: 'PyTorch',  hf: 'Ultralytics/assets',                                      domain: 'industrial' },
    { label: 'RT-DETR-L',          arch: 'rtdetr-l',                     engine: 'PyTorch',  hf: 'PekingU/rtdetr_l',                                        domain: 'generic' },
    { label: 'DETR ResNet-50',     arch: 'detr-resnet-50',               engine: 'PyTorch',  hf: 'facebook/detr-resnet-50',                                 domain: 'generic' },
    { label: 'Grounding DINO',     arch: 'groundingdino-b',              engine: 'PyTorch',  hf: 'IDEA-Research/grounding-dino-base',                       domain: 'industrial' },
  ],
  segmentation: [
    { label: 'SAM ViT-B',          arch: 'sam-vit-b',                    engine: 'PyTorch',  hf: 'facebook/sam-vit-base',                                   domain: 'generic' },
    { label: 'SAM 2 Large',        arch: 'sam2-l',                       engine: 'PyTorch',  hf: 'facebook/sam2-hiera-large',                               domain: 'generic' },
    { label: 'YOLOv8-Seg',         arch: 'yolov8m-seg',                  engine: 'PyTorch',  hf: 'Ultralytics/assets',                                      domain: 'industrial' },
    { label: 'Mask R-CNN',         arch: 'maskrcnn_resnet50_fpn',        engine: 'PyTorch',  hf: 'torchvision/fasterrcnn_resnet50_fpn',                     domain: 'generic' },
    { label: 'SegFormer-B2',       arch: 'segformer-b2',                 engine: 'PyTorch',  hf: 'nvidia/segformer-b2-finetuned-ade-512-512',               domain: 'generic' },
    { label: 'Mask2Former',        arch: 'mask2former',                  engine: 'PyTorch',  hf: 'facebook/mask2former-swin-base-coco-panoptic',            domain: 'generic' },
  ],
}

interface ModelDetail extends Model {
  project_id: number
  model_name: string
  learning_rate: number
  optimizer: string
  imgsz: number
  notes: string
  log: string
}

function fmtDate(ts: number | null) {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString('th-TH', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function parseValAcc(log: string): string | null {
  const matches = [...log.matchAll(/val_acc=([\d.]+)/g)]
  if (!matches.length) return null
  return (parseFloat(matches[matches.length - 1][1]) * 100).toFixed(1) + '%'
}

function parseFinalLoss(log: string): string | null {
  const matches = [...log.matchAll(/val_loss=([\d.]+)/g)]
  if (!matches.length) return null
  return parseFloat(matches[matches.length - 1][1]).toFixed(4)
}

const TYPE_COLORS: Record<string, string> = {
  classification:  '#6366f1',
  detection:       '#f59e0b',
  segmentation:    '#10b981',
  'llm-text':      '#8b5cf6',
  'vlm-finetune':  '#3b82f6',
  'self-supervised':'#14b8a6',
  'export-edge':   '#64748b',
}

// ─── Domain badges ────────────────────────────────────────────────────────────
type ModelDomain = 'generic' | 'edge' | 'industrial' | 'medical' | 'reasoning' | 'multilingual'

const DOMAIN_CONFIG: Record<ModelDomain, { label: string; color: string }> = {
  generic:      { label: 'Generic',      color: '#64748b' },
  edge:         { label: 'Edge',         color: '#0d9488' },
  industrial:   { label: 'Industrial',   color: '#f97316' },
  medical:      { label: 'Medical',      color: '#ec4899' },
  reasoning:    { label: 'Reasoning',    color: '#6366f1' },
  multilingual: { label: 'Multilingual', color: '#0ea5e9' },
}

function inferDomain(model: string): ModelDomain {
  const m = model.toLowerCase()
  if (/med|chex|medsam|biomedclip|retinal|fundus|clinical|biopsy/.test(m)) return 'medical'
  if (/deepseek.*r1|qwen3/.test(m)) return 'reasoning'
  if (/qwen2[\.\-]5|qwen2-vl|qwen2_vl|internvl/.test(m)) return 'multilingual'
  if (/nano|mobilenet|efficientvit|smolvlm|smol.vlm|phi.?3[\._\-]5.?mini|paligemma|efficientsam/.test(m)) return 'edge'
  if (/padim|patchcore|groundingdino|grounding.dino|yolov9/.test(m)) return 'industrial'
  return 'generic'
}

function DomainBadge({ model, domain }: { model?: string; domain?: ModelDomain }) {
  const d: ModelDomain = domain ?? (model ? inferDomain(model) : 'generic')
  const cfg = DOMAIN_CONFIG[d]
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 10,
      background: cfg.color + '20', color: cfg.color, letterSpacing: '0.03em',
    }}>
      {cfg.label}
    </span>
  )
}

export default function Models() {
  const navigate = useNavigate()
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'classification' | 'detection' | 'segmentation'>('all')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  // Log modal
  const [logModel, setLogModel] = useState<{ id: string; name: string } | null>(null)
  const [logText, setLogText] = useState('')
  const [logLoading, setLogLoading] = useState(false)
  const logRef = useRef<HTMLPreElement>(null)

  // Edit modal
  const [editModel, setEditModel] = useState<ModelDetail | null>(null)
  const [editName, setEditName] = useState('')
  const [editNotes, setEditNotes] = useState('')
  const [editSaving, setEditSaving] = useState(false)

  // Delete confirm
  const [deleteId, setDeleteId] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

  // Import modal
  const [showImport, setShowImport] = useState(false)

  // Multi-select
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [deletingBulk, setDeletingBulk] = useState(false)

  const fetchModels = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/jobs?view=models')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const completed: Model[] = data.jobs || []
      setModels(completed)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchModels() }, [])

  // Fetch log
  const openLog = async (m: Model) => {
    setLogModel({ id: m.id, name: m.name })
    setLogLoading(true)
    setLogText('')
    try {
      const res = await fetch(`/api/jobs/${m.id}`)
      const data = await res.json()
      setLogText(data.log || '(ไม่มี log)')
      setTimeout(() => logRef.current?.scrollTo(0, logRef.current.scrollHeight), 50)
    } catch (e: any) {
      setLogText(`Error: ${e.message}`)
    } finally {
      setLogLoading(false)
    }
  }

  const refreshLog = async () => {
    if (!logModel) return
    setLogLoading(true)
    try {
      const res = await fetch(`/api/jobs/${logModel.id}`)
      const data = await res.json()
      setLogText(data.log || '(ไม่มี log)')
      setTimeout(() => logRef.current?.scrollTo(0, logRef.current.scrollHeight), 50)
    } finally {
      setLogLoading(false)
    }
  }

  // Open edit
  const openEdit = async (m: Model) => {
    const res = await fetch(`/api/jobs/${m.id}`)
    const data: ModelDetail = await res.json()
    setEditModel(data)
    setEditName(data.name)
    setEditNotes(data.notes || '')
  }

  const saveEdit = async () => {
    if (!editModel) return
    setEditSaving(true)
    try {
      await fetch(`/api/jobs/${editModel.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: editName, notes: editNotes }),
      })
      setEditModel(null)
      fetchModels()
    } finally {
      setEditSaving(false)
    }
  }

  // Delete
  const confirmDelete = async () => {
    if (!deleteId) return
    setDeleting(true)
    try {
      await fetch(`/api/jobs/${deleteId}?from_view=models`, { method: 'DELETE' })
      setDeleteId(null)
      setSelected(prev => { const s = new Set(prev); s.delete(deleteId); return s })
      fetchModels()
    } finally {
      setDeleting(false)
    }
  }

  const toggleSelect = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSelected(prev => {
      const s = new Set(prev)
      s.has(id) ? s.delete(id) : s.add(id)
      return s
    })
  }

  const toggleSelectAll = () => {
    const ids = filtered.map(m => m.id)
    const allSelected = ids.every(id => selected.has(id))
    setSelected(allSelected ? new Set() : new Set(ids))
  }

  const deleteSelected = async () => {
    if (!confirm(`ลบ ${selected.size} model ที่เลือก?\n\nการกระทำนี้ไม่สามารถย้อนกลับได้`)) return
    setDeletingBulk(true)
    await Promise.all([...selected].map(id => fetch(`/api/jobs/${id}?from_view=models`, { method: 'DELETE' })))
    setSelected(new Set())
    setDeletingBulk(false)
    fetchModels()
  }

  const filtered = filter === 'all' ? models : models.filter(m => m.training_type === filter)

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <BrainCircuit size={22} color="var(--primary)" />
            <h1 style={{ fontWeight: 700, fontSize: 22, color: 'var(--text-primary)' }}>Models</h1>
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
            จัดการ model ที่ train เสร็จแล้ว &middot; {models.length} model{models.length !== 1 ? 's' : ''}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: 6 }} onClick={fetchModels}>
            <RefreshCw size={14} /> Refresh
          </button>
          <button
            className="btn btn-primary"
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={() => setShowImport(true)}
          >
            <Download size={14} /> Import Model
          </button>
        </div>
      </div>

      {/* Global endpoint config */}
      <GlobalEndpointConfig />

      {/* Bulk action toolbar */}
      {selected.size > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px',
          background: 'var(--bg-elevated)', borderRadius: 10, marginBottom: 16,
          border: '1px solid var(--primary)', boxShadow: '0 0 0 3px var(--primary-dim, #6366f120)',
        }}>
          <CheckSquare size={16} color="var(--primary)" />
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>
            เลือกไว้ {selected.size} model
          </span>
          <button className="btn btn-secondary btn-sm" onClick={() => setSelected(new Set())}>ยกเลิก</button>
          <button
            className="btn btn-sm"
            style={{ background: 'var(--danger)', color: '#fff', border: 'none', display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={deleteSelected}
            disabled={deletingBulk}
          >
            {deletingBulk ? <Loader size={13} className="animate-spin" /> : <Trash2 size={13} />}
            Delete {selected.size}
          </button>
        </div>
      )}

      {/* Filter tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {(['all', 'classification', 'detection', 'segmentation'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              padding: '5px 14px',
              borderRadius: 20,
              fontSize: 12,
              fontWeight: 500,
              border: '1px solid',
              cursor: 'pointer',
              transition: 'all .15s',
              borderColor: filter === f ? 'var(--primary)' : 'var(--border-default)',
              background: filter === f ? 'var(--primary)' : 'var(--bg-surface)',
              color: filter === f ? '#fff' : 'var(--text-secondary)',
            }}
          >
            {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* States */}
      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-muted)', padding: '40px 0' }}>
          <Loader size={16} className="animate-spin" /> กำลังโหลด...
        </div>
      )}
      {error && (
        <div className="card" style={{ borderColor: 'var(--danger)', color: 'var(--danger)', fontSize: 13 }}>
          ⚠ {error} — Backend ยังไม่พร้อม
        </div>
      )}
      {!loading && !error && filtered.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: '60px 24px' }}>
          <BrainCircuit size={40} color="var(--text-muted)" style={{ margin: '0 auto 16px' }} />
          <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>ยังไม่มี model</p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>ไปที่ Train เพื่อเริ่ม training job</p>
        </div>
      )}

      {/* Model cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {/* Select-all row */}
        {filtered.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingBottom: 4 }}>
            <button
              onClick={toggleSelectAll}
              style={{ background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', fontSize: 12 }}
            >
              {filtered.every(m => selected.has(m.id))
                ? <CheckSquare size={15} color="var(--primary)" />
                : <Square size={15} />}
              เลือกทั้งหมด ({filtered.length})
            </button>
          </div>
        )}

        {filtered.map(m => {
          const typeColor = TYPE_COLORS[m.training_type] || '#6366f1'
          const expanded = expandedId === m.id
          const isSelected = selected.has(m.id)
          return (
            <div key={m.id} className="card" style={{ padding: 0, overflow: 'hidden', outline: isSelected ? '2px solid var(--primary)' : 'none', outlineOffset: -2 }}>
              {/* Main row */}
              <div
                style={{ padding: '16px 20px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 12 }}
                onClick={() => setExpandedId(expanded ? null : m.id)}
              >
                {/* Checkbox */}
                <button
                  onClick={e => toggleSelect(m.id, e)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', flexShrink: 0, padding: 0, color: isSelected ? 'var(--primary)' : 'var(--text-muted)' }}
                >
                  {isSelected ? <CheckSquare size={16} /> : <Square size={16} />}
                </button>

                {/* Icon */}
                <div style={{
                  width: 44, height: 44, borderRadius: 10, flexShrink: 0,
                  background: typeColor + '18', display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <BrainCircuit size={22} color={typeColor} />
                </div>

                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>
                      {m.name}
                    </span>
                    <span style={{
                      padding: '2px 8px', borderRadius: 12, fontSize: 10, fontWeight: 600,
                      background: typeColor + '20', color: typeColor, textTransform: 'uppercase', letterSpacing: '0.05em',
                    }}>
                      {m.training_type}
                    </span>
                    <span className="badge-success" style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <CheckCircle2 size={10} /> Completed
                    </span>
                    {m.source === 'imported' && (
                      <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 10, background: '#0ea5e920', color: '#0ea5e9', display: 'flex', alignItems: 'center', gap: 3 }}>
                        <Download size={9} /> Pretrained
                      </span>
                    )}
                    {m.training_type === 'llm-text' && (
                      <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 10, background: '#8b5cf620', color: '#8b5cf6' }}>
                        LLM
                      </span>
                    )}
                    {m.training_type === 'vlm-finetune' && (
                      <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 10, background: '#3b82f620', color: '#3b82f6' }}>
                        VLM
                      </span>
                    )}
                    <DomainBadge model={m.model} />
                    {m.modal_url && m.inference_provider === 'modal' && (
                      <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 10, background: '#8b5cf620', color: '#8b5cf6', display: 'flex', alignItems: 'center', gap: 3 }}>
                        <Cloud size={9} /> Modal
                      </span>
                    )}
                    {m.ray_serve_url && m.inference_provider === 'ray' && (
                      <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 10, background: '#f59e0b20', color: '#f59e0b', display: 'flex', alignItems: 'center', gap: 3 }}>
                        <Wifi size={9} /> Ray
                      </span>
                    )}
                  </div>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 3 }}>
                    {m.model} &middot; {m.engine} &middot; {m.dataset} &middot; {m.epochs} epochs
                  </p>
                </div>

                {/* Meta */}
                <div style={{ textAlign: 'right', flexShrink: 0 }}>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fmtDate(m.finished_at)}</p>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
                    #{m.id}
                  </p>
                </div>
                <div style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
                  {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
              </div>

              {/* Expanded detail */}
              {expanded && <ModelExpanded modelId={m.id} onLog={() => openLog(m)} onEdit={() => openEdit(m)} onDelete={() => setDeleteId(m.id)} onRetrain={() => navigate(`/train?retrain=${m.id}`)} />}
            </div>
          )
        })}
      </div>

      {/* Log Modal */}
      {logModel && (
        <div className="modal-overlay" onClick={() => setLogModel(null)}>
          <div className="modal" style={{ maxWidth: 800, width: '95vw' }} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div>
                <h3 style={{ fontWeight: 600, fontSize: 15, color: 'var(--text-primary)' }}>Training Log</h3>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>{logModel.name}</p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-secondary btn-sm" onClick={refreshLog}>
                  {logLoading ? <Loader size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => setLogModel(null)}><X size={13} /></button>
              </div>
            </div>
            <pre ref={logRef} style={{
              background: 'var(--bg-base)', border: '1px solid var(--border-subtle)', borderRadius: 8,
              padding: '12px 16px', fontSize: 12, lineHeight: 1.6, color: 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)', maxHeight: '60vh', overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
            }}>
              {logLoading && !logText ? 'Loading...' : logText || '(ไม่มี log)'}
            </pre>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {editModel && (
        <div className="modal-overlay" onClick={() => setEditModel(null)}>
          <div className="modal" style={{ maxWidth: 520 }} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
              <h3 style={{ fontWeight: 600, fontSize: 16, color: 'var(--text-primary)' }}>Edit Model</h3>
              <button className="btn btn-secondary btn-sm" onClick={() => setEditModel(null)}><X size={14} /></button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                  Model Name
                </label>
                <input
                  value={editName}
                  onChange={e => setEditName(e.target.value)}
                  style={{
                    width: '100%', padding: '8px 12px', borderRadius: 8, fontSize: 13,
                    border: '1px solid var(--border-default)', background: 'var(--bg-surface)',
                    color: 'var(--text-primary)', boxSizing: 'border-box',
                  }}
                />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                  Notes
                </label>
                <textarea
                  value={editNotes}
                  onChange={e => setEditNotes(e.target.value)}
                  rows={4}
                  placeholder="บันทึกหมายเหตุเกี่ยวกับ model นี้..."
                  style={{
                    width: '100%', padding: '8px 12px', borderRadius: 8, fontSize: 13,
                    border: '1px solid var(--border-default)', background: 'var(--bg-surface)',
                    color: 'var(--text-primary)', resize: 'vertical', boxSizing: 'border-box',
                    fontFamily: 'inherit',
                  }}
                />
              </div>

              {/* Read-only info */}
              <div style={{ background: 'var(--bg-base)', borderRadius: 8, padding: '12px 14px', fontSize: 12 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 16px' }}>
                  {[
                    ['Architecture', editModel.model_name],
                    ['Engine',       editModel.engine],
                    ['Epochs',       String(editModel.epochs)],
                    ['Batch size',   String(editModel.batch_size)],
                    ['LR',           String(editModel.learning_rate)],
                    ['Optimizer',    editModel.optimizer],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <span style={{ color: 'var(--text-muted)' }}>{k}: </span>
                      <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 24 }}>
              <button className="btn btn-secondary" onClick={() => setEditModel(null)}>Cancel</button>
              <button className="btn btn-primary" onClick={saveEdit} disabled={editSaving}>
                {editSaving ? <><Loader size={13} className="animate-spin" /> Saving...</> : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirm Modal */}
      {/* Import Model Modal */}
      {showImport && (
        <ImportModal onClose={() => setShowImport(false)} onImported={() => { setShowImport(false); fetchModels() }} />
      )}

      {deleteId && (
        <div className="modal-overlay" onClick={() => setDeleteId(null)}>
          <div className="modal" style={{ maxWidth: 420 }} onClick={e => e.stopPropagation()}>
            <h3 style={{ fontWeight: 600, fontSize: 16, color: 'var(--text-primary)', marginBottom: 10 }}>ลบ Model?</h3>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>
              การลบจะไม่สามารถกู้คืนได้ คุณแน่ใจหรือไม่?
            </p>
            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button className="btn btn-secondary" onClick={() => setDeleteId(null)}>Cancel</button>
              <button
                className="btn btn-primary"
                style={{ background: 'var(--danger)', borderColor: 'var(--danger)' }}
                onClick={confirmDelete}
                disabled={deleting}
              >
                {deleting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Expanded detail sub-component: fetches full model details when opened
function ModelExpanded({ modelId, onLog, onEdit, onDelete, onRetrain }: {
  modelId: string
  onLog: () => void
  onEdit: () => void
  onDelete: () => void
  onRetrain: () => void
}) {
  const [detail, setDetail] = useState<ModelDetail | null>(null)

  useEffect(() => {
    fetch(`/api/jobs/${modelId}`)
      .then(r => r.json())
      .then(setDetail)
  }, [modelId])

  const downloadWeights = async () => {
    try {
      const res = await fetch(`/api/jobs/${modelId}/download-weights`)
      if (!res.ok) throw new Error((await res.json()).detail || 'Download failed')
      const { url, filename } = await res.json()
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      a.target = '_blank'
      a.click()
    } catch (e: any) {
      alert(`Download failed: ${e.message}`)
    }
  }

  if (!detail) return (
    <div style={{ padding: '12px 20px 16px', borderTop: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: 12 }}>
      <Loader size={13} className="animate-spin" style={{ display: 'inline-block', marginRight: 8 }} />Loading...
    </div>
  )

  const valAcc = parseValAcc(detail.log)
  const valLoss = parseFinalLoss(detail.log)

  return (
    <div style={{ padding: '0 20px 16px', borderTop: '1px solid var(--border-subtle)' }}>
      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12, marginTop: 14, marginBottom: 16 }}>
        {[
          { label: 'Final Val Acc', value: valAcc ?? '—' },
          { label: 'Final Val Loss', value: valLoss ?? '—' },
          { label: 'Optimizer', value: detail.optimizer },
          { label: 'Image Size', value: `${detail.imgsz}px` },
          { label: 'Learning Rate', value: String(detail.learning_rate) },
          { label: 'Batch Size', value: String(detail.batch_size) },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'var(--bg-base)', borderRadius: 8, padding: '10px 12px',
            border: '1px solid var(--border-subtle)',
          }}>
            <p style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>{label}</p>
            <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{value}</p>
          </div>
        ))}
      </div>

      {/* Notes */}
      {detail.notes && (
        <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14, padding: '8px 12px', background: 'var(--bg-base)', borderRadius: 8, borderLeft: '3px solid var(--primary)' }}>
          {detail.notes}
        </p>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button className="btn btn-primary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6 }} onClick={onRetrain}>
          <RotateCcw size={13} /> Re-train
        </button>
        {detail.s3_weights_path && (
          <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--primary)', borderColor: 'var(--primary)' }} onClick={downloadWeights}>
            <Download size={13} /> Download Weights
          </button>
        )}
        <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6 }} onClick={onLog}>
          <Terminal size={13} /> View Log
        </button>
        <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6 }} onClick={onEdit}>
          <Pencil size={13} /> Edit
        </button>
        <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--danger)' }} onClick={onDelete}>
          <Trash2 size={13} /> Delete
        </button>
      </div>

      {/* Deployment section — all model types */}
      <DeploySection
        modelId={detail.id}
        trainingType={detail.training_type}
        initialProvider={detail.inference_provider ?? ''}
        initialModalUrl={detail.modal_url ?? ''}
        initialModalKey={detail.modal_api_key ?? ''}
        initialRayUrl={detail.ray_serve_url ?? ''}
        s3Path={detail.s3_weights_path ?? ''}
      />
    </div>
  )
}

// ─── Shared sub-components ──────────────────────────────────────────────────

function StatusBadge({ online, checking }: { online: boolean | null; checking: boolean }) {
  if (checking) return <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}><Loader size={10} className="animate-spin" /> Checking…</span>
  if (online === true)  return <span style={{ fontSize: 11, fontWeight: 600, color: '#22c55e', display: 'flex', alignItems: 'center', gap: 4 }}><Wifi size={10} /> Online</span>
  if (online === false) return <span style={{ fontSize: 11, fontWeight: 600, color: '#ef4444', display: 'flex', alignItems: 'center', gap: 4 }}><WifiOff size={10} /> Offline</span>
  return null
}

function RayServeTab({ rayUrl, setRayUrl, onAutoSave, modelId }: {
  rayUrl: string
  setRayUrl: (v: string) => void
  onAutoSave: (serveUrl: string) => Promise<void>
  modelId?: string
}) {
  const [modelDeployStatus, setModelDeployStatus] = useState<'idle' | 'deploying' | 'running' | 'error'>('idle')
  const [modelDeployLogs, setModelDeployLogs] = useState<string[]>([])
  const [showModelLogs, setShowModelLogs] = useState(false)

  // Auto-fill dashboard URL from localStorage ray_head_url
  const [dashUrl, setDashUrl] = useState(() => localStorage.getItem('ray_head_url') || 'http://100.68.53.118:8265')

  // Derive serve URL (8265→8000) whenever dashUrl changes
  useEffect(() => {
    if (dashUrl) {
      try {
        const u = new URL(dashUrl)
        const serve = `${u.protocol}//${u.hostname}:8000`
        if (serve !== rayUrl) setRayUrl(serve)
      } catch {
        const serve = dashUrl.replace(':8265', ':8000').replace(/\/$/, '')
        if (serve !== rayUrl) setRayUrl(serve)
      }
    }
  }, [dashUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  // Sync per-model deploy state on mount
  useEffect(() => {
    if (!modelId) return
    fetch(`/api/jobs/${modelId}/deploy-ray/status`).then(r => r.ok ? r.json() : null).then(d => {
      if (!d) return
      setModelDeployStatus(d.status)
      setModelDeployLogs(d.logs || [])
      if (d.status === 'running' && d.url) {
        setRayUrl(d.url)
      }
    }).catch(() => {})
  }, [modelId]) // eslint-disable-line react-hooks/exhaustive-deps

  // Poll while per-model deploying
  useEffect(() => {
    if (modelDeployStatus !== 'deploying' || !modelId) return
    const id = setInterval(async () => {
      try {
        const r = await fetch(`/api/jobs/${modelId}/deploy-ray/status`)
        const d = r.ok ? await r.json() : null
        if (!d) return
        setModelDeployStatus(d.status)
        setModelDeployLogs(d.logs || [])
        if (d.status !== 'deploying') {
          clearInterval(id)
          if (d.status === 'running' && d.url) {
            setRayUrl(d.url)
            await onAutoSave(d.url)
          }
        }
      } catch { /* ignore */ }
    }, 4000)
    return () => clearInterval(id)
  }, [modelDeployStatus, modelId]) // eslint-disable-line react-hooks/exhaustive-deps

  async function stopModel() {
    if (!modelId) return
    setModelDeployStatus('deploying')
    setModelDeployLogs(['Stopping Ray Serve deployment…'])
    setShowModelLogs(true)
    try {
      const r = await fetch(`/api/jobs/${modelId}/deploy-ray/stop`, { method: 'POST' })
      const d = r.ok ? await r.json() : null
      setModelDeployStatus('idle')
      setModelDeployLogs(d?.logs || ['Stopped'])
      setRayUrl('')
      await onAutoSave('')
    } catch (e) {
      setModelDeployStatus('error')
      setModelDeployLogs([(e as Error).message])
    }
  }

  async function deployModel() {
    setModelDeployStatus('deploying')
    setModelDeployLogs(['Submitting model to Ray cluster…'])
    setShowModelLogs(true)
    try {
      const r = await fetch(`/api/jobs/${modelId}/deploy-ray`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ray_dashboard_url: dashUrl }),
      })
      if (!r.ok) {
        const e = await r.json()
        setModelDeployStatus('error')
        setModelDeployLogs([e.detail || 'Deploy failed'])
      }
    } catch (e) {
      setModelDeployStatus('error')
      setModelDeployLogs([(e as Error).message])
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

      {/* ── Per-model one-click deploy (only when modelId provided) ── */}
      {modelId && (
        <div style={{ padding: '10px 12px', borderRadius: 8, background: modelDeployStatus === 'running' ? '#10b98110' : '#f59e0b0a', border: `1px solid ${modelDeployStatus === 'running' ? '#10b98130' : '#f59e0b25'}` }}>
          <div style={{ marginBottom: 8 }}>
            <label style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: 4 }}>Ray Dashboard URL (port 8265)</label>
            <input type="url" value={dashUrl} onChange={e => setDashUrl(e.target.value)}
              placeholder="http://100.68.53.118:8265"
              style={{ width: '100%', boxSizing: 'border-box', padding: '6px 10px', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 7, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Cloud size={13} color="#f59e0b" />
              Deploy โมเดลนี้บน Ray Cluster
            </span>
            {modelDeployStatus === 'running' && <span style={{ fontSize: 11, fontWeight: 600, color: '#10b981', display: 'flex', alignItems: 'center', gap: 4 }}><Wifi size={10} /> Live</span>}
            {modelDeployStatus === 'error'   && <span style={{ fontSize: 11, fontWeight: 600, color: '#ef4444' }}>Error</span>}
          </div>
          {modelDeployStatus === 'running' && rayUrl && (
            <div style={{ padding: '5px 8px', borderRadius: 6, background: '#10b98115', border: '1px solid #10b98130', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
              <CheckCircle2 size={11} color="#10b981" style={{ flexShrink: 0 }} />
              <code style={{ fontSize: 10, flex: 1, color: '#10b981', wordBreak: 'break-all' }}>{rayUrl}</code>
              <a href={rayUrl} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)', display: 'flex' }}><ExternalLink size={11} /></a>
            </div>
          )}
          <div style={{ display: 'flex', gap: 7, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className="btn btn-sm"
              style={{ display: 'flex', alignItems: 'center', gap: 5, background: '#f59e0b', color: '#fff', border: 'none' }}
              onClick={deployModel}
              disabled={modelDeployStatus === 'deploying' || !dashUrl}
            >
              {modelDeployStatus === 'deploying' ? <Loader size={11} className="animate-spin" /> : <Cloud size={11} />}
              {modelDeployStatus === 'running' ? 'Redeploy' : modelDeployStatus === 'deploying' ? 'Deploying\u2026' : 'Deploy to Ray'}
            </button>
            {modelDeployLogs.length > 0 && (
              <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 4 }} onClick={() => setShowModelLogs(v => !v)}>
                <Terminal size={11} /> Logs
              </button>
            )}
            {modelDeployStatus === 'running' && (
              <button
                className="btn btn-sm"
                style={{ display: 'flex', alignItems: 'center', gap: 5, background: '#ef444420', color: '#ef4444', border: '1px solid #ef444440' }}
                onClick={stopModel}
              >
                <StopCircle size={11} /> Stop
              </button>
            )}
          </div>
          {showModelLogs && modelDeployLogs.length > 0 && (
            <div style={{ position: 'relative', margin: '8px 0 0' }}>
              <pre style={{ margin: 0, padding: '6px 8px', fontSize: 10, lineHeight: 1.5, borderRadius: 6, background: 'var(--bg-base)', color: modelDeployStatus === 'error' ? '#ef4444' : 'var(--text-secondary)', fontFamily: 'var(--font-mono)', maxHeight: 120, overflowY: 'auto', border: `1px solid ${modelDeployStatus === 'error' ? '#ef444440' : 'var(--border-subtle)'}` }}>{modelDeployLogs.join('\n')}</pre>
              <CopyLogButton text={modelDeployLogs.join('\n')} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Modal Deploy Tab ─────────────────────────────────────────────────────────

function ModalDeployTab({ modalUrl, setModalUrl, modalKey, setModalKey, online, checking }: {
  modalUrl: string
  setModalUrl: (v: string) => void
  modalKey: string
  setModalKey: (v: string) => void
  online: boolean | null
  checking: boolean
}) {
  const [tokenId, setTokenId]         = useState(() => {
    try { const s = JSON.parse(localStorage.getItem('ray_modal_config') || '{}') as Record<string, string>; return s.tokenId || '' } catch { return '' }
  })
  const [tokenSecret, setTokenSecret] = useState(() => {
    try { const s = JSON.parse(localStorage.getItem('ray_modal_config') || '{}') as Record<string, string>; return s.tokenSecret || '' } catch { return '' }
  })
  const [gpuType, setGpuType]         = useState('T4')
  const [deployStatus, setDeployStatus] = useState<'idle' | 'deploying' | 'running' | 'error'>('idle')
  const [deployLogs, setDeployLogs]   = useState<string[]>([])
  const [showLogs, setShowLogs]       = useState(false)
  const [showSecret, setShowSecret]   = useState(false)
  const [showApiKey, setShowApiKey]   = useState(false)

  // Persist tokens whenever they change
  useEffect(() => {
    try {
      const prev = JSON.parse(localStorage.getItem('ray_modal_config') || '{}')
      localStorage.setItem('ray_modal_config', JSON.stringify({ ...prev, tokenId, tokenSecret }))
    } catch { /* ignore */ }
  }, [tokenId, tokenSecret])

  // Sync deploy state on mount
  useEffect(() => {
    fetch('/api/modal/inference/status').then(r => r.ok ? r.json() : null).then(d => {
      if (!d) return
      setDeployStatus(d.status)
      setDeployLogs(d.logs || [])
      if (d.status === 'running' && d.url && !modalUrl) setModalUrl(d.url)
    }).catch(() => {})
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Poll while deploying
  useEffect(() => {
    if (deployStatus !== 'deploying') return
    const id = setInterval(async () => {
      try {
        const r = await fetch('/api/modal/inference/status')
        const d = r.ok ? await r.json() : null
        if (!d) return
        setDeployStatus(d.status)
        setDeployLogs(d.logs || [])
        if (d.status !== 'deploying') {
          clearInterval(id)
          if (d.status === 'running' && d.url) setModalUrl(d.url)
        }
      } catch { /* ignore */ }
    }, 3000)
    return () => clearInterval(id)
  }, [deployStatus]) // eslint-disable-line react-hooks/exhaustive-deps

  async function deploy() {
    if (!tokenId || !tokenSecret) return
    setDeployStatus('deploying')
    setDeployLogs(['กำลัง deploy บน Modal.com… (ใช้เวลา 1–3 นาที)'])
    setShowLogs(true)
    try {
      const r = await fetch('/api/modal/inference/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token_id: tokenId, token_secret: tokenSecret, gpu_type: gpuType }),
      })
      if (!r.ok) {
        const e = await r.json()
        setDeployStatus('error')
        setDeployLogs([e.detail || 'Deploy failed'])
      }
    } catch (e) {
      setDeployStatus('error')
      setDeployLogs([(e as Error).message])
    }
  }

  const inp: React.CSSProperties = {
    width: '100%', boxSizing: 'border-box', padding: '7px 10px', fontSize: 12,
    background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
    borderRadius: 7, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0 }}>
          Deploy inference app บน <strong style={{ color: '#8b5cf6' }}>Modal.com</strong> GPU
        </p>
        <StatusBadge online={online} checking={checking} />
      </div>

      {/* Deployed URL display */}
      {modalUrl && (
        <div style={{ padding: '7px 10px', borderRadius: 7, background: '#8b5cf615', border: '1px solid #8b5cf630', display: 'flex', alignItems: 'center', gap: 8 }}>
          <Cloud size={12} color="#8b5cf6" style={{ flexShrink: 0 }} />
          <code style={{ fontSize: 11, flex: 1, color: '#8b5cf6', wordBreak: 'break-all' }}>{modalUrl}</code>
          <a href={modalUrl} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)', display: 'flex' }}><ExternalLink size={12} /></a>
          <button onClick={() => setModalUrl('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 0, display: 'flex' }}><X size={12} /></button>
        </div>
      )}

      {/* Token ID */}
      <div>
        <label style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: 4 }}>Modal Token ID</label>
        <input type="text" value={tokenId} onChange={e => setTokenId(e.target.value)}
          placeholder="ak-xxxxxxxxxxxxxxxx"
          style={inp}
        />
      </div>

      {/* Token Secret */}
      <div>
        <label style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: 4 }}>Modal Token Secret</label>
        <div style={{ position: 'relative' }}>
          <input type={showSecret ? 'text' : 'password'} value={tokenSecret} onChange={e => setTokenSecret(e.target.value)}
            placeholder="as-xxxxxxxxxxxxxxxx"
            style={{ ...inp, paddingRight: 32 }}
          />
          <button onClick={() => setShowSecret(v => !v)} style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 0, display: 'flex' }}>
            {showSecret ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
          ดู token ได้ที่ <a href="https://modal.com/settings" target="_blank" rel="noopener noreferrer" style={{ color: '#8b5cf6' }}>modal.com/settings</a> — Token ID เก็บไว้ใน localStorage แล้ว
        </p>
      </div>

      {/* GPU type */}
      <div>
        <label style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: 4 }}>GPU Type</label>
        <select value={gpuType} onChange={e => setGpuType(e.target.value)} style={{ ...inp, cursor: 'pointer' }}>
          <option value="T4">T4 (ประหยัด)</option>
          <option value="A10G">A10G (แนะนำ)</option>
          <option value="A100">A100 (สูงสุด)</option>
          <option value="cpu">CPU only</option>
        </select>
      </div>

      {/* Deploy button + status */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <button
          className="btn btn-sm"
          style={{ display: 'flex', alignItems: 'center', gap: 5, background: '#8b5cf6', color: '#fff', border: 'none' }}
          onClick={deploy}
          disabled={deployStatus === 'deploying' || !tokenId || !tokenSecret}
        >
          {deployStatus === 'deploying' ? <Loader size={11} className="animate-spin" /> : <Cloud size={11} />}
          {deployStatus === 'running' ? 'Redeploy' : deployStatus === 'deploying' ? 'Deploying…' : 'Deploy Inference App'}
        </button>
        {deployLogs.length > 0 && (
          <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 4 }} onClick={() => setShowLogs(v => !v)}>
            <Terminal size={11} /> Logs
          </button>
        )}
        {deployStatus === 'running' && <span style={{ fontSize: 11, color: '#22c55e', display: 'flex', alignItems: 'center', gap: 4 }}><CheckCircle2 size={11} /> Deployed</span>}
        {deployStatus === 'error'   && <span style={{ fontSize: 11, color: '#ef4444' }}>Deploy failed</span>}
      </div>

      {showLogs && deployLogs.length > 0 && (
        <pre style={{
          margin: 0, padding: '8px 10px', fontSize: 10, lineHeight: 1.5, borderRadius: 7,
          background: 'var(--bg-base)', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)',
          maxHeight: 160, overflowY: 'auto', border: '1px solid var(--border-subtle)',
        }}>{deployLogs.join('\n')}</pre>
      )}

      {/* API key for already-deployed endpoint */}
      {modalUrl && (
        <div>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: 4 }}>API Key (ถ้ามี)</label>
          <div style={{ position: 'relative' }}>
            <input type={showApiKey ? 'text' : 'password'} value={modalKey} onChange={e => setModalKey(e.target.value)}
              placeholder="Bearer token (เว้นว่างถ้าไม่มี)"
              style={{ ...inp, paddingRight: 32 }}
            />
            <button onClick={() => setShowApiKey(v => !v)} style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 0, display: 'flex' }}>
              {showApiKey ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Global Endpoint Config ──────────────────────────────────────────────────

function GlobalEndpointConfig() {
  type Tab = 'modal' | 'ray'
  const [tab, setTab]           = useState<Tab>('ray')
  const [modalUrl, setModalUrl] = useState('')
  const [modalKey, setModalKey] = useState('')
  const [rayUrl, setRayUrl]     = useState('')
  const [saving, setSaving]     = useState(false)
  const [saved, setSaved]       = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [modalOnline, setModalOnline] = useState<boolean | null>(null)
  const [rayOnline, setRayOnline]     = useState<boolean | null>(null)
  const [checking, setChecking]       = useState(false)

  useEffect(() => {
    fetch('/api/settings/inference').then(r => r.ok ? r.json() : null).then(d => {
      if (!d) return
      if (d.provider) setTab(d.provider as Tab)
      if (d.modal_url) setModalUrl(d.modal_url)
      if (d.modal_api_key) setModalKey(d.modal_api_key)
      if (d.ray_serve_url) setRayUrl(d.ray_serve_url)
    }).catch(() => {})
  }, [])

  async function checkStatus() {
    setChecking(true)
    try {
      if (tab === 'modal') {
        const r = await fetch('/api/settings/inference/modal-status')
        const d = r.ok ? await r.json() : { online: false }
        setModalOnline(d.online)
      } else if (tab === 'ray') {
        const r = await fetch('/api/settings/inference/ray-status')
        const d = r.ok ? await r.json() : { online: false }
        setRayOnline(d.online)
      }
    } catch {
      if (tab === 'modal') setModalOnline(false)
      else setRayOnline(false)
    } finally {
      setChecking(false)
    }
  }

  async function save() {
    setSaving(true)
    try {
      await fetch('/api/settings/inference', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: tab,
          modal_url: modalUrl,
          modal_api_key: modalKey,
          ray_serve_url: rayUrl,
        }),
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
      if (tab === 'modal' && modalUrl) checkStatus()
    } finally {
      setSaving(false)
    }
  }

  const TABS: { id: Tab; label: string; color: string }[] = [
    { id: 'modal',    label: 'Modal',     color: '#8b5cf6' },
    { id: 'ray',      label: 'Ray Serve', color: '#f59e0b' },
  ]

  const activeColor = TABS.find(t => t.id === tab)?.color ?? '#f59e0b'

  return (
    <div style={{ marginBottom: 20, borderRadius: 12, border: `1px solid ${activeColor}40`, background: 'var(--bg-card)', overflow: 'hidden' }}>
      {/* Always-visible header */}
      <button
        onClick={() => setExpanded(v => !v)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 10,
          padding: '13px 18px', background: 'none', border: 'none', cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        <Globe size={15} color={activeColor} />
        <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)' }}>
          Global Inference Default
        </span>
        <span style={{
          fontSize: 11, fontWeight: 700, padding: '2px 9px', borderRadius: 10,
          background: activeColor + '20', color: activeColor,
        }}>
          {tab === 'modal' ? 'Modal' : 'Ray Serve'}
        </span>
        {(tab === 'modal' && modalOnline !== null) && (
          <span style={{ fontSize: 11, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4, color: modalOnline ? '#22c55e' : '#ef4444' }}>
            {modalOnline ? <Wifi size={10} /> : <WifiOff size={10} />}
            {modalOnline ? 'Online' : 'Offline'}
          </span>
        )}
        {(tab === 'ray' && rayOnline !== null) && (
          <span style={{ fontSize: 11, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4, color: rayOnline ? '#22c55e' : '#ef4444' }}>
            {rayOnline ? <Wifi size={10} /> : <WifiOff size={10} />}
            {rayOnline ? 'Online' : 'Offline'}
          </span>
        )}
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-muted)' }}>
          ทุก model ที่ไม่ได้ตั้งเองจะใช้ค่านี้
        </span>
        <ChevronDown size={14} color="var(--text-muted)" style={{ transform: expanded ? 'rotate(180deg)' : undefined, transition: 'transform 0.18s' }} />
      </button>

      {expanded && (
        <div style={{ padding: '0 18px 16px', borderTop: `1px solid ${activeColor}30` }}>
          {/* Tab switcher */}
          <div style={{ display: 'flex', gap: 6, margin: '14px 0 14px' }}>
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                style={{
                  padding: '4px 14px', borderRadius: 8, fontSize: 12, fontWeight: 600,
                  cursor: 'pointer', border: 'none',
                  background: tab === t.id ? t.color + '25' : 'var(--bg-base)',
                  color: tab === t.id ? t.color : 'var(--text-muted)',
                  outline: tab === t.id ? `1.5px solid ${t.color}60` : 'none',
                }}
              >{t.label}</button>
            ))}
          </div>

          {tab === 'modal' && (
            <div style={{ marginBottom: 14 }}>
              <ModalDeployTab
                modalUrl={modalUrl} setModalUrl={setModalUrl}
                modalKey={modalKey} setModalKey={setModalKey}
                online={modalOnline} checking={checking && tab === 'modal'}
              />
            </div>
          )}

          {tab === 'ray' && (
            <div style={{ marginBottom: 14 }}>
              <RayServeTab
                rayUrl={rayUrl} setRayUrl={setRayUrl}
                onAutoSave={async (url) => {
                  setRayUrl(url)
                  setTab('ray')
                  await fetch('/api/settings/inference', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ provider: 'ray', modal_url: modalUrl, modal_api_key: modalKey, ray_serve_url: url }),
                  })
                  setSaved(true); setTimeout(() => setSaved(false), 2000)
                }}
              />
            </div>
          )}

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button className="btn btn-primary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 5 }} onClick={save} disabled={saving}>
              {saving ? <Loader size={11} className="animate-spin" /> : <Save size={11} />}
              {saved ? 'Saved!' : 'Save & Apply to All'}
            </button>
            {(modalUrl || rayUrl) && (
              <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 5 }} onClick={checkStatus} disabled={checking}>
                {checking ? <Loader size={11} className="animate-spin" /> : <RefreshCw size={11} />} Check Status
              </button>
            )}
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 10 }}>
            model ที่ตั้ง endpoint เองจะใช้ค่าของตัวเองแทน — global setting เป็นเพียง fallback
          </p>
        </div>
      )}
    </div>
  )
}


// ─── Deploy Section (Modal.com + Ray Serve endpoint config) ──────────────────

function DeploySection({ modelId, trainingType, initialProvider, initialModalUrl, initialModalKey, initialRayUrl, s3Path }: {
  modelId: string
  trainingType: string
  initialProvider: string
  initialModalUrl: string
  initialModalKey: string
  initialRayUrl: string
  s3Path: string
}) {
  type Tab = 'modal' | 'ray'
  const [tab, setTab]         = useState<Tab>((initialProvider as Tab) || 'ray')
  const [modalUrl, setModalUrl]   = useState(initialModalUrl)
  const [modalKey, setModalKey]   = useState(initialModalKey)
  const [rayUrl, setRayUrl]       = useState(initialRayUrl)
  const [saving, setSaving]       = useState(false)
  const [saved, setSaved]         = useState(false)
  const [modalOnline, setModalOnline] = useState<boolean | null>(null)
  const [checkingModal, setCheckingModal] = useState(false)

  useEffect(() => {
    if (initialModalUrl) checkStatus()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  async function checkStatus() {
    setCheckingModal(true)
    try {
      const r = await fetch(`/api/jobs/${modelId}/modal-status`)
      const d = r.ok ? await r.json() : { online: false }
      setModalOnline(d.online)
    } catch {
      setModalOnline(false)
    } finally {
      setCheckingModal(false)
    }
  }

  async function save() {
    setSaving(true)
    try {
      await fetch(`/api/jobs/${modelId}/deployment`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modal_url: modalUrl,
          modal_api_key: modalKey,
          ray_serve_url: rayUrl,
          inference_provider: tab,
        }),
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
      if (tab === 'modal' && modalUrl) checkStatus()
    } finally {
      setSaving(false)
    }
  }

  async function clearAll() {
    await fetch(`/api/jobs/${modelId}/deployment`, { method: 'DELETE' })
    setModalUrl(''); setModalKey(''); setRayUrl(''); setTab('ray')
    setModalOnline(null)
  }

  const TABS: { id: Tab; label: string; color: string }[] = [
    { id: 'modal',    label: 'Modal',     color: '#8b5cf6' },
    { id: 'ray',      label: 'Ray Serve', color: '#f59e0b' },
  ]

  const _isLLM = trainingType === 'llm-text' || trainingType === 'vlm-finetune'; void _isLLM

  return (
    <div style={{ marginTop: 18, borderRadius: 10, background: 'var(--bg-base)', border: '1px solid var(--border-subtle)', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <Cloud size={14} color="#8b5cf6" />
        <span style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)' }}>Inference Provider</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding: '3px 10px', borderRadius: 8, fontSize: 11, fontWeight: 600,
                cursor: 'pointer', border: 'none',
                background: tab === t.id ? t.color + '25' : 'transparent',
                color: tab === t.id ? t.color : 'var(--text-muted)',
                outline: tab === t.id ? `1.5px solid ${t.color}50` : 'none',
              }}
            >{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: '14px 16px' }}>
        {/* S3 weights path (display only, shown for all tabs) */}
        {s3Path && (
          <div style={{ marginBottom: 14, padding: '8px 10px', borderRadius: 7, background: '#0ea5e910', border: '1px solid #0ea5e930' }}>
            <p style={{ fontSize: 10, color: '#0ea5e9', fontWeight: 600, marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.05em' }}>S3 / MinIO Weights</p>
            <code style={{ fontSize: 11, color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{s3Path}</code>
          </div>
        )}

        {/* Modal tab */}
        {tab === 'modal' && (
          <ModalDeployTab
            modalUrl={modalUrl} setModalUrl={setModalUrl}
            modalKey={modalKey} setModalKey={setModalKey}
            online={modalOnline} checking={checkingModal}
          />
        )}

        {/* Ray Serve tab */}
        {tab === 'ray' && (
          <RayServeTab
            rayUrl={rayUrl} setRayUrl={setRayUrl}
            modelId={modelId}
            onAutoSave={async (url) => {
              setRayUrl(url)
              await fetch(`/api/jobs/${modelId}/deployment`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ modal_url: modalUrl, modal_api_key: modalKey, ray_serve_url: url, inference_provider: 'ray' }),
              })
              setTab('ray')
              setSaved(true); setTimeout(() => setSaved(false), 2000)
            }}
          />
        )}

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 14 }}>
          {tab === 'modal' && (
            <button className="btn btn-primary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 5 }} onClick={save} disabled={saving}>
              {saving ? <Loader size={11} className="animate-spin" /> : <Save size={11} />}
              {saved ? 'Saved!' : 'Save & Activate'}
            </button>
          )}
          {tab === 'modal' && modalUrl && (
            <>
              <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 5 }} onClick={checkStatus} disabled={checkingModal}>
                {checkingModal ? <Loader size={11} className="animate-spin" /> : <RefreshCw size={11} />} Check
              </button>
              <a href={modalUrl} target="_blank" rel="noopener noreferrer" className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 5, textDecoration: 'none' }}>
                <ExternalLink size={11} /> Open
              </a>
            </>
          )}
          {tab === 'modal' && rayUrl && null}
          {(modalUrl || modalKey || rayUrl) && (
            <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--danger)' }} onClick={clearAll}>
              <X size={11} /> Clear All
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Import Modal ─────────────────────────────────────────────────────────────

function ImportModal({ onClose, onImported }: { onClose: () => void; onImported: () => void }) {
  const [step, setStep] = useState<'form' | 'importing' | 'done'>('form')
  const [trainingType, setTrainingType] = useState<'classification' | 'detection' | 'segmentation'>('classification')
  const [sourceType, setSourceType] = useState<'huggingface' | 'url' | 'builtin'>('huggingface')
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [arch, setArch] = useState('')
  const [engine, setEngine] = useState('PyTorch')
  const [sourceUrl, setSourceUrl] = useState('')
  const [notes, setNotes] = useState('')
  const [progress, setProgress] = useState(0)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // HuggingFace search
  const [hfQuery, setHfQuery] = useState('')
  const [hfResults, setHfResults] = useState<HFModel[]>([])
  const [hfSearching, setHfSearching] = useState(false)
  const [hfSearched, setHfSearched] = useState(false)

  const searchHF = async () => {
    if (!hfQuery.trim()) return
    setHfSearching(true)
    setHfResults([])
    try {
      const res = await fetch(
        `https://huggingface.co/api/models?search=${encodeURIComponent(hfQuery.trim())}&limit=10&sort=downloads&direction=-1`,
        { signal: AbortSignal.timeout(8000) }
      )
      const data: HFModel[] = await res.json()
      setHfResults(data)
    } catch {
      setHfResults([])
    } finally {
      setHfSearching(false)
      setHfSearched(true)
    }
  }

  const applyHFResult = (r: HFModel) => {
    const archGuess = r.id.split('/').pop() ?? r.id
    setSourceType('huggingface')
    setSourceUrl(r.id)
    if (!name) setName(archGuess)
    setArch(archGuess)
    setSelectedPreset(null)
    setHfResults([])
    setHfQuery('')
    setHfSearched(false)
  }

  const presets = PRETRAINED[trainingType] ?? []

  const applyPreset = (preset: typeof presets[0]) => {
    setSelectedPreset(preset.label)
    setArch(preset.arch)
    setEngine(preset.engine)
    setName(preset.label)
    if (preset.hf) {
      setSourceType('huggingface')
      setSourceUrl(preset.hf)
    }
  }

  const handleImport = async () => {
    if (!name || !arch) return
    setStep('importing')
    setProgress(0)
    try {
      const res = await fetch('/api/models/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, training_type: trainingType, model_name: arch, engine, source_type: sourceType, source_url: sourceUrl, notes }),
      })
      const data = await res.json()
      const jobId = data.job_id

      pollRef.current = setInterval(async () => {
        const r = await fetch(`/api/jobs/${jobId}`)
        const d = await r.json()
        setProgress(d.progress ?? 0)
        if (d.status === 'completed' || d.status === 'error') {
          clearInterval(pollRef.current!)
          setStep('done')
        }
      }, 600)
    } catch {
      setStep('form')
    }
  }

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const TYPE_COLOR: Record<string, string> = { classification: '#6366f1', detection: '#f59e0b', segmentation: '#10b981' }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 620, width: '95vw', padding: 28, maxHeight: '92vh', overflowY: 'auto', overflowX: 'hidden' }} onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 34, height: 34, borderRadius: 8, background: '#0ea5e920', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Download size={17} color="#0ea5e9" />
            </div>
            <div>
              <h3 style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-primary)', margin: 0 }}>Import Pretrained Model</h3>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: 0 }}>Load weights from HuggingFace Hub or a direct URL</p>
            </div>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={onClose}><X size={14} /></button>
        </div>

        {step === 'form' && (
          <>
            {/* Training type tabs */}
            <div style={{ display: 'flex', gap: 6, marginBottom: 18 }}>
              {(['classification', 'detection', 'segmentation'] as const).map(t => (
                <button key={t} onClick={() => { setTrainingType(t); setSelectedPreset(null) }} style={{
                  flex: 1, padding: '7px 0', borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: 'pointer',
                  border: `1px solid ${trainingType === t ? TYPE_COLOR[t] : 'var(--border-default)'}`,
                  background: trainingType === t ? TYPE_COLOR[t] + '18' : 'var(--bg-surface)',
                  color: trainingType === t ? TYPE_COLOR[t] : 'var(--text-muted)',
                  transition: 'all .15s',
                }}>
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>

            {/* HuggingFace Search */}
            <div style={{ marginBottom: 18 }}>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>
                Search HuggingFace Hub
              </label>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  value={hfQuery}
                  onChange={e => setHfQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && searchHF()}
                  placeholder="e.g. yolov8, bert-base, whisper-small..."
                  style={{ flex: 1, padding: '8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)' }}
                />
                <button
                  onClick={searchHF}
                  disabled={hfSearching || !hfQuery.trim()}
                  style={{ padding: '8px 14px', borderRadius: 7, fontSize: 12, fontWeight: 600, cursor: 'pointer', border: '1px solid var(--border-default)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap', opacity: hfSearching || !hfQuery.trim() ? 0.5 : 1 }}
                >
                  {hfSearching ? <Loader size={13} className="animate-spin" /> : <Search size={13} />}
                  Search
                </button>
              </div>

              {/* Results */}
              {hfResults.length > 0 && (
                <div style={{ marginTop: 8, border: '1px solid var(--border-default)', borderRadius: 8, overflow: 'hidden', maxHeight: 220, overflowY: 'auto' }}>
                  {hfResults.map((r, i) => (
                    <button
                      key={r.id}
                      onClick={() => applyHFResult(r)}
                      style={{
                        width: '100%', display: 'flex', alignItems: 'center', gap: 10, padding: '9px 12px', cursor: 'pointer',
                        background: i % 2 === 0 ? 'var(--bg-surface)' : 'var(--bg-base)',
                        border: 'none', borderBottom: i < hfResults.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                        textAlign: 'left',
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {r.id}
                        </div>
                        {r.pipeline_tag && (
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 1 }}>{r.pipeline_tag}</div>
                        )}
                      </div>
                      <div style={{ flexShrink: 0, textAlign: 'right' }}>
                        {r.downloads !== undefined && (
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                            ↓ {r.downloads >= 1000 ? (r.downloads / 1000).toFixed(0) + 'k' : r.downloads}
                          </div>
                        )}
                        {r.likes !== undefined && (
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>♥ {r.likes}</div>
                        )}
                      </div>
                      <span style={{ fontSize: 11, color: 'var(--primary)', fontWeight: 600, flexShrink: 0 }}>Select →</span>
                    </button>
                  ))}
                </div>
              )}
              {hfSearched && hfResults.length === 0 && !hfSearching && (
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, textAlign: 'center' }}>ไม่พบผลลัพธ์ ลองใช้คำค้นอื่น</p>
              )}
            </div>

            {/* Presets */}
            <div style={{ marginBottom: 18 }}>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>
                Quick Presets
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {presets.map(p => (
                  <button key={p.label} onClick={() => applyPreset(p)} style={{
                    padding: '5px 10px', borderRadius: 20, fontSize: 12, cursor: 'pointer',
                    border: `1px solid ${selectedPreset === p.label ? TYPE_COLOR[trainingType] : 'var(--border-default)'}`,
                    background: selectedPreset === p.label ? TYPE_COLOR[trainingType] + '18' : 'var(--bg-surface)',
                    color: selectedPreset === p.label ? TYPE_COLOR[trainingType] : 'var(--text-secondary)',
                    fontWeight: selectedPreset === p.label ? 600 : 400,
                    transition: 'all .1s',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}>
                    {p.label}
                    {p.domain && (() => {
                      const cfg = DOMAIN_CONFIG[p.domain]
                      return (
                        <span style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 8, background: cfg.color + '22', color: cfg.color }}>
                          {cfg.label}
                        </span>
                      )
                    })()}
                  </button>
                ))}
              </div>
            </div>

            {/* Form fields */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 5 }}>Model Name *</label>
                  <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. YOLOv8s Medical" style={{ width: '100%', padding: '8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)', boxSizing: 'border-box' }} />
                </div>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 5 }}>Architecture *</label>
                  <input value={arch} onChange={e => setArch(e.target.value)} placeholder="e.g. yolov8s" style={{ width: '100%', padding: '8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)', boxSizing: 'border-box' }} />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 5 }}>Engine</label>
                  <select value={engine} onChange={e => setEngine(e.target.value)} style={{ width: '100%', padding: '8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)' }}>
                    {['PyTorch', 'ONNX', 'TensorFlow', 'TensorRT', 'Hugging Face'].map(e => <option key={e}>{e}</option>)}
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 5 }}>Source</label>
                  <select value={sourceType} onChange={e => setSourceType(e.target.value as 'huggingface' | 'url' | 'builtin')} style={{ width: '100%', padding: '8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)' }}>
                    <option value="huggingface">HuggingFace Hub</option>
                    <option value="url">Direct URL</option>
                    <option value="builtin">Built-in</option>
                  </select>
                </div>
              </div>

              {sourceType !== 'builtin' && (
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 5 }}>
                    {sourceType === 'huggingface' ? 'HuggingFace Repo ID' : 'Download URL'}
                  </label>
                  <div style={{ position: 'relative' }}>
                    <input
                      value={sourceUrl} onChange={e => setSourceUrl(e.target.value)}
                      placeholder={sourceType === 'huggingface' ? 'e.g. google/vit-base-patch16-224' : 'https://…/model.pt'}
                      style={{ width: '100%', padding: '8px 36px 8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)', boxSizing: 'border-box' }}
                    />
                    {sourceType === 'huggingface' && sourceUrl && (
                      <a href={`https://huggingface.co/${sourceUrl}`} target="_blank" rel="noopener noreferrer" style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}>
                        <ExternalLink size={13} />
                      </a>
                    )}
                  </div>
                </div>
              )}

              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 5 }}>Notes</label>
                <textarea value={notes} onChange={e => setNotes(e.target.value)} rows={2} placeholder="Notes about dataset, weights origin, etc." style={{ width: '100%', padding: '8px 10px', borderRadius: 7, fontSize: 13, border: '1px solid var(--border-default)', background: 'var(--bg-surface)', color: 'var(--text-primary)', resize: 'vertical', boxSizing: 'border-box', fontFamily: 'inherit' }} />
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 22 }}>
              <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
              <button
                className="btn btn-primary"
                style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#0ea5e9', borderColor: '#0ea5e9' }}
                onClick={handleImport}
                disabled={!name || !arch}
              >
                <Download size={14} /> Import Model
              </button>
            </div>
          </>
        )}

        {step === 'importing' && (
          <div style={{ padding: '20px 0', textAlign: 'center' }}>
            <div style={{ width: 56, height: 56, borderRadius: 14, background: '#0ea5e918', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
              <Download size={26} color="#0ea5e9" />
            </div>
            <p style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>Importing {name}…</p>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20 }}>
              {sourceType === 'huggingface' ? `Downloading from HuggingFace: ${sourceUrl}` : sourceUrl || 'Loading built-in weights'}
            </p>
            <div style={{ height: 8, borderRadius: 4, background: 'var(--bg-elevated)', overflow: 'hidden', maxWidth: 360, margin: '0 auto' }}>
              <div style={{ height: '100%', width: `${progress}%`, background: 'linear-gradient(90deg, #0ea5e9, #6366f1)', borderRadius: 4, transition: 'width 0.4s ease' }} />
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, fontVariantNumeric: 'tabular-nums' }}>{progress}%</p>
          </div>
        )}

        {step === 'done' && (
          <div style={{ padding: '20px 0', textAlign: 'center' }}>
            <div style={{ width: 56, height: 56, borderRadius: 14, background: '#10b98118', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
              <CheckCircle2 size={28} color="#10b981" />
            </div>
            <p style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>Import Complete!</p>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 22 }}>{name} is ready. You can now test it in Playground.</p>
            <button className="btn btn-primary" onClick={onImported} style={{ background: '#10b981', borderColor: '#10b981' }}>
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
