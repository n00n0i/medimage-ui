import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { BrainCircuit, Terminal, Pencil, Trash2, CheckCircle2, RefreshCw, Loader, X, ChevronDown, ChevronUp, RotateCcw, CheckSquare, Square, Download, ExternalLink } from 'lucide-react'

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
