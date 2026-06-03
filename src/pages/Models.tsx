import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { BrainCircuit, Terminal, Pencil, Trash2, CheckCircle2, RefreshCw, Loader, X, ChevronDown, ChevronUp, RotateCcw, CheckSquare, Square } from 'lucide-react'

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
  classification: '#6366f1',
  detection:      '#f59e0b',
  segmentation:   '#10b981',
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
        <button className="btn btn-secondary flex items-center gap-1" onClick={fetchModels}>
          <RefreshCw size={14} /> Refresh
        </button>
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
