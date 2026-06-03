import { useState, useEffect, useRef, useCallback } from 'react'
import {
  FlaskConical, Upload, Play, X, Loader2, ImageIcon,
  Zap, CheckCircle2, AlertCircle, ChevronDown,
} from 'lucide-react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface Model {
  id: string
  name: string
  training_type: 'classification' | 'detection' | 'segmentation'
  model: string
  engine: string
}

interface ClsResult {
  type: 'classification'
  predictions: Array<{ label: string; confidence: number }>
  top_label: string
  top_confidence: number
  inference_time_ms: number
  model_name: string
}

interface Detection {
  label: string
  confidence: number
  bbox: [number, number, number, number]
  color: string
}

interface DetResult {
  type: 'detection'
  detections: Detection[]
  count: number
  inference_time_ms: number
  model_name: string
}

interface SegMask {
  label: string
  confidence: number
  area_pct: number
  color: string
  polygon: Array<[number, number]>
}

interface SegResult {
  type: 'segmentation'
  masks: SegMask[]
  inference_time_ms: number
  model_name: string
}

type InferenceResult = ClsResult | DetResult | SegResult

// ─── Colours per training type ────────────────────────────────────────────────

const TYPE_COLOR: Record<string, string> = {
  classification: '#6366f1',
  detection:      '#f59e0b',
  segmentation:   '#10b981',
}

const TYPE_LABEL: Record<string, string> = {
  classification: 'Classification',
  detection:      'Detection',
  segmentation:   'Segmentation',
}

// ─── Confidence bar colours ───────────────────────────────────────────────────

function barColor(conf: number) {
  if (conf >= 0.7) return '#10b981'
  if (conf >= 0.4) return '#f59e0b'
  return '#ef4444'
}

// ─── Canvas drawing helpers ──────────────────────────────────────────────────

function drawDetections(canvas: HTMLCanvasElement, img: HTMLImageElement, detections: Detection[], threshold: number) {
  const ctx = canvas.getContext('2d')!
  canvas.width  = img.naturalWidth
  canvas.height = img.naturalHeight
  ctx.drawImage(img, 0, 0)

  const W = canvas.width
  const H = canvas.height
  const scale = Math.max(W, H)

  for (const det of detections) {
    if (det.confidence < threshold) continue
    const [x1, y1, x2, y2] = det.bbox
    const px = x1 * W, py = y1 * H
    const pw = (x2 - x1) * W, ph = (y2 - y1) * H

    // Box
    ctx.strokeStyle = det.color
    ctx.lineWidth = Math.max(2, scale * 0.003)
    ctx.strokeRect(px, py, pw, ph)

    // Label background
    const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`
    const fs = Math.max(11, scale * 0.015)
    ctx.font = `bold ${fs}px Inter, sans-serif`
    const tw = ctx.measureText(label).width
    const pad = fs * 0.4
    ctx.fillStyle = det.color
    ctx.fillRect(px - ctx.lineWidth / 2, py - fs - pad * 2, tw + pad * 2, fs + pad * 2)

    // Label text
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, px + pad - ctx.lineWidth / 2, py - pad)
  }
}

function drawSegmentation(canvas: HTMLCanvasElement, img: HTMLImageElement, masks: SegMask[], threshold: number) {
  const ctx = canvas.getContext('2d')!
  canvas.width  = img.naturalWidth
  canvas.height = img.naturalHeight
  ctx.drawImage(img, 0, 0)

  const W = canvas.width
  const H = canvas.height

  for (const mask of masks) {
    if (mask.confidence < threshold) continue
    // Parse hex color + add alpha
    const hex = mask.color.replace('#', '')
    const r = parseInt(hex.slice(0,2), 16)
    const g = parseInt(hex.slice(2,4), 16)
    const b = parseInt(hex.slice(4,6), 16)

    ctx.beginPath()
    mask.polygon.forEach(([px, py], i) => {
      const x = px * W, y = py * H
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.closePath()
    ctx.fillStyle = `rgba(${r},${g},${b},0.35)`
    ctx.fill()
    ctx.strokeStyle = `rgba(${r},${g},${b},0.85)`
    ctx.lineWidth = Math.max(1.5, W * 0.002)
    ctx.stroke()

    // Label near centroid
    const cx = mask.polygon.reduce((s, p) => s + p[0], 0) / mask.polygon.length * W
    const cy = mask.polygon.reduce((s, p) => s + p[1], 0) / mask.polygon.length * H
    const label = `${mask.label} ${(mask.confidence * 100).toFixed(0)}%`
    const fs = Math.max(11, W * 0.015)
    ctx.font = `bold ${fs}px Inter, sans-serif`
    ctx.fillStyle = `rgba(${r},${g},${b},1)`
    ctx.fillText(label, cx, cy)
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Playground() {
  const [models, setModels] = useState<Model[]>([])
  const [modelsLoading, setModelsLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [modelOpen, setModelOpen] = useState(false)

  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const [threshold, setThreshold] = useState(0.3)

  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef<HTMLImageElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Fetch completed models
  useEffect(() => {
    setModelsLoading(true)
    fetch('/api/jobs?view=models')
      .then(r => r.json())
      .then(d => {
        const ms: Model[] = (d.jobs ?? []).map((j: any) => ({
          id:            j.id,
          name:          j.name,
          training_type: j.training_type,
          model:         j.model,
          engine:        j.engine,
        }))
        setModels(ms)
        if (ms.length > 0) setSelectedModel(ms[0])
      })
      .catch(() => {})
      .finally(() => setModelsLoading(false))
  }, [])

  // Redraw canvas when result or threshold changes
  useEffect(() => {
    if (!result || !canvasRef.current || !imgRef.current) return
    const img = imgRef.current
    const draw = () => {
      if (result.type === 'detection') drawDetections(canvasRef.current!, img, result.detections, threshold)
      if (result.type === 'segmentation') drawSegmentation(canvasRef.current!, img, result.masks, threshold)
    }
    if (img.complete) draw()
    else img.onload = draw
  }, [result, threshold])

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return
    setImageFile(file)
    const url = URL.createObjectURL(file)
    setImageUrl(url)
    setResult(null)
    setError(null)
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const runInference = async () => {
    if (!selectedModel || !imageFile) return
    setRunning(true)
    setResult(null)
    setError(null)
    try {
      const form = new FormData()
      form.append('model_id', selectedModel.id)
      form.append('image', imageFile)
      const res = await fetch('/api/inference', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? res.statusText)
      }
      const data: InferenceResult = await res.json()
      setResult(data)
    } catch (e: any) {
      setError(e.message ?? 'Inference failed')
    } finally {
      setRunning(false)
    }
  }

  const clearImage = () => {
    setImageFile(null)
    setImageUrl(null)
    setResult(null)
    setError(null)
  }

  const canRun = !!selectedModel && !!imageFile && !running

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <FlaskConical size={20} color="#fff" />
          </div>
          <h1 style={{ fontSize: 24, fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
            Playground
          </h1>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: 14, margin: 0 }}>
          Test inference on your trained models · upload an image and run predictions
        </p>
      </div>

      {/* Config bar */}
      <div style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-default)',
        borderRadius: 12,
        padding: '16px 20px',
        marginBottom: 24,
        display: 'flex',
        gap: 24,
        alignItems: 'center',
        flexWrap: 'wrap',
      }}>
        {/* Model selector */}
        <div style={{ flex: '1 1 300px', minWidth: 220 }}>
          <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>
            Model
          </label>
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setModelOpen(o => !o)}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', gap: 10,
                padding: '9px 12px', borderRadius: 8,
                border: '1px solid var(--border-default)',
                background: 'var(--bg-elevated)',
                color: 'var(--text-primary)',
                cursor: 'pointer', fontSize: 13, textAlign: 'left',
              }}
            >
              {modelsLoading ? (
                <><Loader2 size={14} className="animate-spin" /><span style={{ color: 'var(--text-muted)' }}>Loading…</span></>
              ) : selectedModel ? (
                <>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                    background: TYPE_COLOR[selectedModel.training_type],
                  }} />
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {selectedModel.name}
                  </span>
                  <span style={{
                    fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 4,
                    background: TYPE_COLOR[selectedModel.training_type] + '22',
                    color: TYPE_COLOR[selectedModel.training_type],
                    flexShrink: 0,
                  }}>
                    {TYPE_LABEL[selectedModel.training_type]}
                  </span>
                </>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>No completed models</span>
              )}
              <ChevronDown size={14} style={{ color: 'var(--text-muted)', flexShrink: 0, marginLeft: 'auto' }} />
            </button>

            {modelOpen && models.length > 0 && (
              <div style={{
                position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border-default)',
                borderRadius: 8, zIndex: 50,
                boxShadow: '0 8px 24px rgba(0,0,0,0.18)',
                maxHeight: 260, overflowY: 'auto',
              }}>
                {models.map(m => (
                  <button
                    key={m.id}
                    onClick={() => { setSelectedModel(m); setModelOpen(false); setResult(null) }}
                    style={{
                      width: '100%', display: 'flex', alignItems: 'center', gap: 10,
                      padding: '9px 14px', border: 'none', cursor: 'pointer',
                      background: selectedModel?.id === m.id ? 'var(--bg-surface)' : 'transparent',
                      color: 'var(--text-primary)', fontSize: 13, textAlign: 'left',
                    }}
                  >
                    <span style={{
                      width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                      background: TYPE_COLOR[m.training_type],
                    }} />
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {m.name}
                    </span>
                    <span style={{
                      fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 4,
                      background: TYPE_COLOR[m.training_type] + '22',
                      color: TYPE_COLOR[m.training_type], flexShrink: 0,
                    }}>
                      {TYPE_LABEL[m.training_type]}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Confidence threshold */}
        <div style={{ flex: '0 1 260px' }}>
          <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <span>Confidence Threshold</span>
            <span style={{ color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{(threshold * 100).toFixed(0)}%</span>
          </label>
          <input
            type="range" min={0} max={1} step={0.05} value={threshold}
            onChange={e => setThreshold(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: '#6366f1' }}
          />
        </div>

        {/* Run button */}
        <div style={{ marginLeft: 'auto', flexShrink: 0 }}>
          <button
            className="btn"
            onClick={runInference}
            disabled={!canRun}
            style={{
              background: canRun
                ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                : 'var(--bg-elevated)',
              color: canRun ? '#fff' : 'var(--text-muted)',
              border: 'none',
              padding: '10px 22px',
              borderRadius: 8,
              fontWeight: 700,
              fontSize: 14,
              cursor: canRun ? 'pointer' : 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              transition: 'opacity 0.15s',
              opacity: canRun ? 1 : 0.55,
            }}
          >
            {running
              ? <><Loader2 size={16} className="animate-spin" />Running…</>
              : <><Play size={15} fill="currentColor" />Run Inference</>
            }
          </button>
        </div>
      </div>

      {/* Main 2-col */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>

        {/* Left: Upload */}
        <div>
          <div
            onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => !imageFile && fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${isDragging ? '#6366f1' : imageFile ? 'var(--border-default)' : 'var(--border-subtle)'}`,
              borderRadius: 12,
              background: isDragging ? 'rgba(99,102,241,0.06)' : 'var(--bg-surface)',
              minHeight: imageFile ? 'auto' : 280,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: imageFile ? 'flex-start' : 'center',
              cursor: imageFile ? 'default' : 'pointer',
              overflow: 'hidden',
              transition: 'border-color 0.15s, background 0.15s',
              position: 'relative',
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={e => { if (e.target.files?.[0]) handleFile(e.target.files[0]) }}
            />

            {!imageFile ? (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <div style={{
                  width: 60, height: 60, borderRadius: 14,
                  background: 'var(--bg-elevated)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  margin: '0 auto 16px',
                }}>
                  <Upload size={26} color="var(--text-muted)" />
                </div>
                <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 6px' }}>
                  Drop image here or click to upload
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0 }}>
                  PNG, JPG, TIFF, DICOM preview — max 50 MB
                </p>
              </div>
            ) : (
              <>
                {/* Image preview header */}
                <div style={{
                  width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '10px 14px', borderBottom: '1px solid var(--border-subtle)',
                  background: 'var(--bg-elevated)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                    <ImageIcon size={14} color="var(--text-muted)" />
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {imageFile.name}
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', flexShrink: 0 }}>
                      {(imageFile.size / 1024).toFixed(0)} KB
                    </span>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); clearImage() }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 2 }}
                  >
                    <X size={14} />
                  </button>
                </div>

                {/* Image or Canvas */}
                <div style={{ width: '100%', position: 'relative' }}>
                  {result?.type === 'classification' || !result ? (
                    <img
                      ref={imgRef}
                      src={imageUrl!}
                      alt="uploaded"
                      style={{ width: '100%', display: 'block', maxHeight: 520, objectFit: 'contain', background: '#000' }}
                    />
                  ) : (
                    <>
                      <img
                        ref={imgRef}
                        src={imageUrl!}
                        alt="uploaded"
                        style={{ display: 'none' }}
                        onLoad={() => {
                          if (!canvasRef.current || !imgRef.current) return
                          if (result.type === 'detection') drawDetections(canvasRef.current, imgRef.current, result.detections, threshold)
                          if (result.type === 'segmentation') drawSegmentation(canvasRef.current, imgRef.current, result.masks, threshold)
                        }}
                      />
                      <canvas
                        ref={canvasRef}
                        style={{ width: '100%', display: 'block', maxHeight: 520, objectFit: 'contain', background: '#000' }}
                      />
                    </>
                  )}

                  {running && (
                    <div style={{
                      position: 'absolute', inset: 0,
                      background: 'rgba(0,0,0,0.5)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      flexDirection: 'column', gap: 10,
                    }}>
                      <Loader2 size={32} color="#fff" className="animate-spin" />
                      <span style={{ color: '#fff', fontSize: 13, fontWeight: 600 }}>Running inference…</span>
                    </div>
                  )}
                </div>

                {/* Re-upload hint */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    width: '100%', padding: '8px 0', textAlign: 'center',
                    color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer',
                    borderTop: '1px solid var(--border-subtle)',
                    background: 'var(--bg-elevated)',
                  }}
                >
                  Click to replace image
                </div>
              </>
            )}
          </div>
        </div>

        {/* Right: Results */}
        <div>
          {!result && !error && (
            <div style={{
              border: '1px solid var(--border-subtle)',
              borderRadius: 12,
              background: 'var(--bg-surface)',
              minHeight: 280,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 40,
              color: 'var(--text-muted)',
              textAlign: 'center',
            }}>
              <Zap size={36} style={{ opacity: 0.2, marginBottom: 14 }} />
              <p style={{ fontSize: 14, fontWeight: 600, margin: '0 0 6px', color: 'var(--text-secondary)' }}>
                No results yet
              </p>
              <p style={{ fontSize: 12, margin: 0 }}>
                {!selectedModel
                  ? 'Select a model to get started'
                  : !imageFile
                  ? 'Upload an image and click Run Inference'
                  : 'Click Run Inference to see predictions'}
              </p>
            </div>
          )}

          {error && (
            <div style={{
              border: '1px solid #ef444440',
              borderRadius: 12,
              background: '#ef444408',
              padding: 24,
              display: 'flex',
              gap: 12,
              alignItems: 'flex-start',
            }}>
              <AlertCircle size={18} color="#ef4444" style={{ flexShrink: 0, marginTop: 1 }} />
              <div>
                <p style={{ margin: '0 0 4px', fontWeight: 600, fontSize: 14, color: '#ef4444' }}>Inference failed</p>
                <p style={{ margin: 0, fontSize: 13, color: 'var(--text-muted)' }}>{error}</p>
              </div>
            </div>
          )}

          {result && (
            <div style={{
              border: '1px solid var(--border-default)',
              borderRadius: 12,
              background: 'var(--bg-surface)',
              overflow: 'hidden',
            }}>
              {/* Result header */}
              <div style={{
                padding: '14px 18px',
                borderBottom: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: 'var(--bg-elevated)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <CheckCircle2 size={16} color="#10b981" />
                  <span style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)' }}>
                    Inference Complete
                  </span>
                  <span style={{
                    fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 5,
                    background: TYPE_COLOR[result.type] + '20',
                    color: TYPE_COLOR[result.type],
                    textTransform: 'capitalize',
                  }}>
                    {result.type}
                  </span>
                </div>
                <span style={{ fontSize: 11, color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                  {result.inference_time_ms} ms · {result.model_name}
                </span>
              </div>

              {/* Classification results */}
              {result.type === 'classification' && (() => {
                const visible = result.predictions.filter(p => p.confidence >= threshold)
                const hidden  = result.predictions.filter(p => p.confidence < threshold)
                return (
                  <div style={{ padding: '18px 18px 14px' }}>
                    {/* Top prediction hero */}
                    <div style={{
                      padding: '14px 16px',
                      borderRadius: 10,
                      background: '#10b98110',
                      border: '1px solid #10b98130',
                      marginBottom: 18,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}>
                      <div>
                        <div style={{ fontSize: 11, color: '#10b981', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
                          Top Prediction
                        </div>
                        <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--text-primary)' }}>
                          {result.top_label}
                        </div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: 32, fontWeight: 800, color: '#10b981', fontVariantNumeric: 'tabular-nums' }}>
                          {(result.top_confidence * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>confidence</div>
                      </div>
                    </div>

                    {/* Filtered predictions */}
                    {visible.length === 0 ? (
                      <div style={{ textAlign: 'center', padding: '16px 0', color: 'var(--text-muted)', fontSize: 13 }}>
                        No predictions above {(threshold * 100).toFixed(0)}% threshold
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                        {visible.map((p, i) => (
                          <div key={p.label}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, alignItems: 'center' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                {i === 0 && <span style={{ fontSize: 10, color: '#f59e0b' }}>★</span>}
                                <span style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: i === 0 ? 700 : 400 }}>{p.label}</span>
                              </div>
                              <span style={{ fontSize: 12, color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                                {(p.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-elevated)', overflow: 'hidden' }}>
                              <div style={{
                                height: '100%',
                                width: `${p.confidence * 100}%`,
                                background: i === 0 ? '#10b981' : barColor(p.confidence),
                                borderRadius: 3,
                                transition: 'width 0.6s cubic-bezier(0.4,0,0.2,1)',
                              }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {hidden.length > 0 && (
                      <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
                        {hidden.length} class{hidden.length > 1 ? 'es' : ''} hidden below {(threshold * 100).toFixed(0)}% threshold
                      </div>
                    )}
                  </div>
                )
              })()}

              {/* Detection results */}
              {result.type === 'detection' && (
                <div style={{ padding: '14px 18px' }}>
                  <div style={{ marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                      <strong style={{ color: 'var(--text-primary)' }}>{result.detections.filter(d => d.confidence >= threshold).length}</strong> detection{result.detections.filter(d => d.confidence >= threshold).length !== 1 ? 's' : ''} above threshold
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {result.detections.map((d, i) => {
                      const visible = d.confidence >= threshold
                      return (
                        <div key={i} style={{
                          display: 'flex', alignItems: 'center', gap: 10,
                          padding: '8px 12px', borderRadius: 8,
                          background: visible ? 'var(--bg-elevated)' : 'transparent',
                          border: `1px solid ${visible ? d.color + '40' : 'var(--border-subtle)'}`,
                          opacity: visible ? 1 : 0.4,
                        }}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, background: d.color, flexShrink: 0 }} />
                          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>
                            {d.label}
                          </span>
                          <span style={{ fontSize: 12, color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                            [{d.bbox.map(v => (v * 100).toFixed(0) + '%').join(', ')}]
                          </span>
                          <span style={{
                            fontSize: 12, fontWeight: 700, padding: '2px 8px', borderRadius: 6,
                            background: d.color + '22', color: d.color, fontVariantNumeric: 'tabular-nums',
                          }}>
                            {(d.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Segmentation results */}
              {result.type === 'segmentation' && (
                <div style={{ padding: '14px 18px' }}>
                  <div style={{ marginBottom: 14 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                      <strong style={{ color: 'var(--text-primary)' }}>{result.masks.filter(m => m.confidence >= threshold).length}</strong> region{result.masks.filter(m => m.confidence >= threshold).length !== 1 ? 's' : ''} segmented
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {result.masks.map((m, i) => {
                      const visible = m.confidence >= threshold
                      return (
                        <div key={i} style={{
                          display: 'flex', alignItems: 'center', gap: 10,
                          padding: '8px 12px', borderRadius: 8,
                          background: visible ? 'var(--bg-elevated)' : 'transparent',
                          border: `1px solid ${visible ? m.color + '40' : 'var(--border-subtle)'}`,
                          opacity: visible ? 1 : 0.4,
                        }}>
                          <span style={{
                            width: 16, height: 16, borderRadius: 4,
                            background: m.color + '66',
                            border: `2px solid ${m.color}`,
                            flexShrink: 0,
                          }} />
                          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>
                            {m.label.replace('_', ' ')}
                          </span>
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            area {m.area_pct.toFixed(1)}%
                          </span>
                          <span style={{
                            fontSize: 12, fontWeight: 700, padding: '2px 8px', borderRadius: 6,
                            background: m.color + '22', color: m.color, fontVariantNumeric: 'tabular-nums',
                          }}>
                            {(m.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Metadata footer */}
              <div style={{
                padding: '10px 18px',
                borderTop: '1px solid var(--border-subtle)',
                background: 'var(--bg-elevated)',
                display: 'flex', gap: 20, flexWrap: 'wrap',
              }}>
                {[
                  ['Model',      result.model_name],
                  ['Type',       TYPE_LABEL[result.type]],
                  ['Image',      imageFile?.name ?? '—'],
                  ['Latency',    `${result.inference_time_ms} ms`],
                ].map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>{k}</div>
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 500 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
