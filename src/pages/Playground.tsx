import { useState, useEffect, useRef, useCallback } from 'react'
import {
  FlaskConical, Upload, Play, X, Loader2, ImageIcon,
  Zap, CheckCircle2, AlertCircle, ChevronDown, MessageSquare, Eye,
  Send, History, Trash2, Clock, Copy, Check,
} from 'lucide-react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface Model {
  id: string
  name: string
  training_type: string
  model: string
  engine: string
  modal_url?: string
  ray_serve_url?: string
  inference_provider?: string
}

interface TextResult {
  type: 'text'
  response: string
  tokens_generated: number
  tokens_per_second: number
  inference_time_ms: number
  model_name: string
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

// ─── History (server-side via /api/inference/history) ────────────────────────

interface HistoryEntry {
  id: string
  created_at: number
  mode: 'image' | 'text' | 'vl'
  model_id: string
  model_name: string
  model_type: string
  // image
  image_name?: string
  thumbnail_key?: string
  result?: any
  // text / vl
  system_prompt?: string
  user_prompt?: string
  text_result?: TextResult
  inference_time_ms?: number
}

function fmtTime(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

async function makeThumbnail(file: File): Promise<string> {
  return new Promise(resolve => {
    const img = new Image()
    const url = URL.createObjectURL(file)
    img.onload = () => {
      const maxW = 120, maxH = 80
      const scale = Math.min(maxW / img.width, maxH / img.height, 1)
      const c = document.createElement('canvas')
      c.width = Math.round(img.width * scale)
      c.height = Math.round(img.height * scale)
      c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height)
      URL.revokeObjectURL(url)
      resolve(c.toDataURL('image/jpeg', 0.7))
    }
    img.onerror = () => { URL.revokeObjectURL(url); resolve('') }
    img.src = url
  })
}

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      // Strip "data:mime;base64," prefix
      const b64 = result.includes(',') ? result.split(',', 2)[1] : result
      resolve(b64)
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

// ─── Colours per training type ────────────────────────────────────────────────

const TYPE_COLOR: Record<string, string> = {
  classification: '#6366f1',
  detection:      '#f59e0b',
  segmentation:   '#10b981',
  'llm-text':     '#8b5cf6',
  'vlm-finetune': '#3b82f6',
}

const TYPE_LABEL: Record<string, string> = {
  classification: 'Classification',
  detection:      'Detection',
  segmentation:   'Segmentation',
  'llm-text':     'LLM',
  'vlm-finetune': 'VLM',
}

// ─── Confidence bar colours ───────────────────────────────────────────────────

function barColor(conf: number) {
  if (conf >= 0.7) return '#10b981'
  if (conf >= 0.4) return '#f59e0b'
  return '#ef4444'
}

// ─── Canvas drawing helpers ──────────────────────────────────────────────────

function drawDetections(canvas: HTMLCanvasElement, img: HTMLImageElement, detections: Detection[], threshold: number) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  canvas.width  = img.naturalWidth
  canvas.height = img.naturalHeight
  ctx.drawImage(img, 0, 0)

  const W = canvas.width
  const H = canvas.height
  if (!W || !H) return
  const scale = Math.max(W, H)
  if (!detections || detections.length === 0) return

  const palette = ['#8b5cf6', '#10b981', '#ef4444', '#f59e0b', '#3b82f6', '#ec4899', '#14b8a6']

  for (let _i = 0; _i < detections.length; _i++) {
    const det = detections[_i]
    const confidence = det.confidence ?? 0
    if (confidence < threshold) continue
    if (!Array.isArray(det.bbox) || det.bbox.length < 4) continue
    const [x1, y1, x2, y2] = det.bbox
    const px = x1 * W, py = y1 * H
    const pw = (x2 - x1) * W, ph = (y2 - y1) * H
    const color = (det.color && typeof det.color === 'string') ? det.color : palette[_i % palette.length]

    // Box
    ctx.strokeStyle = color
    ctx.lineWidth = Math.max(2, scale * 0.003)
    ctx.strokeRect(px, py, pw, ph)

    // Label background
    const label = `${det.label ?? 'object'} ${(confidence * 100).toFixed(0)}%`
    const fs = Math.max(11, scale * 0.015)
    ctx.font = `bold ${fs}px Inter, sans-serif`
    const tw = ctx.measureText(label).width
    const pad = fs * 0.4
    ctx.fillStyle = color
    ctx.fillRect(px - ctx.lineWidth / 2, py - fs - pad * 2, tw + pad * 2, fs + pad * 2)

    // Label text
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, px + pad - ctx.lineWidth / 2, py - pad)
  }
}

function drawSegmentation(canvas: HTMLCanvasElement, img: HTMLImageElement, masks: SegMask[], threshold: number) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  canvas.width  = img.naturalWidth
  canvas.height = img.naturalHeight
  ctx.drawImage(img, 0, 0)

  const W = canvas.width
  const H = canvas.height
  if (!masks || masks.length === 0) return

  // Fixed color palette — assign by index to ensure consistency
  const palette = ['#8b5cf6', '#10b981', '#ef4444', '#f59e0b', '#3b82f6', '#ec4899', '#14b8a6']

  for (let _i = 0; _i < masks.length; _i++) {
    const mask = masks[_i]
    const confidence = mask.confidence ?? 0
    if (confidence < threshold) continue
    // Defensive color parsing
    const color = (mask.color && typeof mask.color === 'string') ? mask.color : palette[_i % palette.length]
    const hex = color.startsWith('#') ? color.slice(1) : color
    if (hex.length < 6) continue
    const r = parseInt(hex.slice(0, 2), 16)
    const g = parseInt(hex.slice(2, 4), 16)
    const b = parseInt(hex.slice(4, 6), 16)
    if (isNaN(r) || isNaN(g) || isNaN(b)) continue
    // Defensive polygon access
    if (!Array.isArray(mask.polygon) || mask.polygon.length < 3) continue

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
  const [playMode, setPlayMode] = useState<'image' | 'text' | 'vl'>('image')

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

  // Cluster reachability — used to flag the "Ray" badge as offline
  // when the underlying cluster isn't responding. We no longer gate
  // the Modal badge on cluster state: per-model Modal Web Functions
  // are independent of the Modal Ray cluster.
  // null = unknown / not yet probed.
  const [rayClusterOnline, setRayClusterOnline] = useState<boolean | null>(null)

  // Text / VL inference state
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful industrial AI assistant.')
  const [userPrompt, setUserPrompt] = useState('')
  const [textResult, setTextResult] = useState<TextResult | null>(null)
  const [textError, setTextError] = useState<string | null>(null)
  const [copiedResponse, setCopiedResponse] = useState(false)
  const [copiedError, setCopiedError] = useState(false)
  const [vlImageFile, setVlImageFile] = useState<File | null>(null)
  const [vlImageUrl, setVlImageUrl] = useState<string | null>(null)
  const [vlIsDragging, setVlIsDragging] = useState(false)
  const vlFileInputRef = useRef<HTMLInputElement>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef<HTMLImageElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Pre-loaded image used ONLY for canvas drawing. Kept separate from the
  // displayed <img> because toggling that element's display between block/none
  // (when the result type changes) can cause the browser to re-decode the
  // image, briefly setting naturalWidth=0 — which races with the canvas draw.
  const canvasImgRef     = useRef<HTMLImageElement | null>(null)
  const [canvasImgReady, setCanvasImgReady] = useState(false)

  // History
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [historyThumbs, setHistoryThumbs] = useState<Record<string, string>>({})
  const [historyOpen, setHistoryOpen] = useState(true)

  const fetchHistory = useCallback(async () => {
    try {
      const r = await fetch('/api/inference/history?limit=50')
      if (r.ok) {
        const d = await r.json()
        const items = d.history || []
        setHistory(items)
        // Pre-load thumbnails as blob URLs (works without auth cookies)
        items.forEach((it: HistoryEntry) => {
          if (it.thumbnail_key && !historyThumbs[it.id]) {
            fetch(`/api/inference/thumbnail/${it.id}`)
              .then(resp => resp.ok ? resp.blob() : null)
              .then(blob => {
                if (blob) {
                  const url = URL.createObjectURL(blob)
                  setHistoryThumbs(prev => ({ ...prev, [it.id]: url }))
                }
              })
              .catch(() => {})
          }
        })
      }
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { fetchHistory() }, [fetchHistory])

  const deleteHistoryEntry = async (id: string) => {
    setHistory(h => h.filter(e => e.id !== id))
    try { await fetch(`/api/inference/history/${id}`, { method: 'DELETE' }) } catch {}
  }
  const clearHistory = async () => {
    setHistory([])
    try { await fetch('/api/inference/history', { method: 'DELETE' }) } catch {}
  }

  const pushHistory = async (params: {
    mode: 'image' | 'text' | 'vl'
    modelId: string
    modelName: string
    modelType: string
    imageName?: string
    imageFile?: File
    vlImageFile?: File
    userPrompt?: string
    systemPrompt?: string
    result: any
    inferenceTimeMs?: number
  }) => {
    try {
      const id = crypto.randomUUID()
      let thumb = ''
      let imageB64 = ''
      if (params.mode === 'image' && params.imageFile) {
        thumb = await makeThumbnail(params.imageFile)
        imageB64 = await fileToBase64(params.imageFile)
      } else if (params.mode === 'vl' && params.vlImageFile) {
        thumb = await makeThumbnail(params.vlImageFile)
        imageB64 = await fileToBase64(params.vlImageFile)
      }
      const res = await fetch('/api/inference/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id,
          mode: params.mode,
          model_id: params.modelId,
          model_name: params.modelName,
          model_type: params.modelType,
          image_name: params.imageName,
          user_prompt: params.userPrompt,
          system_prompt: params.systemPrompt,
          result: params.result,
          inference_time_ms: params.inferenceTimeMs,
          thumbnail_base64: thumb,
          image_base64: imageB64,
        }),
      })
      if (res.ok) fetchHistory()
    } catch { /* ignore */ }
  }

  // Fetch completed models
  useEffect(() => {
    setModelsLoading(true)
    Promise.all([
      fetch('/api/jobs?view=models').then(r => r.json()),
      fetch('/api/settings/inference/ray-status')
        .then(r => r.ok ? r.json() : { online: false })
        .catch(() => ({ online: false })),
    ]).then(([d, rayStatus]) => {
      const ms: Model[] = (d.jobs ?? []).map((j: any) => ({
        id:            j.id,
        name:          j.name,
        training_type: j.training_type,
        model:         j.model,
        engine:        j.engine,
        modal_url:          j.modal_url ?? '',
        ray_serve_url:      j.ray_serve_url ?? '',
        inference_provider: j.inference_provider ?? '',
      }))
      setModels(ms)
      setRayClusterOnline(!!rayStatus.online)
      // Default-select from the *filtered* visible list, not the raw models.
      // Otherwise the closed button shows a model that the dropdown hides.
      // Same URL-required check as the render-time filter below.
      const providerOnline = (m: typeof ms[number]): boolean => {
        if (m.inference_provider === 'ray')
          return !!rayStatus.online && !!m.ray_serve_url
        if (m.inference_provider === 'modal')
          return !!m.modal_url   // Modal Web Function — independent of Modal Ray cluster
        return false
      }
      const typeOk = playMode === 'text'
        ? (m: typeof ms[number]) => m.training_type === 'llm-text'
        : playMode === 'vl'
        ? (m: typeof ms[number]) => m.training_type === 'vlm-finetune'
        : (m: typeof ms[number]) => m.training_type !== 'llm-text' && m.training_type !== 'vlm-finetune'
      const first = ms.find(m => typeOk(m) && providerOnline(m))
      setSelectedModel(first ?? null)
    })
    .catch(() => {})
    .finally(() => setModelsLoading(false))
  }, [])

  const isLlmModel = (m: Model) => m.training_type === 'llm-text'
  const isVlmModel = (m: Model) => m.training_type === 'vlm-finetune'
  const isImageModel = (m: Model) => !isLlmModel(m) && !isVlmModel(m)

  // Only show models whose inference provider is actually reachable. The
  // DB column `inference_provider` is sticky — it can say 'ray' even after
  // the Ray cluster goes down, or 'modal' without a running cluster — so we
  // gate visibility on the live cluster-status probes. Sim / no-provider
  // models are also hidden: there's nothing to call, and showing them
  // alongside real providers confuses the dropdown.
  //
  // Additionally require the model's own URL to be non-empty: a model
  // with inference_provider='ray' but no ray_serve_url has the flag
  // set but the actor was never deployed (or was torn down). Show it
  // in the dropdown and inference will silently fail with a confusing
  // network error.
  //
  // For Ray-deployed models the Ray cluster must also be online —
  // the Serve actor lives on the cluster. For Modal-deployed models
  // we don't gate on the Modal Ray cluster state: the model's Web
  // Function is its own Modal app, independent of the Ray cluster
  // (which is the *training* cluster). As long as modal_url is set
  // and Modal.com is reachable (always true), the model works.
  const providerOnline = (m: Model): boolean => {
    if (m.inference_provider === 'ray')
      return rayClusterOnline === true && !!m.ray_serve_url
    if (m.inference_provider === 'modal')
      return !!m.modal_url
    return false  // no provider assigned — not deployable
  }

  // For the model selector, prefer models that are deployable, but don't
  // hide models that are simply not yet deployed. Empty inference_provider
  // should be visible (greyed out if needed) so the user can pick & deploy.
  const providerAvailable = (m: Model): boolean => {
    if (!m.inference_provider) return false  // hide if no provider assigned at all
    return providerOnline(m)
  }

  const visibleModels = (playMode === 'text'
    ? models.filter(isLlmModel)
    : playMode === 'vl'
    ? models.filter(isVlmModel)
    : models.filter(isImageModel)
  ).filter(providerAvailable)

  const runTextInference = async () => {
    if (!selectedModel || !userPrompt.trim()) return
    setRunning(true)
    setTextResult(null)
    setTextError(null)
    try {
      const form = new FormData()
      form.append('model_id', selectedModel.id)
      form.append('prompt', userPrompt)
      form.append('system_prompt', systemPrompt)
      if (playMode === 'vl' && vlImageFile) form.append('image', vlImageFile)
      const res = await fetch('/api/inference', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? res.statusText)
      }
      const data = await res.json()
      setTextResult({ ...data, type: 'text' })
      // push to history
      await pushHistory({
        mode: playMode as 'text' | 'vl',
        modelId: selectedModel.id,
        modelName: selectedModel.name,
        modelType: selectedModel.training_type,
        systemPrompt,
        userPrompt,
        vlImageFile: playMode === 'vl' ? vlImageFile || undefined : undefined,
        result: { ...data, type: 'text' },
        inferenceTimeMs: data.inference_time_ms,
      })
    } catch (e: any) {
      setTextError(e.message ?? 'Inference failed')
    } finally {
      setRunning(false)
    }
  }

  const handleVlFile = (file: File) => {
    if (!file.type.startsWith('image/')) return
    setVlImageFile(file)
    setVlImageUrl(URL.createObjectURL(file))
  }

  // Pre-load imageUrl into a detached Image for canvas drawing whenever it
  // changes. This is independent of the visible <img> and is always ready
  // before the result arrives.
  useEffect(() => {
    if (!imageUrl) {
      canvasImgRef.current = null
      setCanvasImgReady(false)
      return
    }
    const img = new Image()
    const onLoad = () => {
      canvasImgRef.current = img
      setCanvasImgReady(true)
    }
    const onError = () => {
      canvasImgRef.current = null
      setCanvasImgReady(false)
    }
    img.onload = onLoad
    img.onerror = onError
    img.src = imageUrl
    return () => {
      img.onload = onLoad
      img.onerror = onError
    }
  }, [imageUrl])

  // Redraw canvas when result or threshold changes. Uses the pre-loaded
  // canvasImgRef (decoupled from the DOM <img>) so display toggles on the
  // visible <img> can't cause naturalWidth=0 races.
  useEffect(() => {
    if (!result || !canvasRef.current || !canvasImgRef.current || !canvasImgReady) return
    if (result.type !== 'detection' && result.type !== 'segmentation') return
    const canvas = canvasRef.current
    const img    = canvasImgRef.current
    if (result.type === 'detection')   try { drawDetections(canvas, img, result.detections, threshold) } catch (e) { console.error('drawDetections:', e) }
    if (result.type === 'segmentation') {
      // TorchVision returns `regions`; legacy SMP returns `masks`
      const segs = (result as any).regions ?? (result as any).masks
      if (segs) try { drawSegmentation(canvas, img, segs, threshold) } catch (e) { console.error('drawSegmentation:', e) }
    }
  }, [result, threshold, canvasImgReady])

  const handleFile = useCallback((file: File) => {
    // Accept all common medical image formats (PNG, JPG, WebP, TIFF, DICOM, NIfTI)
    const name = (file.name || '').toLowerCase()
    const allowedExts = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp',
                         '.tif', '.tiff', '.dcm', '.dicom', '.nii', '.nii.gz']
    const isAllowedExt = allowedExts.some(ext => name.endsWith(ext))
    if (!file.type.startsWith('image/') && !isAllowedExt) return
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
      // push to history
      await pushHistory({
        mode: 'image',
        modelId: selectedModel.id,
        modelName: selectedModel.name,
        modelType: selectedModel.training_type,
        imageName: imageFile?.name,
        imageFile,
        result: data,
        inferenceTimeMs: (data as any).inference_time_ms,
      })
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
      <div style={{ marginBottom: 20 }}>
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
          Test inference on your trained models
        </p>
      </div>

      {/* Mode tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, background: 'var(--bg-surface)', borderRadius: 10, padding: 4, border: '1px solid var(--border-default)', width: 'fit-content' }}>
        {([['image','Image','#6366f1'] , ['text','Text (LLM)','#8b5cf6'], ['vl','Vision-Language','#3b82f6']] as const).map(([mode, label, color]) => (
          <button key={mode} onClick={() => { setPlayMode(mode); setSelectedModel(null); setResult(null); setTextResult(null) }}
            style={{
              padding: '7px 18px', borderRadius: 7, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: 600, transition: 'all 0.15s',
              background: playMode === mode ? color : 'transparent',
              color: playMode === mode ? '#fff' : 'var(--text-muted)',
            }}
          >
            {label}
          </button>
        ))}
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
        {/* Model selector (filtered by mode) */}
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
                  {selectedModel.inference_provider === 'modal' ? (
                    // Per-model Modal Web Function — independent of the
                    // Modal Ray cluster. Show plain 'Modal' (no
                    // '(offline)' suffix) as long as the URL is set.
                    <span title={selectedModel.modal_url ? `Deployed on Modal — ${selectedModel.modal_url}` : 'No modal_url set'} style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 4, background: '#8b5cf620', color: '#8b5cf6', flexShrink: 0 }}>
                      Modal
                    </span>
                  ) : selectedModel.inference_provider === 'ray' ? (
                    <span title={rayClusterOnline === false ? 'Ray cluster unreachable' : 'Deployed on Ray'} style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 4, background: rayClusterOnline === false ? '#ef444420' : '#f59e0b20', color: rayClusterOnline === false ? '#ef4444' : '#f59e0b', flexShrink: 0 }}>
                      Ray{rayClusterOnline === false ? ' (offline)' : ''}
                    </span>
                  ) : (
                    <span style={{ fontSize: 9, fontWeight: 600, padding: '1px 5px', borderRadius: 4, background: 'var(--bg-base)', color: 'var(--text-muted)', flexShrink: 0 }}>Sim</span>
                  )}
                </>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>No completed models</span>
              )}
              <ChevronDown size={14} style={{ color: 'var(--text-muted)', flexShrink: 0, marginLeft: 'auto' }} />
            </button>

            {modelOpen && visibleModels.length === 0 && (
              <div style={{
                position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
                background: 'var(--bg-elevated)', border: '1px solid var(--border-default)',
                borderRadius: 8, zIndex: 50, boxShadow: '0 8px 24px rgba(0,0,0,0.18)',
                padding: '14px 16px', fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5,
              }}>
                {models.length === 0
                  ? <>No completed models yet — train one first.</>
                  : <>Models are hidden because their inference provider is offline.<br />
                      <span style={{ fontSize: 11 }}>
                        • Ray-deployed models need the on-prem Ray cluster reachable<br />
                        • Modal-deployed models need the Modal Ray cluster running
                      </span></>}
              </div>
            )}

            {modelOpen && visibleModels.length > 0 && (
              <div style={{
                position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border-default)',
                borderRadius: 8, zIndex: 50,
                boxShadow: '0 8px 24px rgba(0,0,0,0.18)',
                maxHeight: 260, overflowY: 'auto',
              }}>
                {visibleModels.map(m => (
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
                    {m.inference_provider === 'modal' ? (
                      // Modal-deployed: the per-model Web Function is its own
                      // Modal app and is independent of the Modal *Ray*
                      // cluster. Showing '(offline)' whenever the Ray
                      // cluster is idle is wrong — the model is reachable
                      // as long as modal_url is set. (The Ray cluster
                      // status is only relevant for the *train* flow.)
                      <span title={m.modal_url ? `Deployed on Modal — ${m.modal_url}` : 'No modal_url set'} style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 4, background: '#8b5cf620', color: '#8b5cf6', flexShrink: 0 }}>
                        Modal
                      </span>
                    ) : m.inference_provider === 'ray' ? (
                      <span title={rayClusterOnline === false ? 'Ray cluster unreachable' : 'Deployed on Ray'} style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 4, background: rayClusterOnline === false ? '#ef444420' : '#f59e0b20', color: rayClusterOnline === false ? '#ef4444' : '#f59e0b', flexShrink: 0 }}>
                        Ray{rayClusterOnline === false ? ' (offline)' : ''}
                      </span>
                    ) : (
                      <span style={{ fontSize: 9, fontWeight: 600, padding: '2px 6px', borderRadius: 4, background: 'var(--bg-base)', color: 'var(--text-muted)', flexShrink: 0 }}>Sim</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Confidence threshold — image only */}
        {playMode === 'image' && <div style={{ flex: '0 1 260px' }}>
          <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <span>Confidence Threshold</span>
            <span style={{ color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{(threshold * 100).toFixed(0)}%</span>
          </label>
          <input
            type="range" min={0} max={1} step={0.05} value={threshold}
            onChange={e => setThreshold(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: '#6366f1' }}
          />
        </div>}

        {/* Run button */}
        <div style={{ marginLeft: 'auto', flexShrink: 0 }}>
          <button
            className="btn"
            onClick={playMode === 'image' ? runInference : runTextInference}
            disabled={playMode === 'image' ? !canRun : (!selectedModel || !userPrompt.trim() || running)}
            style={{
              background: (playMode === 'image' ? canRun : (!selectedModel || !userPrompt.trim() || running) === false)
                ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                : 'var(--bg-elevated)',
              color: (playMode === 'image' ? canRun : (!selectedModel || !userPrompt.trim() || running) === false) ? '#fff' : 'var(--text-muted)',
              border: 'none',
              padding: '10px 22px',
              borderRadius: 8,
              fontWeight: 700,
              fontSize: 14,
              cursor: (playMode === 'image' ? canRun : (!selectedModel || !userPrompt.trim() || running) === false) ? 'pointer' : 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              transition: 'opacity 0.15s',
              opacity: (playMode === 'image' ? canRun : (!selectedModel || !userPrompt.trim() || running) === false) ? 1 : 0.55,
            }}
          >
            {running
              ? <><Loader2 size={16} className="animate-spin" />Running…</>
              : playMode === 'image'
              ? <><Play size={15} fill="currentColor" />Run Inference</>
              : <><Send size={15} />Generate</>
            }
          </button>
        </div>
      </div>

      {/* ── Text (LLM) mode ── */}
      {(playMode === 'text' || playMode === 'vl') && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          {/* Left: Inputs */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* VL image upload */}
            {playMode === 'vl' && (
              <div>
                <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>Image</label>
                <div
                  onDragOver={e => { e.preventDefault(); setVlIsDragging(true) }}
                  onDragLeave={() => setVlIsDragging(false)}
                  onDrop={e => { e.preventDefault(); setVlIsDragging(false); const f = e.dataTransfer.files[0]; if (f) handleVlFile(f) }}
                  onClick={() => vlFileInputRef.current?.click()}
                  style={{
                    border: `2px dashed ${vlIsDragging ? '#3b82f6' : vlImageFile ? 'var(--border-default)' : 'var(--border-subtle)'}`,
                    borderRadius: 10, background: vlIsDragging ? 'rgba(59,130,246,0.06)' : 'var(--bg-surface)',
                    padding: vlImageFile ? 0 : '24px 0', textAlign: 'center', cursor: 'pointer', overflow: 'hidden',
                  }}
                >
                  <input ref={vlFileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={e => { if (e.target.files?.[0]) handleVlFile(e.target.files[0]) }} />
                  {vlImageFile ? (
                    <div style={{ position: 'relative' }}>
                      <img src={vlImageUrl!} alt="vl" style={{ width: '100%', maxHeight: 200, objectFit: 'contain', display: 'block', background: '#000' }} />
                      <button onClick={e => { e.stopPropagation(); setVlImageFile(null); setVlImageUrl(null) }}
                        style={{ position: 'absolute', top: 6, right: 6, background: 'rgba(0,0,0,0.6)', border: 'none', color: '#fff', borderRadius: 6, cursor: 'pointer', padding: '2px 6px' }}>
                        <X size={12} />
                      </button>
                    </div>
                  ) : (
                    <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                      <Eye size={22} style={{ margin: '0 auto 6px', display: 'block', opacity: 0.4 }} />
                      Drop image or click to upload
                    </div>
                  )}
                </div>
              </div>
            )}
            {/* System prompt */}
            <div>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>System Prompt</label>
              <textarea
                value={systemPrompt}
                onChange={e => setSystemPrompt(e.target.value)}
                rows={3}
                style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', fontSize: 13, resize: 'vertical', fontFamily: 'inherit' }}
                placeholder="System instructions…"
              />
            </div>
            {/* User prompt */}
            <div>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>User Prompt</label>
              <textarea
                value={userPrompt}
                onChange={e => setUserPrompt(e.target.value)}
                rows={5}
                style={{ width: '100%', boxSizing: 'border-box', padding: '10px 12px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', fontSize: 13, resize: 'vertical', fontFamily: 'inherit' }}
                placeholder={playMode === 'vl' ? 'Describe what you see in this image…' : 'Ask a question about this product or process…'}
              />
            </div>
          </div>

          {/* Right: Response */}
          <div>
            {!textResult && !textError && (
              <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 12, background: 'var(--bg-surface)', minHeight: 280, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 40, color: 'var(--text-muted)', textAlign: 'center' }}>
                <MessageSquare size={36} style={{ opacity: 0.2, marginBottom: 14 }} />
                <p style={{ fontSize: 14, fontWeight: 600, margin: '0 0 6px', color: 'var(--text-secondary)' }}>No response yet</p>
                <p style={{ fontSize: 12, margin: 0 }}>Enter a prompt and click Generate</p>
              </div>
            )}
            {textError && (
              <div style={{ border: '1px solid #ef444440', borderRadius: 12, background: '#ef444408', padding: 24, display: 'flex', gap: 12 }}>
                <AlertCircle size={18} color="#ef4444" style={{ flexShrink: 0, marginTop: 2 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: 14, color: '#ef4444' }}>Generation failed</p>
                    <button
                      onClick={() => { navigator.clipboard.writeText(textError); setCopiedError(true); setTimeout(() => setCopiedError(false), 2000) }}
                      style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'none', border: '1px solid #ef444440', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', color: copiedError ? '#10b981' : '#ef4444', fontSize: 11, fontFamily: 'inherit', flexShrink: 0 }}
                      title="Copy error message"
                    >
                      {copiedError ? <Check size={12} /> : <Copy size={12} />}
                      {copiedError ? 'Copied' : 'Copy'}
                    </button>
                  </div>
                  <p style={{ margin: 0, fontSize: 13, color: 'var(--text-muted)', wordBreak: 'break-word' }}>{textError}</p>
                </div>
              </div>
            )}
            {textResult && (
              <div style={{ border: '1px solid var(--border-default)', borderRadius: 12, background: 'var(--bg-surface)', overflow: 'hidden' }}>
                <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-elevated)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <CheckCircle2 size={16} color="#10b981" />
                    <span style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)' }}>Response</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{textResult.tokens_generated ?? 0} tokens · {(textResult.tokens_per_second ?? 0).toFixed?.(0) ?? 0} tok/s · {textResult.inference_time_ms ?? 0} ms</span>
                    <button
                      onClick={() => { navigator.clipboard.writeText(textResult.response); setCopiedResponse(true); setTimeout(() => setCopiedResponse(false), 2000) }}
                      style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'none', border: '1px solid var(--border-default)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', color: copiedResponse ? '#10b981' : 'var(--text-muted)', fontSize: 11, fontFamily: 'inherit' }}
                      title="Copy response"
                    >
                      {copiedResponse ? <Check size={12} /> : <Copy size={12} />}
                      {copiedResponse ? 'Copied' : 'Copy'}
                    </button>
                  </div>
                </div>
                <div style={{ padding: '16px 18px', fontSize: 14, color: 'var(--text-primary)', lineHeight: 1.7, whiteSpace: 'pre-wrap', maxHeight: 600, overflowY: 'auto', wordBreak: 'break-word' }}>
                  {textResult.response}
                </div>
                <div style={{ padding: '10px 18px', borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-elevated)', display: 'flex', gap: 20 }}>
                  {[['Model', textResult.model_name], ['Tokens', String(textResult.tokens_generated ?? 0)], ['Speed', `${(textResult.tokens_per_second ?? 0).toFixed?.(0) ?? 0} tok/s`], ['Latency', `${textResult.inference_time_ms ?? 0} ms`]].map(([k, v]) => (
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
      )}

      {/* ── Image mode ── */}
      {playMode === 'image' && <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>

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
                  {/* Always render img element to prevent recreation */}
                  <img
                    ref={imgRef}
                    src={imageUrl!}
                    alt="uploaded"
                    style={{
                      width: '100%',
                      display: (result?.type === 'detection' || result?.type === 'segmentation') ? 'none' : 'block',
                      maxHeight: 520,
                      objectFit: 'contain',
                      background: '#000'
                    }}
                  />
                  {/* Canvas for detection/segmentation */}
                  {(result?.type === 'detection' || result?.type === 'segmentation') && (
                    <canvas
                      ref={canvasRef}
                      style={{ width: '100%', display: 'block', maxHeight: 520, objectFit: 'contain', background: '#000' }}
                    />
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
                      <strong style={{ color: 'var(--text-primary)' }}>{(result.detections ?? []).filter(d => d.confidence >= threshold).length}</strong> detection{(result.detections ?? []).filter(d => d.confidence >= threshold).length !== 1 ? 's' : ''} above threshold
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {(result.detections ?? []).map((d, i) => {
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
                      <strong style={{ color: 'var(--text-primary)' }}>{(result.masks ?? []).filter(m => m.confidence >= threshold).length}</strong> region{(result.masks ?? []).filter(m => m.confidence >= threshold).length !== 1 ? 's' : ''} segmented
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {(result.masks ?? []).map((m, i) => {
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
      </div>}

      {/* ── History ── */}
      <div style={{ marginTop: 36 }}>
        {/* History header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, marginBottom: historyOpen ? 14 : 0,
          cursor: 'pointer', userSelect: 'none',
        }} onClick={() => setHistoryOpen(o => !o)}>
          <History size={16} style={{ color: 'var(--text-muted)' }} />
          <span style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
            History
          </span>
          {history.length > 0 && (
            <span style={{
              fontSize: 11, fontWeight: 700, padding: '1px 7px', borderRadius: 10,
              background: 'var(--primary-dim)', color: 'var(--primary-hover)',
            }}>{history.length}</span>
          )}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
            {history.length > 0 && historyOpen && (
              <button
                onClick={e => { e.stopPropagation(); clearHistory() }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '4px 10px', borderRadius: 6, border: '1px solid #ef444440',
                  background: 'transparent', color: '#ef4444', fontSize: 12, fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                <Trash2 size={12} /> Clear All
              </button>
            )}
            <ChevronDown size={15} style={{ color: 'var(--text-muted)', transform: historyOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
          </div>
        </div>

        {historyOpen && (
          history.length === 0 ? (
            <div style={{
              border: '1px dashed var(--border-subtle)', borderRadius: 12,
              padding: '32px 0', textAlign: 'center', color: 'var(--text-muted)',
            }}>
              <Clock size={28} style={{ opacity: 0.2, display: 'block', margin: '0 auto 8px' }} />
              <span style={{ fontSize: 13 }}>No inference history yet</span>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {history.map(entry => (
                <div
                  key={entry.id}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    padding: '10px 14px', borderRadius: 10,
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border-default)',
                    cursor: 'pointer',
                    transition: 'border-color 0.15s',
                  }}
                  onClick={async () => {
                    // Restore the run into main view
                    const m = entry.mode === 'image' ? 'image' : entry.mode === 'text' ? 'text' : 'vl'
                    setPlayMode(m)
                    const matchedModel = models.find(mm => mm.id === entry.model_id)
                    if (matchedModel) setSelectedModel(matchedModel)
                    if (entry.mode === 'image' && entry.result) { setResult(entry.result); setError(null) }
                    if ((entry.mode === 'text' || entry.mode === 'vl') && entry.result) { setTextResult(entry.result); setTextError(null) }
                    if (entry.user_prompt !== undefined) setUserPrompt(entry.user_prompt)
                    if (entry.system_prompt !== undefined) setSystemPrompt(entry.system_prompt)
                    // Restore image: fetch ORIGINAL image blob and convert to File
                    if ((entry.mode === 'image' || entry.mode === 'vl') && (entry as any).image_key) {
                      try {
                        const url = `/api/inference/image/${entry.id}?v=${entry.created_at}`
                        const resp = await fetch(url)
                        if (resp.ok) {
                          const blob = await resp.blob()
                          const ct = resp.headers.get('content-type') || 'image/jpeg'
                          const extFromCt = ct.split('/')[1]?.split(';')[0] || 'jpg'
                          const filename = entry.image_name || `restored-${entry.id}.${extFromCt}`
                          const file = new File([blob], filename, { type: ct })
                          const objUrl = URL.createObjectURL(blob)
                          if (entry.mode === 'image') {
                            setImageFile(file)
                            setImageUrl(objUrl)
                            setResult(null)  // clear old result so re-run is possible
                          } else if (entry.mode === 'vl') {
                            setVlImageFile(file)
                            setVlImageUrl(objUrl)
                          }
                        }
                      } catch { /* ignore */ }
                    } else if ((entry.mode === 'image' || entry.mode === 'vl') && entry.thumbnail_key) {
                      // Fallback: old history entries have only thumbnail (small + blurry)
                      try {
                        const url = `/api/inference/thumbnail/${entry.id}?v=${entry.created_at}`
                        const resp = await fetch(url)
                        if (resp.ok) {
                          const blob = await resp.blob()
                          const filename = entry.image_name || `restored-${entry.id}.jpg`
                          const file = new File([blob], filename, { type: 'image/jpeg' })
                          const objUrl = URL.createObjectURL(blob)
                          if (entry.mode === 'image') {
                            setImageFile(file)
                            setImageUrl(objUrl)
                            setResult(null)
                          } else if (entry.mode === 'vl') {
                            setVlImageFile(file)
                            setVlImageUrl(objUrl)
                          }
                        }
                      } catch { /* ignore */ }
                    }
                    window.scrollTo({ top: 0, behavior: 'smooth' })
                  }}
                >
                  {/* Thumbnail or icon */}
                  {entry.thumbnail_key && historyThumbs[entry.id] ? (
                    <img
                      src={historyThumbs[entry.id]}
                      alt=""
                      style={{ width: 56, height: 40, objectFit: 'cover', borderRadius: 6, flexShrink: 0, background: '#000' }}
                    />
                  ) : entry.thumbnail_key ? (
                    <div style={{
                      width: 56, height: 40, borderRadius: 6, flexShrink: 0,
                      background: '#000',
                    }} />
                  ) : (
                    <div style={{
                      width: 56, height: 40, borderRadius: 6, flexShrink: 0,
                      background: TYPE_COLOR[entry.model_type] + '18',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      <MessageSquare size={16} style={{ color: TYPE_COLOR[entry.model_type] }} />
                    </div>
                  )}

                  {/* Info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                      <span style={{
                        fontSize: 10, fontWeight: 700, padding: '1px 6px', borderRadius: 4,
                        background: TYPE_COLOR[entry.model_type] + '20',
                        color: TYPE_COLOR[entry.model_type],
                        flexShrink: 0,
                      }}>
                        {TYPE_LABEL[entry.model_type] ?? entry.model_type}
                      </span>
                      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {entry.model_name}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {entry.mode === 'image'
                        ? entry.image_name ?? 'Image inference'
                        : entry.user_prompt
                          ? `"${entry.user_prompt.slice(0, 80)}${entry.user_prompt.length > 80 ? '…' : ''}"`
                          : 'Text inference'
                      }
                    </div>
                  </div>

                  {/* Result summary */}
                  <div style={{ flexShrink: 0, textAlign: 'right', minWidth: 90 }}>
                    {entry.result?.type === 'classification' && (
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#10b981' }}>
                        {entry.result.top_label}
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>
                          {(entry.result.top_confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    )}
                    {entry.result?.type === 'detection' && (
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#f59e0b' }}>
                        {entry.result.count} detections
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>
                          {entry.result.inference_time_ms} ms
                        </div>
                      </div>
                    )}
                    {entry.result?.type === 'segmentation' && (
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#10b981' }}>
                        {entry.result.masks?.length ?? 0} regions
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>
                          {entry.result.inference_time_ms} ms
                        </div>
                      </div>
                    )}
                    {(entry.result?.type === 'llm-text' || entry.result?.type === 'vlm-finetune') && (
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#8b5cf6' }}>
                        {entry.result.tokens_generated ?? 0} tokens
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>
                          {(entry.result.tokens_per_second ?? 0).toFixed?.(0) ?? 0} tok/s
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Timestamp */}
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', flexShrink: 0, minWidth: 90, textAlign: 'right' }}>
                    {fmtTime(entry.created_at)}
                  </div>

                  {/* Delete */}
                  <button
                    onClick={e => { e.stopPropagation(); deleteHistoryEntry(entry.id) }}
                    style={{
                      flexShrink: 0, background: 'transparent', border: 'none', cursor: 'pointer',
                      color: 'var(--text-muted)', padding: 4, borderRadius: 6,
                      display: 'flex', alignItems: 'center',
                    }}
                    title="Remove from history"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  )
}
