import { useState, useEffect, useMemo } from 'react'
import {
  Server, Cloud, Loader2, CheckCircle2, AlertCircle, ExternalLink,
  Database, FileText, Zap, Eye, EyeOff, Sparkles, Wand2,
} from 'lucide-react'
import {
  DEPLOY_CATALOG, MODEL_DATA_TYPES, TRAINING_MATRIX, type DataType,
} from './TrainModel'
import type { TrainOption } from './TrainModel'

type DeployProvider = 'ray' | 'modal' | 'hf'

interface DeployStatus {
  state: 'idle' | 'submitting' | 'deploying' | 'ready' | 'error'
  url?: string
  error?: string
}

const DATA_TYPE_LABELS: Record<DataType, string> = {
  rgb:        'Standard RGB Camera',
  thermal:    'Thermal / IR Camera',
  xray:       'X-Ray / CT Scan',
  microscopy: 'Microscopy / Macro',
  lidar:      'LiDAR / Depth / Point Cloud',
  general:    'General Images',
}

const DATA_TYPE_COLORS: Record<DataType, string> = {
  rgb:        '#6366f1',
  thermal:    '#f59e0b',
  xray:       '#ec4899',
  microscopy: '#14b8a6',
  lidar:      '#0ea5e9',
  general:    '#64748b',
}

// The deploy catalog = two sources:
//   1. DEPLOY_CATALOG — models that CANNOT be fine-tuned via the Unsloth
//      flow (custom vision encoders). Deploy-only.
//   2. Any model in TRAINING_MATRIX['vlm-finetune'] with `zeroShot: true`
//      — SOTA off-the-shelf models that can be deployed as-is even
//      though they are also fine-tunable via the Train menu.
//
// We tag each with `kind` so the UI can show the right badge.
type Kind = 'deploy-only' | 'zero-shot-tunable'

interface DeployEntry {
  opt:       TrainOption
  dataTypes: DataType[]
  kind:      Kind
  source?:   string
  paper?:    string
}

const _ALL_TYPES: DataType[] = ['rgb', 'thermal', 'xray', 'microscopy', 'lidar', 'general']

function buildCatalog(): DeployEntry[] {
  const out: DeployEntry[] = []

  // 1. Deploy-only (RaDialog_v2, Med3DVLM)
  for (const opt of DEPLOY_CATALOG) {
    out.push({
      opt,
      dataTypes: opt.dataTypes ?? _ALL_TYPES,
      kind:      'deploy-only',
      source:    opt.source,
      paper:     opt.paper,
    })
  }

  // 2. Zero-shot fine-tunable VLMs (medr1-3b, hulumed-7b, huatuogpt-v-7b, bimedix2-8b)
  const vlms = TRAINING_MATRIX['vlm-finetune'] ?? []
  for (const opt of vlms) {
    if (!opt.zeroShot) continue
    out.push({
      opt,
      dataTypes: MODEL_DATA_TYPES[opt.value] ?? _ALL_TYPES,
      kind:      'zero-shot-tunable',
    })
  }

  return out
}

export default function DeployModels() {
  const entries: DeployEntry[] = useMemo(() => buildCatalog(), [])
  const [selected, setSelected] = useState<TrainOption | null>(entries[0]?.opt ?? null)
  const [provider, setProvider] = useState<DeployProvider>('ray')
  const [status, setStatus]   = useState<DeployStatus>({ state: 'idle' })
  const [showFullDesc, setShowFullDesc] = useState<Set<string>>(new Set())

  // Reset state when switching model
  useEffect(() => {
    setStatus({ state: 'idle' })
  }, [selected?.value])

  async function handleDeploy() {
    if (!selected) return
    setStatus({ state: 'submitting' })
    try {
      // Stub: hit the backend deploy endpoint. Real wiring depends on the
      // /api/deploy route the user wires up (Ray Serve / Modal).
      const r = await fetch('/api/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id:   selected.value,
          model_name: selected.model,
          provider,
        }),
      })
      if (!r.ok) {
        const t = await r.text()
        setStatus({ state: 'error', error: t || `HTTP ${r.status}` })
        return
      }
      const data = await r.json()
      setStatus({
        state: 'ready',
        url: data.url ?? data.endpoint ?? `${provider}://${selected.value}`,
      })
    } catch (e) {
      setStatus({ state: 'error', error: (e as Error).message })
    }
  }

  const toggleDesc = (val: string) => {
    setShowFullDesc(prev => {
      const s = new Set(prev)
      s.has(val) ? s.delete(val) : s.add(val)
      return s
    })
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
          Zero-Shot / Deploy Models
        </h1>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', maxWidth: 760, lineHeight: 1.6 }}>
          SOTA pre-trained models that you can <strong>deploy as-is</strong> via Ray Serve or Modal.
          Two flavors:
        </p>
        <ul style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.7, marginTop: 8, paddingLeft: 22, maxWidth: 760 }}>
          <li>
            <span style={{
              display: 'inline-block', fontSize: 10, fontWeight: 700,
              padding: '1px 6px', borderRadius: 6,
              background: '#8b5cf620', color: '#8b5cf6', marginRight: 6,
            }}>Deploy-Only</span>
            custom vision encoders (BioViL / DCFormer) — ไม่สามารถ fine-tune ผ่าน Unsloth ได้
          </li>
          <li>
            <span style={{
              display: 'inline-block', fontSize: 10, fontWeight: 700,
              padding: '1px 6px', borderRadius: 6,
              background: '#f59e0b20', color: '#f59e0b', border: '1px solid #f59e0b', marginRight: 6,
            }}>Zero-Shot + Fine-tunable</span>
            SOTA off-the-shelf — deploy ได้เลย, หรือจะ fine-tune ผ่าน Train &gt; LLM &gt; VLM Fine-tune ก็ได้
          </li>
        </ul>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-5">
        {/* Left: model cards */}
        <div className="grid grid-cols-1 gap-4">
          {entries.map(({ opt, dataTypes, kind, source, paper }) => {
            const isSelected = selected?.value === opt.value
            const expanded  = showFullDesc.has(opt.value)
            const isDeployOnly = kind === 'deploy-only'
            return (
              <div
                key={opt.value}
                onClick={() => setSelected(opt)}
                className="card"
                style={{
                  cursor: 'pointer',
                  borderColor: isSelected ? 'var(--primary)' : undefined,
                  boxShadow: isSelected ? '0 0 0 1px var(--primary)' : undefined,
                  transition: 'all .15s',
                }}
              >
                <div className="flex items-start gap-4">
                  <div style={{
                    width: 44, height: 44, borderRadius: 10, flexShrink: 0,
                    background: isDeployOnly
                      ? 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)'
                      : 'linear-gradient(135deg, #f59e0b 0%, #ec4899 100%)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#fff', fontWeight: 700, fontSize: 18,
                  }}>
                    {opt.label.charAt(0)}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
                      <h3 style={{ fontWeight: 600, fontSize: 15, color: 'var(--text-primary)' }}>
                        {opt.label}
                      </h3>
                      <span style={{
                        fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 10,
                        background: '#3b82f620', color: '#3b82f6',
                      }}>VLM</span>
                      {isDeployOnly ? (
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: 3,
                          fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 10,
                          background: '#8b5cf620', color: '#8b5cf6',
                        }}>
                          <Sparkles size={10} /> Deploy-Only
                        </span>
                      ) : (
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: 3,
                          fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 10,
                          background: '#f59e0b20', color: '#f59e0b', border: '1px solid #f59e0b',
                        }}>
                          <Wand2 size={10} /> Zero-Shot + Fine-tunable
                        </span>
                      )}
                      <span style={{
                        fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 10,
                        background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                      }}>{opt.hardware}</span>
                    </div>
                    <div style={{
                      fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
                      marginBottom: 8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {opt.model}
                    </div>
                    <p style={{
                      fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.55, marginBottom: 8,
                      display: '-webkit-box', WebkitLineClamp: expanded ? 99 : 2,
                      WebkitBoxOrient: 'vertical', overflow: 'hidden',
                    }}>
                      {opt.description}
                    </p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                      {dataTypes.map(dt => (
                        <span key={dt} style={{
                          fontSize: 9.5, padding: '2px 7px', borderRadius: 6, fontWeight: 600,
                          background: DATA_TYPE_COLORS[dt] + '20',
                          color: DATA_TYPE_COLORS[dt],
                          border: `1px solid ${DATA_TYPE_COLORS[dt]}40`,
                        }}>
                          {DATA_TYPE_LABELS[dt]}
                        </span>
                      ))}
                      <button
                        onClick={e => { e.stopPropagation(); toggleDesc(opt.value) }}
                        style={{
                          fontSize: 11, padding: '2px 8px', background: 'transparent',
                          border: '1px solid var(--border-default)', borderRadius: 6,
                          color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
                        }}
                      >
                        {expanded ? <><EyeOff size={10} /> ย่อ</> : <><Eye size={10} /> อ่านเพิ่ม</>}
                      </button>
                      {source && (
                        <a
                          href={source}
                          target="_blank"
                          rel="noreferrer"
                          onClick={e => e.stopPropagation()}
                          style={{
                            fontSize: 11, padding: '2px 8px',
                            background: 'var(--bg-elevated)', border: '1px solid var(--border-default)',
                            borderRadius: 6, color: 'var(--text-secondary)',
                            textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 4,
                          }}
                        >
                          <FileText size={10} /> Source <ExternalLink size={9} />
                        </a>
                      )}
                      {paper && (
                        <a
                          href={paper}
                          target="_blank"
                          rel="noreferrer"
                          onClick={e => e.stopPropagation()}
                          style={{
                            fontSize: 11, padding: '2px 8px',
                            background: 'var(--bg-elevated)', border: '1px solid var(--border-default)',
                            borderRadius: 6, color: 'var(--text-secondary)',
                            textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 4,
                          }}
                        >
                          Paper <ExternalLink size={9} />
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* Right: deploy panel */}
        <div style={{ position: 'sticky', top: 16, alignSelf: 'flex-start' }}>
          <div className="card" style={{ padding: 20 }}>
            <h2 style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
              {selected ? `Deploy: ${selected.label}` : 'เลือก Model ก่อน'}
            </h2>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
              {selected
                ? 'เลือก provider แล้วกด Deploy — model จะถูก serve ตาม endpoint ที่กำหนด'
                : 'คลิกที่ model card ทางซ้ายเพื่อเลือก'}
            </p>

            {selected && (
              <>
                {/* Provider picker */}
                <div className="mb-4">
                  <label style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-secondary)', marginBottom: 8, display: 'block' }}>
                    Provider
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {([
                      { val: 'ray',   label: 'Ray Serve', icon: Server },
                      { val: 'modal', label: 'Modal',     icon: Cloud },
                    ] as const).map(({ val, label, icon: Icon }) => {
                      const active = provider === val
                      return (
                        <button
                          key={val}
                          onClick={() => setProvider(val)}
                          style={{
                            padding: '10px 12px', borderRadius: 8,
                            border: `1px solid ${active ? 'var(--primary)' : 'var(--border-default)'}`,
                            background: active ? 'var(--primary-dim)' : 'var(--bg-elevated)',
                            color: active ? 'var(--primary-hover)' : 'var(--text-secondary)',
                            cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: 8,
                            fontSize: 12.5, fontWeight: 500,
                          }}
                        >
                          <Icon size={14} />
                          {label}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Model summary */}
                <div style={{
                  padding: 12, borderRadius: 8, background: 'var(--bg-elevated)',
                  fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <Database size={12} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{selected.model}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Zap size={12} />
                    <span>Engine: <strong>{selected.engine}</strong> · Hardware: <strong>{selected.hardware}</strong></span>
                  </div>
                </div>

                {/* Deploy button */}
                <button
                  className="btn btn-primary"
                  style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
                  onClick={handleDeploy}
                  disabled={status.state === 'submitting' || status.state === 'deploying'}
                >
                  {status.state === 'submitting' || status.state === 'deploying' ? (
                    <><Loader2 size={14} className="animate-spin" /> Deploying...</>
                  ) : status.state === 'ready' ? (
                    <><CheckCircle2 size={14} /> Re-deploy</>
                  ) : (
                    <><Zap size={14} /> Deploy ไปยัง {provider === 'ray' ? 'Ray Serve' : 'Modal'}</>
                  )}
                </button>

                {/* Status */}
                {status.state === 'ready' && status.url && (
                  <div style={{
                    marginTop: 14, padding: 12, borderRadius: 8,
                    background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.3)',
                    fontSize: 12,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--success)', fontWeight: 600, marginBottom: 6 }}>
                      <CheckCircle2 size={13} /> Endpoint พร้อมใช้งาน
                    </div>
                    <code style={{
                      display: 'block', padding: 8, borderRadius: 6,
                      background: 'var(--bg-base)', color: 'var(--text-primary)',
                      fontFamily: 'var(--font-mono)', fontSize: 11, wordBreak: 'break-all',
                    }}>{status.url}</code>
                  </div>
                )}
                {status.state === 'error' && (
                  <div style={{
                    marginTop: 14, padding: 12, borderRadius: 8,
                    background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)',
                    color: 'var(--danger)', fontSize: 12,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600, marginBottom: 4 }}>
                      <AlertCircle size={13} /> Deploy failed
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{status.error}</div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
