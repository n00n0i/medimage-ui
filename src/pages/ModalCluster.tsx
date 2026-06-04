import { useState, useEffect } from 'react'
import { RefreshCw, ExternalLink, Server, X, Cloud, Eye, EyeOff, ChevronDown, ChevronUp } from 'lucide-react'

// ── Modal Ray node type ───────────────────────────────────────────────────────
interface ModalNodeSummary {
  nodeId: string
  hostname?: string
  ip?: string
  state?: string
  isHeadNode?: boolean
  cpuTotal?: number
  cpuUsed?: number
  gpus?: { name?: string; utilizationGpu?: number; memoryUsed?: number; memoryTotal?: number }[]
  memTotal?: number
  memUsed?: number
}

const MODAL_GPUS = [
  { id: 'cpu',       label: 'CPU Only',  vram: null, price: 0.06,  badge: 'Budget',    color: 'var(--text-muted)' },
  { id: 'T4',        label: 'T4',        vram: 16,   price: 0.59,  badge: 'Entry',     color: 'oklch(65% 0.14 160)' },
  { id: 'L4',        label: 'L4',        vram: 24,   price: 0.80,  badge: 'Efficient', color: 'oklch(65% 0.14 160)' },
  { id: 'A10G',      label: 'A10G',      vram: 24,   price: 1.10,  badge: 'Balanced',  color: 'var(--primary)' },
  { id: 'A100-40GB', label: 'A100 40G',  vram: 40,   price: 3.04,  badge: 'High End',  color: 'var(--accent)' },
  { id: 'A100-80GB', label: 'A100 80G',  vram: 80,   price: 4.20,  badge: 'High End',  color: 'var(--accent)' },
  { id: 'H100',      label: 'H100',      vram: 80,   price: 3.95,  badge: 'Top Tier',  color: 'oklch(65% 0.20 30)' },
] as const

type ModalGpuId = typeof MODAL_GPUS[number]['id']
type ModalStatus = 'idle' | 'deploying' | 'running' | 'stopping' | 'error'

interface ModalConfig {
  tokenId: string
  tokenSecret: string
  gpuType: ModalGpuId
  numWorkers: number
}

const MODAL_KEY = 'ray_modal_config'

function loadModalConfig(): ModalConfig {
  try {
    return { tokenId: '', tokenSecret: '', gpuType: 'T4', numWorkers: 1, ...JSON.parse(localStorage.getItem(MODAL_KEY) ?? '{}') }
  } catch {
    return { tokenId: '', tokenSecret: '', gpuType: 'T4', numWorkers: 1 }
  }
}

const STATUS_LABEL: Record<ModalStatus, string> = {
  idle:      'Not deployed',
  deploying: 'Deploying to Modal…',
  running:   'Running on Modal',
  stopping:  'Stopping…',
  error:     'Deploy failed',
}
const STATUS_COLOR: Record<ModalStatus, string> = {
  idle:      'var(--text-muted)',
  deploying: 'var(--warning)',
  running:   'var(--success)',
  stopping:  'var(--warning)',
  error:     'var(--danger)',
}

// ─────────────────────────────────────────────────────────────────────────────

export default function ModalCluster() {
  const [cfg, setCfg] = useState<ModalConfig>(loadModalConfig)
  const [showSecret, setShowSecret] = useState(false)
  const [status, setStatus] = useState<ModalStatus>('idle')
  const [numWorkers, setNumWorkers] = useState(0)
  const [scaling, setScaling] = useState(false)
  const [rayUrl, setRayUrl] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const [errMsg, setErrMsg] = useState<string | null>(null)
  const [modalNodes, setModalNodes] = useState<ModalNodeSummary[]>([])
  const [rayStopping, setRayStopping] = useState(false)

  // Persist config
  useEffect(() => {
    localStorage.setItem(MODAL_KEY, JSON.stringify(cfg))
  }, [cfg])

  // Poll status when active
  useEffect(() => {
    if (status === 'idle') return
    const id = setInterval(async () => {
      try {
        const r = await fetch('/api/modal/status')
        if (!r.ok) return
        const d = await r.json()
        if (d.logs?.length) setLogs(d.logs)
        if (d.ray_url) setRayUrl(d.ray_url)
        if (d.num_workers != null) setNumWorkers(d.num_workers)
        setStatus(d.status as ModalStatus)
        if (d.status === 'idle' || d.status === 'error') clearInterval(id)
      } catch { /* ignore */ }
    }, 3000)
    return () => clearInterval(id)
  }, [status])

  // Sync status on mount
  useEffect(() => {
    fetch('/api/modal/status').then(r => r.ok ? r.json() : null).then(d => {
      if (!d) return
      setStatus(d.status as ModalStatus)
      if (d.ray_url) setRayUrl(d.ray_url)
      if (d.num_workers != null) setNumWorkers(d.num_workers)
      if (d.logs?.length) setLogs(d.logs)
    }).catch(() => {})
  }, [])

  // Poll Modal Ray nodes when running
  useEffect(() => {
    if (status !== 'running') { setModalNodes([]); return }
    const fetchNodes = () =>
      fetch('/api/modal/nodes')
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          const summary: ModalNodeSummary[] = d?.data?.summary ?? []
          if (summary.length) setModalNodes(summary)
        })
        .catch(() => {})
    fetchNodes()
    const id = setInterval(fetchNodes, 10000)
    return () => clearInterval(id)
  }, [status])

  const handleStart = async () => {
    setStatus('deploying')
    setErrMsg(null)
    setLogs([])
    setRayUrl(null)
    try {
      const r = await fetch('/api/modal/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token_id: cfg.tokenId,
          token_secret: cfg.tokenSecret,
          gpu_type: cfg.gpuType,
          num_workers: numWorkers,
        }),
      })
      const d = await r.json()
      if (!r.ok) {
        setStatus('error')
        setErrMsg(d.detail ?? 'Deploy failed')
      }
    } catch (e) {
      setStatus('error')
      setErrMsg((e as Error).message)
    }
  }

  async function handleScale(delta: number) {
    setScaling(true)
    try {
      const r = await fetch('/api/modal/scale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delta }),
      })
      const d = await r.json()
      if (r.ok && d.num_workers != null) setNumWorkers(d.num_workers)
    } catch { /* ignore */ }
    setScaling(false)
  }

  const handleStop = async () => {
    setStatus('stopping')
    try {
      await fetch('/api/modal/stop', { method: 'POST' })
    } catch {
      setStatus('idle')
    }
  }

  async function handleRayStop() {
    setRayStopping(true)
    try {
      const r = await fetch('/api/modal/ray-stop', { method: 'POST' })
      const d = await r.json()
      if (r.ok) {
        const msgs = (d.containers as { container: string; exit_code: number; output: string }[])
          ?.map(c => `ray stop → ${c.container.slice(0, 12)}: exit ${c.exit_code}`)
        setLogs(prev => [...prev, ...(msgs ?? [d.message ?? 'ray stop sent'])])
      } else {
        setLogs(prev => [...prev, `ray stop failed: ${d.detail}`])
      }
    } catch (e) {
      setLogs(prev => [...prev, `ray stop error: ${(e as Error).message}`])
    }
    setRayStopping(false)
  }

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '7px 10px', borderRadius: 7, fontSize: 12,
    background: 'var(--bg-base)', border: '1px solid var(--border)',
    color: 'var(--text-primary)', outline: 'none', boxSizing: 'border-box',
    fontFamily: 'var(--font-mono)',
  }

  const canStart = cfg.tokenId.trim().length > 0 && cfg.tokenSecret.trim().length > 0
  const isActive = status === 'running' || status === 'deploying' || status === 'stopping'
  const selectedGpu = MODAL_GPUS.find(g => g.id === cfg.gpuType)!

  return (
    <div style={{ padding: '24px 28px', maxWidth: 900 }}>
      {/* ── Page Header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)', margin: '0 0 6px' }}>
            Modal Cluster
          </h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', margin: 0 }}>
            Deploy a standalone GPU Ray cluster on Modal.com cloud — independent from local infrastructure.
          </p>
        </div>
        <a
          href="https://modal.com/settings"
          target="_blank"
          rel="noopener noreferrer"
          style={{ fontSize: 12, color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: 4, textDecoration: 'none', marginTop: 4 }}
        >
          modal.com/settings <ExternalLink size={11} />
        </a>
      </div>

      {/* ── Status Badge ── */}
      {status !== 'idle' && (
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 7, marginBottom: 20,
          padding: '6px 14px', borderRadius: 20,
          background: `color-mix(in oklch, ${STATUS_COLOR[status]} 10%, transparent)`,
          border: `1px solid color-mix(in oklch, ${STATUS_COLOR[status]} 25%, transparent)`,
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: STATUS_COLOR[status],
            boxShadow: status === 'running' ? `0 0 6px ${STATUS_COLOR[status]}` : undefined,
            animation: (status === 'deploying' || status === 'stopping') ? 'pulse 1.4s infinite' : undefined,
          }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: STATUS_COLOR[status] }}>
            {errMsg ?? STATUS_LABEL[status]}
          </span>
          {rayUrl && status === 'running' && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              · {rayUrl.replace('https://', '')}
            </span>
          )}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* ── Left: Config ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Credentials card */}
          <div className="card">
            <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 14px', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Cloud size={13} style={{ color: 'var(--primary)' }} /> Credentials
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div>
                <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Token ID</label>
                <input
                  style={inputStyle}
                  type="text"
                  value={cfg.tokenId}
                  onChange={e => setCfg(c => ({ ...c, tokenId: e.target.value }))}
                  placeholder="ak-xxxxxxxxxxxxxxxxxxxxxxxx"
                  autoComplete="off"
                />
              </div>
              <div style={{ position: 'relative' }}>
                <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Token Secret</label>
                <input
                  style={{ ...inputStyle, paddingRight: 32 }}
                  type={showSecret ? 'text' : 'password'}
                  value={cfg.tokenSecret}
                  onChange={e => setCfg(c => ({ ...c, tokenSecret: e.target.value }))}
                  placeholder="as-xxxxxxxxxxxxxxxxxxxxxxxx"
                  autoComplete="new-password"
                />
                <button
                  onClick={() => setShowSecret(s => !s)}
                  style={{ position: 'absolute', right: 8, top: 22, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 2 }}
                >
                  {showSecret ? <EyeOff size={13} /> : <Eye size={13} />}
                </button>
              </div>
            </div>
          </div>

          {/* GPU selector card */}
          <div className="card">
            <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 14px' }}>Instance Type</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {MODAL_GPUS.map(gpu => {
                const sel = cfg.gpuType === gpu.id
                return (
                  <button
                    key={gpu.id}
                    onClick={() => setCfg(c => ({ ...c, gpuType: gpu.id as ModalGpuId }))}
                    disabled={isActive}
                    style={{
                      padding: '9px 12px', borderRadius: 8, cursor: isActive ? 'default' : 'pointer',
                      border: `1.5px solid ${sel ? gpu.color : 'var(--border)'}`,
                      background: sel ? `color-mix(in oklch, ${gpu.color} 10%, var(--bg-elevated))` : 'var(--bg-elevated)',
                      display: 'flex', alignItems: 'center', gap: 10,
                      transition: 'border-color 0.15s, background 0.15s',
                    }}
                  >
                    <div style={{ flex: 1, textAlign: 'left' }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: sel ? gpu.color : 'var(--text-primary)' }}>
                        {gpu.label}
                      </span>
                      {gpu.vram && (
                        <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 6 }}>{gpu.vram} GB VRAM</span>
                      )}
                    </div>
                    <span style={{
                      fontSize: 9, padding: '2px 6px', borderRadius: 4, fontWeight: 700,
                      background: `color-mix(in oklch, ${gpu.color} 15%, transparent)`,
                      color: gpu.color,
                    }}>{gpu.badge}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: gpu.color, minWidth: 60, textAlign: 'right' }}>
                      ${gpu.price}/hr
                    </span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {/* ── Right: Actions + Nodes ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Action card */}
          <div className="card">
            <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 14px' }}>Cluster Control</h3>

            {/* Workers counter */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)', flex: 1 }}>Workers</span>
              <button
                className="btn btn-secondary btn-sm"
                disabled={isActive && status !== 'running'}
                onClick={() => status === 'running' ? handleScale(-1) : setNumWorkers(w => Math.max(0, w - 1))}
                style={{ padding: '2px 10px', fontWeight: 700, fontSize: 16, opacity: scaling ? 0.5 : 1 }}
              >−</button>
              <span style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', minWidth: 28, textAlign: 'center' }}>
                {numWorkers}
              </span>
              <button
                className="btn btn-secondary btn-sm"
                disabled={isActive && status !== 'running'}
                onClick={() => status === 'running' ? handleScale(1) : setNumWorkers(w => Math.min(8, w + 1))}
                style={{ padding: '2px 10px', fontWeight: 700, fontSize: 16, opacity: scaling ? 0.5 : 1 }}
              >+</button>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                ~${((numWorkers + 1) * selectedGpu.price).toFixed(2)}/hr
              </span>
            </div>

            {/* Buttons */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {status === 'running' ? (
                <button
                  className="btn btn-sm"
                  onClick={handleStop}
                  style={{ background: 'var(--danger)', color: '#fff', border: 'none', display: 'flex', alignItems: 'center', gap: 6 }}
                >
                  <X size={12} /> Stop Cluster
                </button>
              ) : (
                <button
                  className="btn btn-sm"
                  onClick={handleStart}
                  disabled={!canStart || status === 'deploying' || status === 'stopping'}
                  style={{
                    background: canStart ? 'var(--primary)' : undefined,
                    color: canStart ? '#fff' : undefined,
                    border: 'none', display: 'flex', alignItems: 'center', gap: 6,
                    opacity: (!canStart || isActive) ? 0.55 : 1,
                  }}
                >
                  {status === 'deploying' ? (
                    <><RefreshCw size={12} className="animate-spin" /> Deploying…</>
                  ) : status === 'stopping' ? (
                    <><RefreshCw size={12} className="animate-spin" /> Stopping…</>
                  ) : (
                    <><Cloud size={12} /> Start Cluster</>
                  )}
                </button>
              )}

              {status === 'running' && (
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleRayStop}
                  disabled={rayStopping}
                  title="Send `ray stop` to Modal containers"
                  style={{ display: 'flex', alignItems: 'center', gap: 6, opacity: rayStopping ? 0.5 : 1 }}
                >
                  {rayStopping ? <RefreshCw size={12} className="animate-spin" /> : <X size={12} />}
                  Ray Stop
                </button>
              )}

              {rayUrl && (
                <a
                  href={rayUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary btn-sm"
                  style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                >
                  <ExternalLink size={12} /> Ray Dashboard
                </a>
              )}
            </div>
          </div>

          {/* Logs card */}
          {logs.length > 0 && (
            <div className="card">
              <button
                onClick={() => setShowLogs(s => !s)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 0, color: 'var(--text-primary)' }}
              >
                <span style={{ fontSize: 13, fontWeight: 600 }}>Deploy Logs</span>
                {showLogs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
              {showLogs && (
                <pre style={{
                  marginTop: 10, maxHeight: 200, overflowY: 'auto',
                  padding: '8px 10px', borderRadius: 7, fontSize: 10,
                  background: 'var(--bg-base)', border: '1px solid var(--border)',
                  color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)',
                  whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                }}>
                  {logs.join('\n')}
                </pre>
              )}
            </div>
          )}

          {/* Modal Nodes card */}
          {status === 'running' && modalNodes.length > 0 && (
            <div className="card">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                <Server size={13} style={{ color: 'var(--text-muted)' }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Modal Nodes</span>
                <span style={{
                  fontSize: 10, padding: '1px 7px', borderRadius: 10, fontWeight: 600,
                  background: 'color-mix(in oklch, var(--success) 12%, transparent)',
                  color: 'var(--success)', border: '1px solid color-mix(in oklch, var(--success) 25%, transparent)',
                }}>
                  {modalNodes.length} alive
                </span>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 'auto' }}>live · 10s</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {modalNodes.map(node => {
                  const cpuPct = node.cpuTotal ? Math.round((node.cpuUsed ?? 0) / node.cpuTotal * 100) : 0
                  const memPct = node.memTotal ? Math.round((node.memUsed ?? 0) / node.memTotal * 100) : 0
                  return (
                    <div key={node.nodeId} style={{
                      padding: '10px 12px', borderRadius: 8,
                      background: 'var(--bg-base)', border: '1px solid var(--border)',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: '2px 6px', borderRadius: 4,
                          background: node.isHeadNode
                            ? 'color-mix(in oklch, var(--primary) 15%, transparent)'
                            : 'color-mix(in oklch, var(--accent) 15%, transparent)',
                          color: node.isHeadNode ? 'var(--primary)' : 'var(--accent)',
                          border: `1px solid color-mix(in oklch, ${node.isHeadNode ? 'var(--primary)' : 'var(--accent)'} 25%, transparent)`,
                        }}>
                          {node.isHeadNode ? 'HEAD' : 'WORKER'}
                        </span>
                        <span style={{ fontSize: 11, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {node.hostname ?? node.ip ?? node.nodeId.slice(0, 12)}
                        </span>
                        <span style={{
                          fontSize: 10, marginLeft: 'auto',
                          color: node.state === 'ALIVE' ? 'var(--success)' : 'var(--warning)',
                        }}>
                          {node.state ?? 'unknown'}
                        </span>
                      </div>
                      <div style={{ display: 'flex', gap: 12 }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>
                            <span>CPU</span><span style={{ fontFamily: 'var(--font-mono)' }}>{cpuPct}%</span>
                          </div>
                          <div style={{ height: 4, borderRadius: 2, background: 'var(--border)' }}>
                            <div style={{ height: '100%', borderRadius: 2, width: `${cpuPct}%`, background: 'var(--primary)', transition: 'width 0.4s' }} />
                          </div>
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>
                            <span>MEM</span><span style={{ fontFamily: 'var(--font-mono)' }}>{memPct}%</span>
                          </div>
                          <div style={{ height: 4, borderRadius: 2, background: 'var(--border)' }}>
                            <div style={{ height: '100%', borderRadius: 2, width: `${memPct}%`, background: 'oklch(65% 0.14 160)', transition: 'width 0.4s' }} />
                          </div>
                        </div>
                      </div>
                      {node.gpus && node.gpus.length > 0 && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 8 }}>
                          {node.gpus.map((gpu, i) => (
                            <span key={i} style={{
                              fontSize: 10, color: 'var(--accent)', padding: '2px 8px', borderRadius: 4,
                              background: 'color-mix(in oklch, var(--accent) 10%, transparent)',
                              border: '1px solid color-mix(in oklch, var(--accent) 20%, transparent)',
                              fontFamily: 'var(--font-mono)',
                            }}>
                              {gpu.name ?? `GPU${i}`}
                              {gpu.memoryUsed != null && ` · ${gpu.memoryUsed}/${gpu.memoryTotal ?? '?'} MB`}
                              {gpu.utilizationGpu != null && (
                                <span style={{
                                  marginLeft: 4, fontWeight: 600,
                                  color: (gpu.utilizationGpu ?? 0) > 80 ? 'var(--danger)' : (gpu.utilizationGpu ?? 0) > 40 ? 'var(--warning)' : 'var(--success)',
                                }}>
                                  {gpu.utilizationGpu}%
                                </span>
                              )}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
