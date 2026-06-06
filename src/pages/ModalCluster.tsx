import { useState, useEffect, useCallback } from 'react'
import { Cloud, Eye, EyeOff, CheckCircle2, AlertCircle, Loader, ExternalLink, Trash2, RefreshCw, X, StopCircle, Play } from 'lucide-react'

interface ModalCredState {
  configured: boolean
  token_id: string
  updated_at: string
}

interface ModalDeployment {
  id: string
  model_name: string
  modal_url: string
  inference_provider: string
  training_type?: string
  engine?: string
  deployed_at?: string
  status?: string
  gpu_type?: string
  num_workers?: number
  memory_mb?: number
  scaledown_window_s?: number
  min_containers?: number
}

type VerifyStatus = 'idle' | 'checking' | 'ok' | 'fail'

export default function ModalConfig() {
  const [tokenId, setTokenId]       = useState('')
  const [tokenSecret, setTokenSecret] = useState('')
  const [showSecret, setShowSecret] = useState(false)
  const [saving, setSaving]         = useState(false)
  const [saved, setSaved]           = useState(false)
  const [cred, setCred]             = useState<ModalCredState | null>(null)
  const [verifyStatus, setVerifyStatus] = useState<VerifyStatus>('idle')
  const [verifyOutput, setVerifyOutput] = useState('')
  const [deployments, setDeployments]   = useState<ModalDeployment[]>([])
  const [stopping, setStopping]         = useState<Record<string, boolean>>({})
  const [deleting, setDeleting]         = useState(false)
  const [clusterStatus, setClusterStatus] = useState<{
    status: string; ray_url: string | null
    gpu_type?: string; num_workers?: number
  } | null>(null)
  const [starting, setStarting]           = useState(false)
  const [stoppingCluster, setStoppingCluster] = useState(false)
  const [gpuType, setGpuType]         = useState('T4')
  const [numWorkers, setNumWorkers]   = useState(1)

  const inputStyle: React.CSSProperties = {
    width: '100%', boxSizing: 'border-box', padding: '8px 11px', fontSize: 13,
    background: 'var(--bg-base)', border: '1px solid var(--border)',
    borderRadius: 8, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)',
    outline: 'none',
  }

  const loadCred = useCallback(async () => {
    try {
      const r = await fetch('/api/modal/credentials')
      if (r.ok) setCred(await r.json())
    } catch { /* ignore */ }
  }, [])

  const loadDeployments = useCallback(async () => {
    try {
      const r = await fetch('/api/modal/deployments')
      if (r.ok) {
        const d = await r.json()
        setDeployments(d.deployments ?? [])
      }
    } catch { /* ignore */ }
  }, [])

  const loadClusterStatus = useCallback(async () => {
    try {
      const r = await fetch('/api/modal/status')
      if (r.ok) {
        const s = await r.json()
        setClusterStatus(s)
        // Sync the selectors to the live cluster config so what you see
        // here matches what the Train popup most-recently started.
        if (s.gpu_type)    setGpuType(s.gpu_type)
        if (s.num_workers) setNumWorkers(s.num_workers)
      }
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    loadCred()
    loadDeployments()
    loadClusterStatus()
    const t = setInterval(loadClusterStatus, 5000)
    return () => clearInterval(t)
  }, [loadCred, loadDeployments, loadClusterStatus])

  async function handleSave() {
    if (!tokenId.trim() || !tokenSecret.trim()) return
    setSaving(true)
    try {
      const r = await fetch('/api/modal/credentials', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token_id: tokenId.trim(), token_secret: tokenSecret.trim() }),
      })
      if (r.ok) {
        setSaved(true)
        setTokenId('')
        setTokenSecret('')
        setVerifyStatus('idle')
        setTimeout(() => setSaved(false), 2500)
        await loadCred()
      }
    } finally { setSaving(false) }
  }

  async function handleVerify() {
    setVerifyStatus('checking')
    setVerifyOutput('')
    try {
      const r = await fetch('/api/modal/verify', { method: 'POST' })
      const d = await r.json()
      setVerifyStatus(d.ok ? 'ok' : 'fail')
      setVerifyOutput(d.output || '')
    } catch (e) {
      setVerifyStatus('fail')
      setVerifyOutput((e as Error).message)
    }
  }

  async function handleDelete() {
    if (!confirm('ลบ Modal credentials? การ deploy ที่ใช้ credentials เหล่านี้จะหยุดทำงาน')) return
    setDeleting(true)
    try {
      await fetch('/api/modal/credentials', { method: 'DELETE' })
      setCred(null)
      setVerifyStatus('idle')
      setVerifyOutput('')
    } finally { setDeleting(false) }
  }

  async function handleStopModel(modelId: string) {
    setStopping(s => ({ ...s, [modelId]: true }))
    try {
      await fetch(`/api/jobs/${modelId}/deploy-modal/stop`, { method: 'POST' })
      await loadDeployments()
    } finally {
      setStopping(s => ({ ...s, [modelId]: false }))
    }
  }

  async function handleStartCluster() {
    if (!cred?.configured) {
      alert('Set Modal credentials first')
      return
    }
    if (!confirm(`Start Modal Ray cluster? (${numWorkers}× ${gpuType} — จะมีค่าใช้จ่ายจาก Modal ตาม GPU ที่เลือก)`)) return
    setStarting(true)
    try {
      const r = await fetch('/api/modal/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gpu_type: gpuType, num_workers: numWorkers }),
      })
      if (!r.ok) {
        const t = await r.text()
        alert(`Start failed: ${t}`)
      } else {
        await loadClusterStatus()
      }
    } finally { setStarting(false) }
  }

  async function handleStopCluster() {
    if (!confirm('Stop Modal Ray cluster? Training jobs running on Modal will be interrupted.')) return
    setStoppingCluster(true)
    try {
      await fetch('/api/modal/stop', { method: 'POST' })
      await loadClusterStatus()
    } finally { setStoppingCluster(false) }
  }

  const clusterRunning = clusterStatus?.status === 'running'
  const clusterBusy   = clusterStatus?.status === 'deploying' || clusterStatus?.status === 'stopping'

  const canSave = tokenId.trim().length > 0 && tokenSecret.trim().length > 0

  return (
    <div style={{ padding: '24px 28px', maxWidth: 720 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)', margin: '0 0 6px', display: 'flex', alignItems: 'center', gap: 9 }}>
            <Cloud size={20} color="#8b5cf6" /> Modal Configuration
          </h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', margin: 0 }}>
            ตั้งค่า credentials สำหรับ deploy model บน Modal.com GPU cloud
          </p>
        </div>
        <a
          href="https://modal.com/settings"
          target="_blank"
          rel="noopener noreferrer"
          style={{ fontSize: 12, color: '#8b5cf6', display: 'flex', alignItems: 'center', gap: 4, textDecoration: 'none', marginTop: 4 }}
        >
          modal.com/settings <ExternalLink size={11} />
        </a>
      </div>

      {/* Current Status */}
      {cred && (
        <div style={{
          marginBottom: 20, padding: '12px 16px', borderRadius: 10,
          background: cred.configured ? '#8b5cf610' : 'var(--bg-elevated)',
          border: `1px solid ${cred.configured ? '#8b5cf630' : 'var(--border)'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {cred.configured
              ? <CheckCircle2 size={16} color="#8b5cf6" />
              : <AlertCircle size={16} color="var(--text-muted)" />}
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: cred.configured ? '#8b5cf6' : 'var(--text-muted)' }}>
                {cred.configured ? 'Credentials ถูกบันทึกแล้ว' : 'ยังไม่ได้ตั้งค่า'}
              </div>
              {cred.configured && (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                  Token ID: {cred.token_id} · อัปเดต {cred.updated_at?.slice(0, 16).replace('T', ' ')}
                </div>
              )}
            </div>
          </div>
          {cred.configured && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {verifyStatus === 'ok' && (
                <span style={{ fontSize: 11, color: '#22c55e', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <CheckCircle2 size={11} /> Valid
                </span>
              )}
              {verifyStatus === 'fail' && (
                <span style={{ fontSize: 11, color: '#ef4444', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <X size={11} /> Invalid
                </span>
              )}
              <button
                className="btn btn-secondary btn-sm"
                style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                onClick={handleVerify}
                disabled={verifyStatus === 'checking'}
              >
                {verifyStatus === 'checking'
                  ? <><Loader size={11} className="animate-spin" /> Verifying…</>
                  : <><RefreshCw size={11} /> Verify</>}
              </button>
              <button
                className="btn btn-sm"
                style={{ display: 'flex', alignItems: 'center', gap: 5, background: '#ef444415', color: '#ef4444', border: '1px solid #ef444430' }}
                onClick={handleDelete}
                disabled={deleting}
              >
                <Trash2 size={11} /> Remove
              </button>
            </div>
          )}
        </div>
      )}

      {/* Verify output */}
      {verifyOutput && (
        <div style={{
          marginBottom: 16, padding: '8px 12px', borderRadius: 8, fontSize: 11,
          fontFamily: 'var(--font-mono)', lineHeight: 1.5,
          background: verifyStatus === 'ok' ? '#22c55e10' : '#ef444410',
          border: `1px solid ${verifyStatus === 'ok' ? '#22c55e30' : '#ef444430'}`,
          color: verifyStatus === 'ok' ? '#22c55e' : '#ef4444',
          whiteSpace: 'pre-wrap',
        }}>{verifyOutput}</div>
      )}

      {/* Credentials Form */}
      <div className="card" style={{ marginBottom: 20 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 16px', display: 'flex', alignItems: 'center', gap: 6 }}>
          <Cloud size={13} color="#8b5cf6" />
          {cred?.configured ? 'อัปเดต Credentials' : 'ตั้งค่า Credentials'}
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <label style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: 5 }}>
              Token ID
            </label>
            <input
              type="text"
              value={tokenId}
              onChange={e => setTokenId(e.target.value)}
              placeholder="ak-xxxxxxxxxxxxxxxxxxxxxxxx"
              style={inputStyle}
              autoComplete="off"
            />
          </div>

          <div>
            <label style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: 5 }}>
              Token Secret
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type={showSecret ? 'text' : 'password'}
                value={tokenSecret}
                onChange={e => setTokenSecret(e.target.value)}
                placeholder="as-xxxxxxxxxxxxxxxxxxxxxxxx"
                style={{ ...inputStyle, paddingRight: 36 }}
                autoComplete="new-password"
              />
              <button
                onClick={() => setShowSecret(v => !v)}
                style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 2, display: 'flex' }}
              >
                {showSecret ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              ดู token ได้ที่{' '}
              <a href="https://modal.com/settings" target="_blank" rel="noopener noreferrer" style={{ color: '#8b5cf6' }}>modal.com/settings</a>
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button
              className="btn btn-sm"
              style={{
                background: canSave ? '#8b5cf6' : undefined,
                color: canSave ? '#fff' : undefined,
                border: 'none',
                display: 'flex', alignItems: 'center', gap: 6,
                opacity: (!canSave || saving) ? 0.55 : 1,
              }}
              onClick={handleSave}
              disabled={!canSave || saving}
            >
              {saving ? <Loader size={11} className="animate-spin" /> : <Cloud size={11} />}
              {saving ? 'Saving…' : saved ? 'Saved!' : 'Save Credentials'}
            </button>
            {saved && <CheckCircle2 size={14} color="#22c55e" />}
          </div>
        </div>
      </div>

      {/* How to use */}
      {cred?.configured && (
        <div style={{
          padding: '12px 16px', borderRadius: 10, marginBottom: 20,
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>วิธีใช้งาน</div>
          <ol style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 5 }}>
            <li style={{ fontSize: 12, color: 'var(--text-secondary)' }}>กด <strong>Start Cluster</strong> ด้านล่างเพื่อ spin up Ray cluster บน Modal (T4, 1 worker)</li>
            <li style={{ fontSize: 12, color: 'var(--text-secondary)' }}>ไปที่หน้า <strong>Train</strong> — เลือก cluster = <strong>Modal</strong> แล้ว submit job</li>
            <li style={{ fontSize: 12, color: 'var(--text-secondary)' }}>หรือ: ไปที่ <strong>Models</strong> → แท็บ <strong>Deploy</strong> → <strong>Modal</strong> tab → <strong>Deploy to Modal</strong></li>
            <li style={{ fontSize: 12, color: 'var(--text-secondary)' }}>เสร็จแล้วกด <strong>Stop Cluster</strong> เพื่อปิด (ลดค่าใช้จ่าย)</li>
          </ol>
        </div>
      )}

      {/* Cluster control — Start / Stop using saved creds */}
      {cred?.configured && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, gap: 10, flexWrap: 'wrap' }}>
            <div>
              <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                Modal Ray Cluster
                {clusterStatus && (
                  <span style={{
                    fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 5,
                    background: clusterRunning ? '#22c55e20' : clusterStatus.status === 'error' ? '#ef444420' : '#f59e0b20',
                    color:      clusterRunning ? '#22c55e'   : clusterStatus.status === 'error' ? '#ef4444'   : '#f59e0b',
                    textTransform: 'uppercase', letterSpacing: '0.06em',
                  }}>{clusterStatus.status}</span>
                )}
              </h3>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                {clusterRunning && clusterStatus?.ray_url
                  ? <>Ray dashboard: <code style={{ fontFamily: 'var(--font-mono)', color: '#8b5cf6' }}>{clusterStatus.ray_url}</code></>
                  : clusterBusy
                    ? <span style={{ color: '#f59e0b' }}>กำลัง {clusterStatus?.status}… (อาจใช้เวลา 1–3 นาที)</span>
                    : <>ยังไม่ได้ start — เลือก cluster <strong style={{ color: '#8b5cf6' }}>Modal</strong> ในหน้า Train ไม่ได้จนกว่าจะ start</>}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {/* GPU + workers selectors — disabled while running */}
              <select
                value={gpuType}
                onChange={e => setGpuType(e.target.value)}
                disabled={clusterRunning || clusterBusy}
                title="GPU type"
                style={{
                  padding: '8px 10px', fontSize: 12, fontWeight: 600,
                  background: 'var(--bg-base)', color: 'var(--text-primary)',
                  border: '1px solid var(--border)', borderRadius: 8,
                  cursor: (clusterRunning || clusterBusy) ? 'not-allowed' : 'pointer',
                  opacity: (clusterRunning || clusterBusy) ? 0.55 : 1,
                }}
              >
                <option value="T4">T4</option>
                <option value="L4">L4</option>
                <option value="A10G">A10G</option>
                <option value="L40S">L40S</option>
                <option value="A100">A100 (40GB)</option>
                <option value="A100-80GB">A100 (80GB)</option>
                <option value="H100">H100</option>
                <option value="cpu">CPU only</option>
              </select>
              <select
                value={numWorkers}
                onChange={e => setNumWorkers(parseInt(e.target.value))}
                disabled={clusterRunning || clusterBusy}
                title="Worker count"
                style={{
                  padding: '8px 10px', fontSize: 12, fontWeight: 600,
                  background: 'var(--bg-base)', color: 'var(--text-primary)',
                  border: '1px solid var(--border)', borderRadius: 8,
                  cursor: (clusterRunning || clusterBusy) ? 'not-allowed' : 'pointer',
                  opacity: (clusterRunning || clusterBusy) ? 0.55 : 1,
                }}
              >
                {[1, 2, 4, 8].map(n => <option key={n} value={n}>{n} worker{n > 1 ? 's' : ''}</option>)}
              </select>
              <button
                onClick={handleStartCluster}
                disabled={!cred?.configured || clusterRunning || clusterBusy || starting}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', fontSize: 12, fontWeight: 600,
                  background: (clusterRunning || clusterBusy) ? 'var(--bg-elevated)' : '#8b5cf6',
                  color:      (clusterRunning || clusterBusy) ? 'var(--text-muted)'  : '#fff',
                  border: 'none', borderRadius: 8, cursor: (clusterRunning || clusterBusy) ? 'not-allowed' : 'pointer',
                  opacity: (!cred?.configured || clusterRunning || clusterBusy || starting) ? 0.55 : 1,
                }}
              >
                {starting ? <Loader size={12} className="animate-spin" /> : <Play size={12} />}
                {starting ? 'Starting…' : 'Start Cluster'}
              </button>
              <button
                onClick={handleStopCluster}
                disabled={!clusterRunning || stoppingCluster}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', fontSize: 12, fontWeight: 600,
                  background: '#ef444415', color: '#ef4444', border: '1px solid #ef444430',
                  borderRadius: 8, cursor: clusterRunning ? 'pointer' : 'not-allowed',
                  opacity: (!clusterRunning || stoppingCluster) ? 0.55 : 1,
                }}
              >
                {stoppingCluster ? <Loader size={12} className="animate-spin" /> : <StopCircle size={12} />}
                {stoppingCluster ? 'Stopping…' : 'Stop Cluster'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal Cluster Status — prominent top-level card summarising the running cluster */}
      {deployments.length > 0 && (() => {
        const first = deployments[0]
        const gpu    = (first.gpu_type ?? 'T4').toUpperCase()
        const memMb  = first.memory_mb ?? 16384
        const workers= first.num_workers ?? 1
        const idleSec= first.scaledown_window_s ?? 300
        const isUp   = (first.status ?? 'running') === 'running'
        return (
          <div className="card" style={{ marginBottom: 20, borderColor: isUp ? '#22c55e40' : 'var(--border)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: isUp ? '#22c55e' : '#f59e0b',
                  boxShadow: isUp ? '0 0 8px #22c55e' : '0 0 8px #f59e0b',
                }} />
                <h3 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                  Modal Cluster Status
                </h3>
                <span style={{
                  fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 5,
                  background: isUp ? '#22c55e20' : '#f59e0b20',
                  color:      isUp ? '#22c55e'   : '#f59e0b',
                  textTransform: 'uppercase', letterSpacing: '0.06em',
                }}>{isUp ? 'Online' : (first.status ?? 'unknown')}</span>
              </div>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {deployments.length} active deployment{deployments.length !== 1 ? 's' : ''}
              </span>
            </div>

            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
              gap: 14, paddingTop: 14, borderTop: '1px solid var(--border-subtle)',
            }}>
              {([
                { label: 'Primary model',  value: first.model_name || first.id,                                       sub: first.training_type ? `${first.training_type} · ${first.engine ?? '?'}` : null },
                { label: 'GPU',            value: gpu === 'CPU' ? 'CPU only' : gpu,                                   sub: `${workers} worker${workers !== 1 ? 's' : ''}` },
                { label: 'Memory / container', value: memMb >= 1024 ? `${(memMb / 1024).toFixed(0)} GB` : `${memMb} MB`, sub: `min ${first.min_containers ?? 1} container` },
                { label: 'Idle shut',      value: idleSec >= 60 ? `${Math.round(idleSec / 60)} min` : `${idleSec} s`,  sub: 'scaledown window' },
                { label: 'Endpoint',       value: first.modal_url.replace(/^https?:\/\//, '').slice(0, 32) + '…',    sub: <a href={first.modal_url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary-hover)', textDecoration: 'none' }}>open ↗</a> },
                { label: 'Deployed at',    value: first.deployed_at ? new Date(first.deployed_at.replace(' ', 'T') + 'Z').toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—', sub: 'last refresh ✓' },
              ] as Array<{ label: string; value: string; sub: React.ReactNode }>).map(s => (
                <div key={s.label}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
                    {s.label}
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums', wordBreak: 'break-all' }}>
                    {s.value}
                  </div>
                  {s.sub != null && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{s.sub}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )
      })()}

      {/* Active Deployments */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
            Active Modal Deployments
          </h3>
          <button
            className="btn btn-secondary btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: 5 }}
            onClick={loadDeployments}
          >
            <RefreshCw size={11} /> Refresh
          </button>
        </div>

        {deployments.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)', fontSize: 13 }}>
            ยังไม่มี model ที่ deploy บน Modal
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {deployments.map(dep => {
              const gpu      = (dep.gpu_type ?? 'T4').toUpperCase()
              const workers  = dep.num_workers ?? 1
              const memMb    = dep.memory_mb ?? 16384
              const idleSec  = dep.scaledown_window_s ?? 300
              const deployed = dep.deployed_at ? new Date(dep.deployed_at.replace(' ', 'T') + (dep.deployed_at.includes('T') ? '' : 'Z')) : null
              const isUp     = (dep.status ?? 'running') === 'running'
              return (
              <div key={dep.id} style={{
                padding: '12px 14px', borderRadius: 8,
                background: 'var(--bg-base)', border: '1px solid var(--border-subtle)',
                display: 'flex', flexDirection: 'column', gap: 10,
              }}>
                {/* Header row: status dot + name + actions */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: isUp ? '#22c55e' : '#f59e0b',
                    boxShadow: isUp ? '0 0 6px #22c55e' : '0 0 6px #f59e0b',
                    flexShrink: 0,
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2, flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                        {dep.model_name || dep.id}
                      </span>
                      {dep.training_type && (
                        <span style={{
                          fontSize: 10, fontWeight: 600, padding: '1px 6px', borderRadius: 4,
                          background: 'var(--bg-elevated)', color: 'var(--text-muted)',
                          textTransform: 'uppercase', letterSpacing: '0.04em',
                        }}>
                          {dep.training_type}
                        </span>
                      )}
                    </div>
                    <code style={{ fontSize: 11, color: '#8b5cf6', wordBreak: 'break-all' }}>{dep.modal_url}</code>
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
                    <a
                      href={dep.modal_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ display: 'flex', alignItems: 'center', padding: '4px 8px', borderRadius: 6, fontSize: 11, color: 'var(--text-muted)', background: 'var(--bg-elevated)', border: '1px solid var(--border)', textDecoration: 'none', gap: 4 }}
                    >
                      <ExternalLink size={11} /> Open
                    </a>
                    <button
                      className="btn btn-sm"
                      style={{ display: 'flex', alignItems: 'center', gap: 4, background: '#ef444415', color: '#ef4444', border: '1px solid #ef444430', fontSize: 11 }}
                      onClick={() => handleStopModel(dep.id)}
                      disabled={stopping[dep.id]}
                    >
                      {stopping[dep.id] ? <Loader size={11} className="animate-spin" /> : <StopCircle size={11} />}
                      Stop
                    </button>
                  </div>
                </div>

                {/* Spec grid */}
                <div style={{
                  display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))',
                  gap: 8, paddingTop: 8, borderTop: '1px solid var(--border-subtle)',
                }}>
                  {([
                    { label: 'GPU',        value: gpu === 'CPU' ? 'CPU only' : gpu },
                    { label: 'Workers',    value: workers.toString() },
                    { label: 'Memory',     value: memMb >= 1024 ? `${(memMb / 1024).toFixed(0)} GB` : `${memMb} MB` },
                    { label: 'Idle shut',  value: idleSec >= 60 ? `${Math.round(idleSec / 60)} min` : `${idleSec} s` },
                    { label: 'Status',     value: isUp ? 'Running' : (dep.status ?? 'unknown'), color: isUp ? 'var(--success)' : 'var(--warning)' },
                    { label: 'Deployed',   value: deployed ? deployed.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—' },
                  ] as Array<{ label: string; value: string; color?: string }>).map(s => (
                    <div key={s.label}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>
                        {s.label}
                      </div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: s.color ?? 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
                        {s.value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
