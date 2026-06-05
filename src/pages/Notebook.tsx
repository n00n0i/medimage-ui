import { useState, useEffect, useRef } from 'react'
import { BookOpen, ExternalLink, Loader, AlertCircle, RefreshCw, ArrowUpCircle, CheckCircle2, Terminal } from 'lucide-react'

interface VersionInfo { current: string; latest: string; update_available: boolean }

export default function Notebook() {
  const [status, setStatus] = useState<'checking' | 'ready' | 'unavailable'>('checking')
  const [version, setVersion] = useState<VersionInfo | null>(null)
  const [updating, setUpdating] = useState(false)
  const [updateLog, setUpdateLog] = useState<string[]>([])
  const [showLog, setShowLog] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const JUPYTER_TOKEN = 'medimage2026'
  const JUPYTER_LAB_URL = `/jupyter/lab?token=${JUPYTER_TOKEN}`

  useEffect(() => {
    checkJupyter()
    fetchVersion()
  }, [])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [updateLog])

  const checkJupyter = async () => {
    setStatus('checking')
    try {
      const res = await fetch(JUPYTER_LAB_URL, { redirect: 'follow', credentials: 'include' })
      setStatus(res.ok ? 'ready' : 'unavailable')
    } catch {
      setStatus('unavailable')
    }
  }

  const fetchVersion = async () => {
    try {
      const res = await fetch('/api/jupyter/version')
      if (res.ok) setVersion(await res.json())
    } catch {}
  }

  const startUpdate = async () => {
    setUpdating(true)
    setUpdateLog([])
    setShowLog(true)
    await fetch('/api/jupyter/update', { method: 'POST' })
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch('/api/jupyter/update-status')
        const d = await r.json()
        setUpdateLog(d.log ?? [])
        if (!d.running) {
          clearInterval(pollRef.current!)
          setUpdating(false)
          // refresh version + jupyter status after update
          setTimeout(() => { fetchVersion(); checkJupyter() }, 3000)
        }
      } catch {}
    }, 1500)
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, padding: '10px 20px',
        background: 'var(--bg-surface)', borderBottom: '1px solid var(--border-default)',
        flexShrink: 0, height: 44,
      }}>
        <BookOpen size={16} color="var(--primary)" />
        <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Jupyter Notebook</span>

        {status === 'checking' && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
            <Loader size={12} className="animate-spin" /> กำลังเชื่อมต่อ...
          </span>
        )}
        {status === 'unavailable' && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--danger)', marginLeft: 8 }}>
            <AlertCircle size={12} /> Jupyter ยังไม่พร้อม
          </span>
        )}
        {status === 'ready' && (
          <span style={{ fontSize: 12, color: 'var(--success, #22c55e)', marginLeft: 8 }}>● Connected</span>
        )}

        {/* Version badge */}
        {version && (
          <span style={{
            fontSize: 11, padding: '2px 8px', borderRadius: 20, marginLeft: 4,
            background: 'var(--bg-elevated)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
          }}>
            v{version.current}
          </span>
        )}

        <div style={{ flex: 1 }} />

        {/* Update available banner */}
        {version?.update_available && !updating && (
          <button
            onClick={startUpdate}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '5px 12px', borderRadius: 7, border: '1px solid #f59e0b60',
              background: '#f59e0b12', color: '#f59e0b', fontSize: 12, fontWeight: 600, cursor: 'pointer',
            }}
          >
            <ArrowUpCircle size={13} />
            Update to v{version.latest}
          </button>
        )}

        {updating && (
          <button
            onClick={() => setShowLog(v => !v)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '5px 12px', borderRadius: 7, border: '1px solid var(--border-default)',
              background: 'var(--bg-elevated)', color: 'var(--text-muted)', fontSize: 12, cursor: 'pointer',
            }}
          >
            <Loader size={12} className="animate-spin" /> Updating… {showLog ? '▲' : '▼'}
          </button>
        )}

        {!version?.update_available && version && !updating && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
            <CheckCircle2 size={11} color="#10b981" /> Up to date
          </span>
        )}

        {status === 'unavailable' && (
          <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 5 }} onClick={checkJupyter}>
            <RefreshCw size={12} /> Retry
          </button>
        )}
        <a
          href={JUPYTER_LAB_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-secondary btn-sm"
          style={{ display: 'flex', alignItems: 'center', gap: 5 }}
        >
          <ExternalLink size={12} /> Open in new tab
        </a>
      </div>

      {/* Update log panel */}
      {showLog && updateLog.length > 0 && (
        <div style={{
          flexShrink: 0, background: '#0d1117', borderBottom: '1px solid var(--border-default)',
          maxHeight: 180, display: 'flex', flexDirection: 'column',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px', borderBottom: '1px solid #ffffff12' }}>
            <Terminal size={11} color="#6b7280" />
            <span style={{ fontSize: 11, color: '#6b7280', fontFamily: 'var(--font-mono)' }}>Update log</span>
            <div style={{ flex: 1 }} />
            <button onClick={() => setShowLog(false)} style={{ background: 'none', border: 'none', color: '#6b7280', cursor: 'pointer', fontSize: 12, lineHeight: 1 }}>✕</button>
          </div>
          <div ref={logRef} style={{ overflow: 'auto', padding: '8px 14px', flex: 1 }}>
            {updateLog.map((line, i) => (
              <div key={i} style={{
                fontFamily: 'var(--font-mono)', fontSize: 11, lineHeight: 1.6,
                color: line.startsWith('ERROR') ? '#f87171' : line.startsWith('✓') ? '#4ade80' : line.startsWith('$') ? '#93c5fd' : '#d1d5db',
              }}>
                {line}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Content area */}
      {status === 'unavailable' ? (
        <div style={{
          flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: 16, color: 'var(--text-muted)',
        }}>
          <BookOpen size={48} style={{ opacity: 0.25 }} />
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>Jupyter ยังไม่พร้อม</p>
            <p style={{ fontSize: 13, marginBottom: 4 }}>Container กำลัง startup หรือยังไม่ได้รัน</p>
            <p style={{ fontSize: 12, fontFamily: 'var(--font-mono)', background: 'var(--bg-elevated)', padding: '4px 10px', borderRadius: 6, display: 'inline-block' }}>
              docker compose up -d jupyter
            </p>
          </div>
          <button className="btn btn-primary" style={{ display: 'flex', alignItems: 'center', gap: 6 }} onClick={checkJupyter}>
            <RefreshCw size={14} /> ลองใหม่
          </button>
        </div>
      ) : (
        <iframe
          src={JUPYTER_LAB_URL}
          style={{ flex: 1, border: 'none', width: '100%' }}
          allow="clipboard-read; clipboard-write"
          title="Jupyter Notebook"
        />
      )}
    </div>
  )
}
