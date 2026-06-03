import { useState, useEffect } from 'react'
import { BookOpen, ExternalLink, Loader, AlertCircle, RefreshCw } from 'lucide-react'

export default function Notebook() {
  const [status, setStatus] = useState<'checking' | 'ready' | 'unavailable'>('checking')
  const JUPYTER_URL = '/jupyter/'

  useEffect(() => {
    checkJupyter()
  }, [])

  const checkJupyter = async () => {
    setStatus('checking')
    try {
      const res = await fetch(JUPYTER_URL, { method: 'HEAD' })
      setStatus(res.ok ? 'ready' : 'unavailable')
    } catch {
      setStatus('unavailable')
    }
  }

  return (
    <div style={{ height: 'calc(100vh - 48px)', display: 'flex', flexDirection: 'column', margin: '-24px' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, padding: '10px 20px',
        background: 'var(--bg-surface)', borderBottom: '1px solid var(--border-default)',
        flexShrink: 0,
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

        <div style={{ flex: 1 }} />

        {status === 'unavailable' && (
          <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 5 }} onClick={checkJupyter}>
            <RefreshCw size={12} /> Retry
          </button>
        )}
        <a
          href={JUPYTER_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-secondary btn-sm"
          style={{ display: 'flex', alignItems: 'center', gap: 5 }}
        >
          <ExternalLink size={12} /> Open in new tab
        </a>
      </div>

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
          src={JUPYTER_URL}
          style={{ flex: 1, border: 'none', width: '100%' }}
          allow="clipboard-read; clipboard-write"
          title="Jupyter Notebook"
        />
      )}
    </div>
  )
}
