import { useState, useEffect, useRef, useCallback } from 'react'
import {
  HardDrive, FolderOpen, Upload, Trash2, Copy, ExternalLink, RefreshCw,
  Plus, ChevronRight, ChevronDown, ChevronUp, Folder, FileText, FileImage, File, Settings, X, Check,
  AlertTriangle, Layers, Database, Search, ArrowLeft, Eraser, Pencil,
} from 'lucide-react'
import {
  listBuckets, listObjects, createBucket as s3CreateBucket,
  deleteBucket as s3DeleteBucket, deleteObject as s3DeleteObject,
  copyObject as s3CopyObject, uploadObject as s3UploadObject, fetchDiskInfo,
  loadMinioConfig, saveMinioConfig, MINIO_DEFAULTS,
  type ObjectInfo, type DiskInfo,
} from '../lib/minioClient'

/* ── Types ──────────────────────────────────────────────────── */
interface Bucket {
  name: string
  creationDate: string
  objectCount: number | null   // null while loading
  sizeBytes: number | null
}

type ModalState = 'none' | 'settings' | 'create-bucket' | 'upload' | 'delete-bucket' | 'delete-object' | 'clear-bucket' | 'rename-bucket'

/* ── Helpers ─────────────────────────────────────────────────── */
function fmtSize(bytes: number) {
  if (bytes >= 1e9)  return `${(bytes / 1e9).toFixed(2)} GB`
  if (bytes >= 1e6)  return `${(bytes / 1e6).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

function fileIcon(key: string) {
  const ext = key.split('.').pop()?.toLowerCase() ?? ''
  const img = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'dcm', 'dicom', 'nii', 'nrrd', 'mha']
  const txt = ['txt', 'json', 'csv', 'xml', 'yaml', 'yml', 'md', 'log']
  if (img.includes(ext)) return <FileImage size={14} style={{ color: 'var(--primary-hover)' }} />
  if (txt.includes(ext)) return <FileText size={14} style={{ color: 'var(--success)' }} />
  return <File size={14} style={{ color: 'var(--text-muted)' }} />
}

function relTime(iso: string) {
  try {
    const diff = (Date.now() - new Date(iso).getTime()) / 1000
    if (diff < 60)     return 'just now'
    if (diff < 3600)   return `${Math.round(diff / 60)}m ago`
    if (diff < 86400)  return `${Math.round(diff / 3600)}h ago`
    return `${Math.round(diff / 86400)}d ago`
  } catch { return iso }
}

/* ── Sub-components ──────────────────────────────────────────── */
function StatCard({ label, value, sub, icon }: {
  label: string; value: string | number; sub?: React.ReactNode; icon: React.ReactNode
}) {
  return (
    <div className="stat-card">
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', marginBottom: 6 }}>
        {icon}
        <span className="stat-label">{label}</span>
      </div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

/* ── Main Component ──────────────────────────────────────────── */
export default function Storage() {
  const [buckets, setBuckets]         = useState<Bucket[]>([])
  const [loading, setLoading]         = useState(true)
  const [connected, setConnected]     = useState(false)
  const [errorMsg, setErrorMsg]       = useState<string | null>(null)
  const [selectedBucket, setSelectedBucket] = useState<string | null>(null)
  const [objects, setObjects]         = useState<ObjectInfo[]>([])
  const [objLoading, setObjLoading]   = useState(false)
  const [prefix, setPrefix]           = useState('')
  const [search, setSearch]           = useState('')
  const cfg = loadMinioConfig()
  const [consoleUrl, setConsoleUrl]   = useState(cfg.consoleUrl)
  const [modal, setModal]             = useState<ModalState>('none')
  const [newBucketName, setNewBucketName] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<{ bucket: string; key?: string } | null>(null)
  const [uploading, setUploading]     = useState(false)
  const [uploadBucket, setUploadBucket] = useState<string | null>(null)
  const [toast, setToast]             = useState<{ msg: string; type: 'ok' | 'err' } | null>(null)
  const [copiedKey, setCopiedKey]     = useState<string | null>(null)
  const [settingsForm, setSettingsForm] = useState({ ...MINIO_DEFAULTS, ...loadMinioConfig() })
  const [diskInfo, setDiskInfo]       = useState<DiskInfo | null>(null)
  const [datasetBuckets, setDatasetBuckets] = useState<Set<string>>(new Set())
  const [clearing, setClearing]       = useState(false)
  const [renamingBucket, setRenamingBucket] = useState<string | null>(null)
  const [renameNewName, setRenameNewName] = useState('')
  const [renameLoading, setRenameLoading] = useState(false)
  const [expandedBuckets, setExpandedBuckets] = useState<Set<string>>(new Set())
  const [bucketDirs, setBucketDirs]   = useState<Record<string, string[]>>({})
  const [dirsLoading, setDirsLoading] = useState<Set<string>>(new Set())
  const fileRef                       = useRef<HTMLInputElement>(null)

  /* ── Toast helper ─────────────────────────────────────────── */
  function showToast(msg: string, type: 'ok' | 'err' = 'ok') {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  /* ── Fetch buckets ────────────────────────────────────────── */
  const fetchBuckets = useCallback(async () => {
    try {
      const raw = await listBuckets()
      setBuckets(raw.map(b => ({ name: b.name, creationDate: b.creationDate, objectCount: null, sizeBytes: null })))
      setConnected(true)
      setErrorMsg(null)
      // Fetch disk info from admin API
      fetchDiskInfo().then(d => setDiskInfo(d))
      // Lazy-load object counts in background
      raw.forEach(async (b) => {
        try {
          const objs = await listObjects(b.name)
          const totalBytes = objs.reduce((s, o) => s + o.size, 0)
          setBuckets(prev => prev.map(x => x.name === b.name
            ? { ...x, objectCount: objs.length, sizeBytes: totalBytes }
            : x
          ))
        } catch { /* ignore per-bucket errors */ }
      })
    } catch (e: any) {
      setConnected(false)
      // Parse S3 XML error if available
      const msg: string = e?.message ?? ''
      if (msg.includes('InvalidAccessKeyId')) setErrorMsg('Invalid access key — update credentials in Settings (⚙)')
      else if (msg.includes('AccessDenied') || msg.includes('403')) setErrorMsg('Access denied — check access key permissions')
      else if (msg.includes('NetworkError') || msg.includes('Failed to fetch')) setErrorMsg('Cannot reach MinIO — check network or API URL')
      else setErrorMsg(msg || 'Cannot connect to MinIO')
    } finally {
      setLoading(false)
    }
  }, [])

  /* ── Fetch objects ────────────────────────────────────────── */
  const fetchObjects = useCallback(async (bucket: string, p = '') => {
    setObjLoading(true)
    try {
      const data = await listObjects(bucket, p)
      setObjects(data)
    } catch {
      setObjects([])
    } finally {
      setObjLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchBuckets()
    const id = setInterval(fetchBuckets, 30000)
    return () => clearInterval(id)
  }, [fetchBuckets])

  // Fetch bucket names used by Label Studio datasets, then auto-create any missing ones in MinIO
  useEffect(() => {
    fetch('/api/ls/storages/s3?project=1', {
      headers: { Authorization: 'Token 160d2644f4d45f84cd09f8931d20891e52f5e4cf' },
    })
      .then(r => r.ok ? r.json() : [])
      .then(async (storages: any[]) => {
        // Derive intended bucket from title (e.g. "medimage-cxr-bucket" → "medimage-cxr")
        const deriveBucket = (s: any): string =>
          (s.title as string).replace(/-bucket$/i, '').toLowerCase() || (s.bucket as string)
        const bucketNames: string[] = [...new Set(
          storages.flatMap((s: any) => [s.bucket, deriveBucket(s)]).filter(Boolean)
        )]
        setDatasetBuckets(new Set(bucketNames))
        if (bucketNames.length === 0) return
        try {
          const existing = await listBuckets()
          const existingNames = new Set(existing.map(b => b.name))
          const missing = bucketNames.filter(n => !existingNames.has(n))
          if (missing.length > 0) {
            const results = await Promise.allSettled(missing.map(n => s3CreateBucket(n)))
            const created = missing.filter((_, i) => results[i].status === 'fulfilled')
            fetchBuckets()
            if (created.length > 0)
              showToast(`สร้าง ${created.length} bucket จาก Dataset: ${created.join(', ')}`)
          }
        } catch { /* silent — MinIO may be unreachable */ }
      })
      .catch(() => {})
  }, [fetchBuckets])

  // Auto-open settings when credentials are clearly wrong
  useEffect(() => {
    if (errorMsg?.includes('Invalid access key') && modal === 'none') {
      setSettingsForm({ ...loadMinioConfig() })
      setModal('settings')
    }
  }, [errorMsg])

  function openBucket(name: string) {
    setSelectedBucket(name)
    setPrefix('')
    setSearch('')
    fetchObjects(name, '')
  }

  function closeBucket() {
    setSelectedBucket(null)
    setObjects([])
    setPrefix('')
    setSearch('')
  }

  /* ── Toggle bucket directory expand ─────────────────────── */
  async function toggleBucketExpand(name: string) {
    if (expandedBuckets.has(name)) {
      setExpandedBuckets(prev => { const s = new Set(prev); s.delete(name); return s })
      return
    }
    setExpandedBuckets(prev => new Set([...prev, name]))
    if (bucketDirs[name]) return // already loaded
    setDirsLoading(prev => new Set([...prev, name]))
    try {
      const objs = await listObjects(name)
      // Extract unique top-level prefixes (directories)
      const dirs = [...new Set(
        objs
          .map(o => o.key.includes('/') ? o.key.split('/')[0] : null)
          .filter(Boolean) as string[]
      )].sort()
      // Files at root (no directory)
      setBucketDirs(prev => ({ ...prev, [name]: dirs }))
    } catch {
      setBucketDirs(prev => ({ ...prev, [name]: [] }))
    } finally {
      setDirsLoading(prev => { const s = new Set(prev); s.delete(name); return s })
    }
  }

  /* ── Create bucket ────────────────────────────────────────── */
  async function createBucket() {
    if (!newBucketName.trim()) return
    try {
      await s3CreateBucket(newBucketName.trim())
      showToast(`Bucket "${newBucketName.trim()}" created`)
      setNewBucketName('')
      setModal('none')
      fetchBuckets()
    } catch (e: any) {
      showToast(e?.message ?? 'Failed to create bucket', 'err')
    }
  }

  /* ── Rename bucket ──────────────────────────────────── */
  async function handleRenameBucket() {
    if (!renamingBucket) return
    const newName = renameNewName.trim().toLowerCase().replace(/[^a-z0-9.-]/g, '-')
    if (!newName || newName === renamingBucket) return
    setRenameLoading(true)
    try {
      await s3CreateBucket(newName)
      const objs = await listObjects(renamingBucket)
      if (objs.length > 0) {
        await Promise.all(objs.map(o => s3CopyObject(renamingBucket, o.key, newName, o.key)))
        await Promise.all(objs.map(o => s3DeleteObject(renamingBucket, o.key)))
      }
      await s3DeleteBucket(renamingBucket)
      showToast(`Bucket renamed to "${newName}"`)
      if (selectedBucket === renamingBucket) closeBucket()
      setModal('none')
      setRenamingBucket(null)
      setRenameNewName('')
      fetchBuckets()
    } catch (e: any) {
      showToast(e?.message ?? 'Failed to rename bucket', 'err')
    } finally {
      setRenameLoading(false)
    }
  }

  /* ── Clear bucket (remove all objects, keep bucket) ────────── */
  async function clearBucket(name: string) {
    setClearing(true)
    try {
      const objs = await listObjects(name)
      await Promise.all(objs.map(o => s3DeleteObject(name, o.key)))
      showToast(`Cleared ${objs.length} object${objs.length !== 1 ? 's' : ''} from "${name}"`)
      setModal('none')
      setDeleteTarget(null)
      fetchBuckets()
      if (selectedBucket === name) fetchObjects(name, prefix)
    } catch (e: any) {
      showToast(e?.message ?? 'Failed to clear bucket', 'err')
    } finally {
      setClearing(false)
    }
  }

  /* ── Delete bucket ────────────────────────────────────────── */
  async function deleteBucket(name: string) {
    try {      // S3 requires empty bucket before deletion — auto-clear first
      const objs = await listObjects(name)
      if (objs.length > 0) {
        await Promise.all(objs.map(o => s3DeleteObject(name, o.key)))
      }      await s3DeleteBucket(name)
      showToast(`Bucket "${name}" deleted`)
      if (selectedBucket === name) closeBucket()
      setModal('none')
      setDeleteTarget(null)
      fetchBuckets()
    } catch (e: any) {
      showToast(e?.message ?? 'Failed to delete bucket', 'err')
    }
  }

  /* ── Delete object ────────────────────────────────────────── */
  async function deleteObject(bucket: string, key: string) {
    try {
      await s3DeleteObject(bucket, key)
      showToast(`Deleted "${key}"`)
      setModal('none')
      setDeleteTarget(null)
      fetchObjects(bucket, prefix)
    } catch (e: any) {
      showToast(e?.message ?? 'Failed to delete object', 'err')
    }
  }

  /* ── Upload ───────────────────────────────────────────────── */
  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? [])
    if (!files.length || !uploadBucket) return
    setUploading(true)
    try {
      for (const f of files) {
        await s3UploadObject(uploadBucket, f.name, f)
      }
      showToast(`Uploaded ${files.length} file${files.length > 1 ? 's' : ''}`)
      fetchBuckets()
      if (selectedBucket === uploadBucket) fetchObjects(uploadBucket, prefix)
    } catch (e: any) {
      showToast(e?.message ?? 'Upload failed', 'err')
    } finally {
      setUploading(false)
      setModal('none')
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  /* ── Copy URL ─────────────────────────────────────────────── */
  function copyUrl(bucket: string, key: string) {
    const url = `${consoleUrl.replace(/\/$/, '')}/buckets/${bucket}/objects/${encodeURIComponent(key)}`
    navigator.clipboard.writeText(url).catch(() => {})
    setCopiedKey(key)
    setTimeout(() => setCopiedKey(null), 2000)
  }

  /* ── Save settings ────────────────────────────────────────── */
  function saveSettings() {
    saveMinioConfig(settingsForm)
    setConsoleUrl(settingsForm.consoleUrl)
    setModal('none')
    showToast('Settings saved — reconnecting…')
    setBuckets([])
    setConnected(false)
    setErrorMsg(null)
    setTimeout(() => fetchBuckets(), 100)
  }

  /* ── Derived ──────────────────────────────────────────────── */
  const totalObjects  = buckets.reduce((s, b) => s + (b.objectCount ?? 0), 0)
  const totalSizeBytes = buckets.reduce((s, b) => s + (b.sizeBytes ?? 0), 0)
  const filteredObjs  = objects.filter(o => o.key.toLowerCase().includes(search.toLowerCase()))

  /* ── Render ──────────────────────────────────────────────────*/
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner" />
      </div>
    )
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto' }}>
      {/* Hidden file input */}
      <input
        ref={fileRef}
        type="file"
        multiple
        style={{ display: 'none' }}
        onChange={handleUpload}
        accept="*/*"
      />

      {/* ── Header ─────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {selectedBucket ? (
            <>
              <button
                onClick={closeBucket}
                style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 13, padding: 0 }}
              >
                <ArrowLeft size={15} />
                Storage
              </button>
              <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />
              <span style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>
                {selectedBucket}
              </span>
            </>
          ) : (
            <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>
              Storage
            </h1>
          )}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 5, fontSize: 11,
            color: connected ? 'var(--success)' : 'var(--danger)',
            background: connected ? 'var(--success-dim)' : 'var(--danger-dim)',
            border: `1px solid ${connected ? 'var(--success)' : 'var(--danger)'}`,
            borderRadius: 20, padding: '2px 8px',
          }}>
            <div style={{ width: 5, height: 5, borderRadius: '50%', background: 'currentColor' }} />
            {connected ? 'Connected' : 'Offline'}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          {selectedBucket && (
            <button
              className="btn btn-primary"
              style={{ fontSize: 12, padding: '5px 12px' }}
              onClick={() => { setUploadBucket(selectedBucket); setModal('upload') }}
            >
              <Upload size={13} />Upload
            </button>
          )}
          <button
            className="btn btn-secondary"
            style={{ fontSize: 12, padding: '5px 12px' }}
            onClick={() => selectedBucket ? fetchObjects(selectedBucket, prefix) : fetchBuckets()}
          >
            <RefreshCw size={13} />Refresh
          </button>
          <a
            href={consoleUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary"
            style={{ fontSize: 12, padding: '5px 12px' }}
          >
            <ExternalLink size={13} />Console
          </a>
          <button
            className="btn btn-secondary"
            style={{ fontSize: 12, padding: '5px 10px' }}
            onClick={() => { setSettingsForm({ ...loadMinioConfig() }); setModal('settings') }}
          >
            <Settings size={13} />
          </button>
        </div>
      </div>

      {/* ── Stats (only on bucket list view) ───────────────── */}
      {!selectedBucket && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <StatCard label="Buckets"      value={buckets.length}            icon={<Layers size={15} />} />
          <StatCard label="Total Objects" value={totalObjects.toLocaleString()} icon={<Database size={15} />} />
          <StatCard
            label="Storage Used"
            value={diskInfo ? `${fmtSize(totalSizeBytes)} / ${fmtSize(diskInfo.totalBytes)}` : fmtSize(totalSizeBytes)}
            sub={diskInfo ? (() => {
              const pct = Math.min(100, (diskInfo.usedBytes / diskInfo.totalBytes) * 100)
              return (
                <div style={{ marginTop: 6 }}>
                  <div style={{ height: 4, background: 'var(--border-subtle)', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${pct.toFixed(1)}%`, background: pct > 85 ? 'var(--danger)' : pct > 60 ? 'oklch(0.75 0.15 60)' : 'var(--primary)', borderRadius: 99, transition: 'width 0.4s' }} />
                  </div>
                  <span style={{ fontSize: 10, marginTop: 3, display: 'block' }}>{pct.toFixed(1)}% used · {fmtSize(diskInfo.freeBytes)} free</span>
                </div>
              ) as any
            })() : undefined}
            icon={<HardDrive size={15} />}
          />
          <StatCard label="MinIO Console" value="Open →"                   icon={<ExternalLink size={15} />}
            sub={consoleUrl.replace(/^https?:\/\//, '')} />
        </div>
      )}

      {/* ── Bucket list ────────────────────────────────────── */}
      {!selectedBucket && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          {/* Toolbar */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 18px', borderBottom: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Buckets</span>
            <button
              className="btn btn-primary"
              style={{ fontSize: 12, padding: '4px 12px' }}
              onClick={() => { setNewBucketName(''); setModal('create-bucket') }}
            >
              <Plus size={13} />New Bucket
            </button>
          </div>

          {buckets.length === 0 ? (
            <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              <HardDrive size={32} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
              {connected ? 'No buckets found — create one to get started' : (
                <div>
                  <div style={{ marginBottom: 12, color: 'var(--danger)', fontWeight: 500 }}>
                    {errorMsg ?? 'Cannot reach MinIO — check backend connection'}
                  </div>
                  <button
                    className="btn btn-primary"
                    style={{ fontSize: 13, padding: '7px 18px' }}
                    onClick={() => { setSettingsForm({ ...loadMinioConfig() }); setModal('settings') }}
                  >
                    <Settings size={14} />Configure Credentials
                  </button>
                </div>
              )}
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  {['Bucket', 'Objects', 'Size', ''].map(h => (
                    <th key={h} style={{ padding: '8px 18px', textAlign: 'left', fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {buckets.map(b => {
                  const isExpanded = expandedBuckets.has(b.name)
                  const isDirLoading = dirsLoading.has(b.name)
                  const dirs = bucketDirs[b.name] ?? []
                  return (
                    <>
                    <tr
                      key={b.name}
                      style={{ borderBottom: isExpanded ? 'none' : '1px solid var(--border-subtle)', cursor: 'pointer', transition: 'background 0.15s' }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-elevated)')}
                      onMouseLeave={e => (e.currentTarget.style.background = '')}
                      onClick={() => openBucket(b.name)}
                    >
                      <td style={{ padding: '12px 18px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <div style={{ width: 30, height: 30, borderRadius: 6, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                            <FolderOpen size={15} style={{ color: 'var(--primary-hover)' }} />
                          </div>
                          <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>{b.name}</span>
                        </div>
                      </td>
                      <td style={{ padding: '12px 18px', fontSize: 13, color: 'var(--text-secondary)' }}>
                        {b.objectCount == null ? <span style={{color:'var(--text-muted)'}}>…</span> : b.objectCount.toLocaleString()}
                      </td>
                      <td style={{ padding: '12px 18px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                        {b.sizeBytes == null ? '…' : fmtSize(b.sizeBytes)}
                      </td>
                      <td style={{ padding: '12px 18px', textAlign: 'right' }} onClick={e => e.stopPropagation()}>
                        <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end', alignItems: 'center' }}>
                          {datasetBuckets.has(b.name) && (
                            <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--primary-hover)', background: 'var(--primary-dim)', border: '1px solid var(--primary)', borderRadius: 20, padding: '1px 7px', flexShrink: 0 }}>
                              Dataset
                            </span>
                          )}
                          <button
                            title="Rename bucket"
                            className="btn btn-secondary"
                            style={{ fontSize: 11, padding: '3px 8px' }}
                            onClick={() => { setRenamingBucket(b.name); setRenameNewName(b.name); setModal('rename-bucket') }}
                          >
                            <Pencil size={11} />
                          </button>
                          {datasetBuckets.has(b.name) ? (
                            <button
                              title="Clear all objects (bucket is used by a Dataset)"
                              className="btn btn-secondary"
                              style={{ fontSize: 11, padding: '3px 8px', color: 'var(--warning, oklch(0.75 0.15 60))' }}
                              onClick={() => { setDeleteTarget({ bucket: b.name }); setModal('clear-bucket') }}
                            >
                              <Eraser size={11} />
                            </button>
                          ) : (
                            <button
                              title="Delete bucket"
                              className="btn btn-secondary"
                              style={{ fontSize: 11, padding: '3px 8px', color: 'var(--danger)' }}
                              onClick={() => { setDeleteTarget({ bucket: b.name }); setModal('delete-bucket') }}
                            >
                              <Trash2 size={11} />
                            </button>
                          )}
                          <button
                            title={isExpanded ? 'Collapse' : 'Show directories'}
                            className="btn btn-secondary"
                            style={{ fontSize: 11, padding: '3px 8px' }}
                            onClick={e => { e.stopPropagation(); toggleBucketExpand(b.name) }}
                          >
                            {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                          </button>
                        </div>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr key={`${b.name}-dirs`} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                        <td colSpan={4} style={{ padding: '0 18px 14px 58px', background: 'var(--bg-elevated)' }}>
                          {isDirLoading ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 0', color: 'var(--text-muted)', fontSize: 12 }}>
                              <div className="loading-spinner" style={{ width: 12, height: 12 }} />Loading directories…
                            </div>
                          ) : dirs.length === 0 ? (
                            <div style={{ padding: '10px 0', fontSize: 12, color: 'var(--text-muted)' }}>No directories (all files at root)</div>
                          ) : (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, paddingTop: 10 }}>
                              {dirs.map(dir => (
                                <button
                                  key={dir}
                                  className="btn btn-secondary"
                                  style={{ fontSize: 11, padding: '4px 10px', fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: 5 }}
                                  onClick={() => { openBucket(b.name); setPrefix(dir + '/'); fetchObjects(b.name, dir + '/') }}
                                >
                                  <Folder size={11} style={{ color: 'var(--primary-hover)' }} />{dir}/
                                </button>
                              ))}
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                    </>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* ── Object browser ─────────────────────────────────── */}
      {selectedBucket && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          {/* Toolbar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 18px', borderBottom: '1px solid var(--border-subtle)' }}>
            <div style={{ position: 'relative', flex: 1, maxWidth: 320 }}>
              <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Filter objects…"
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="input"
                style={{ paddingLeft: 30, fontSize: 12, height: 30 }}
              />
            </div>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
              {filteredObjs.length} object{filteredObjs.length !== 1 ? 's' : ''}
            </span>
          </div>

          {objLoading ? (
            <div style={{ padding: 48, textAlign: 'center' }}>
              <div className="loading-spinner" style={{ margin: '0 auto' }} />
            </div>
          ) : filteredObjs.length === 0 ? (
            <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              <File size={28} style={{ margin: '0 auto 10px', opacity: 0.4 }} />
              {search ? 'No objects match the filter' : 'This bucket is empty — upload files to get started'}
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  {['Name', 'Size', 'Modified', ''].map(h => (
                    <th key={h} style={{ padding: '8px 18px', textAlign: 'left', fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredObjs.map(obj => (
                  <tr
                    key={obj.key}
                    style={{ borderBottom: '1px solid var(--border-subtle)', transition: 'background 0.15s' }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-elevated)')}
                    onMouseLeave={e => (e.currentTarget.style.background = '')}
                  >
                    <td style={{ padding: '10px 18px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        {fileIcon(obj.key)}
                        <span style={{ fontSize: 12.5, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', maxWidth: 360, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {obj.key}
                        </span>
                      </div>
                    </td>
                    <td style={{ padding: '10px 18px', fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
                      {fmtSize(obj.size)}
                    </td>
                    <td style={{ padding: '10px 18px', fontSize: 12, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                      {relTime(obj.lastModified)}
                    </td>
                    <td style={{ padding: '10px 18px', textAlign: 'right' }}>
                      <div style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                        <button
                          title="Copy URL"
                          className="btn btn-secondary"
                          style={{ fontSize: 11, padding: '3px 8px' }}
                          onClick={() => copyUrl(selectedBucket, obj.key)}
                        >
                          {copiedKey === obj.key ? <Check size={11} style={{ color: 'var(--success)' }} /> : <Copy size={11} />}
                        </button>
                        <button
                          title="Delete"
                          className="btn btn-secondary"
                          style={{ fontSize: 11, padding: '3px 8px', color: 'var(--danger)' }}
                          onClick={() => { setDeleteTarget({ bucket: selectedBucket, key: obj.key }); setModal('delete-object') }}
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* ── Modals ─────────────────────────────────────────────*/}
      {modal !== 'none' && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'oklch(0 0 0 / 0.6)', zIndex: 'var(--z-modal-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onClick={() => setModal('none')}
        >
          <div
            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 12, padding: 24, minWidth: 360, maxWidth: 440, width: '90vw', boxShadow: 'var(--shadow-md)' }}
            onClick={e => e.stopPropagation()}
          >
            {/* Settings */}
            {modal === 'settings' && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Storage Settings</h3>
                  <button onClick={() => setModal('none')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}><X size={16} /></button>
                </div>
                {(['apiUrl', 'accessKey', 'secretKey', 'consoleUrl'] as const).map(field => (
                  <div key={field} style={{ marginBottom: 14 }}>
                    <label style={{ display: 'block', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4, fontWeight: 500 }}>
                      {{ apiUrl: 'MinIO API URL', accessKey: 'Access Key', secretKey: 'Secret Key', consoleUrl: 'Console URL' }[field]}
                    </label>
                    <input
                      className="input"
                      style={{ width: '100%', fontSize: 13 }}
                      type={field === 'secretKey' ? 'password' : 'text'}
                      value={settingsForm[field]}
                      onChange={e => setSettingsForm(f => ({ ...f, [field]: e.target.value }))}
                      placeholder={MINIO_DEFAULTS[field]}
                    />
                  </div>
                ))}
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 18 }}>
                  <button className="btn btn-secondary" onClick={() => { setSettingsForm({ ...MINIO_DEFAULTS }); }}>Reset</button>
                  <button className="btn btn-secondary" onClick={() => setModal('none')}>Cancel</button>
                  <button className="btn btn-primary" onClick={saveSettings}>Save</button>
                </div>
              </>
            )}

            {/* Create bucket */}
            {modal === 'create-bucket' && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Create Bucket</h3>
                  <button onClick={() => setModal('none')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}><X size={16} /></button>
                </div>
                <label style={{ display: 'block', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6, fontWeight: 500 }}>Bucket Name</label>
                <input
                  className="input"
                  style={{ width: '100%', marginBottom: 18, fontSize: 13 }}
                  value={newBucketName}
                  onChange={e => setNewBucketName(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))}
                  placeholder="my-bucket"
                  onKeyDown={e => e.key === 'Enter' && createBucket()}
                  autoFocus
                />
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                  <button className="btn btn-secondary" onClick={() => setModal('none')}>Cancel</button>
                  <button className="btn btn-primary" onClick={createBucket} disabled={!newBucketName.trim()}>Create</button>
                </div>
              </>
            )}

            {/* Upload */}
            {modal === 'upload' && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Upload to <span style={{ color: 'var(--primary-hover)' }}>{uploadBucket}</span></h3>
                  <button onClick={() => setModal('none')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}><X size={16} /></button>
                </div>
                <div
                  style={{ border: '2px dashed var(--border-default)', borderRadius: 8, padding: 32, textAlign: 'center', cursor: 'pointer', transition: 'border-color 0.2s', marginBottom: 18 }}
                  onClick={() => fileRef.current?.click()}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--primary)')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border-default)')}
                >
                  {uploading ? (
                    <div className="loading-spinner" style={{ margin: '0 auto 8px' }} />
                  ) : (
                    <Upload size={24} style={{ margin: '0 auto 8px', color: 'var(--text-muted)', display: 'block' }} />
                  )}
                  <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                    {uploading ? 'Uploading…' : 'Click to select files'}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>Any file type supported</div>
                </div>
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                  <button className="btn btn-secondary" onClick={() => setModal('none')}>Cancel</button>
                </div>
              </>
            )}

            {/* Delete bucket */}
            {modal === 'delete-bucket' && deleteTarget && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                  <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--danger-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <AlertTriangle size={18} style={{ color: 'var(--danger)' }} />
                  </div>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Delete Bucket</h3>
                </div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 18, lineHeight: 1.5 }}>
                  Delete <strong style={{ color: 'var(--text-primary)' }}>{deleteTarget.bucket}</strong>? This will remove all objects inside. This cannot be undone.
                </p>
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                  <button className="btn btn-secondary" onClick={() => setModal('none')}>Cancel</button>
                  <button className="btn btn-primary" style={{ background: 'var(--danger)', borderColor: 'var(--danger)' }} onClick={() => deleteBucket(deleteTarget.bucket)}>Delete</button>
                </div>
              </>
            )}

            {/* Clear bucket */}
            {modal === 'clear-bucket' && deleteTarget && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                  <div style={{ width: 36, height: 36, borderRadius: 8, background: 'oklch(0.25 0.05 60)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <Eraser size={18} style={{ color: 'oklch(0.75 0.15 60)' }} />
                  </div>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Clear Bucket</h3>
                </div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4, lineHeight: 1.5 }}>
                  Remove all objects from <strong style={{ color: 'var(--text-primary)' }}>{deleteTarget.bucket}</strong>?
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 18, lineHeight: 1.5 }}>
                  This bucket is linked to a Dataset. The bucket itself will be kept — only its contents will be deleted.
                </p>
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                  <button className="btn btn-secondary" onClick={() => setModal('none')}>Cancel</button>
                  <button
                    className="btn btn-primary"
                    style={{ background: 'oklch(0.55 0.15 60)', borderColor: 'oklch(0.55 0.15 60)' }}
                    disabled={clearing}
                    onClick={() => clearBucket(deleteTarget.bucket)}
                  >
                    {clearing ? <><div className="loading-spinner" style={{ width: 12, height: 12 }} />Clearing…</> : <><Eraser size={13} />Clear All Objects</>}
                  </button>
                </div>
              </>
            )}

            {/* Rename bucket */}
            {modal === 'rename-bucket' && renamingBucket && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Rename Bucket</h3>
                  <button onClick={() => setModal('none')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}><X size={16} /></button>
                </div>
                <div style={{ marginBottom: 14 }}>
                  <label style={{ display: 'block', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4, fontWeight: 500 }}>Current Name</label>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-muted)', padding: '8px 12px', background: 'var(--bg-surface)', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>{renamingBucket}</div>
                </div>
                <label style={{ display: 'block', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6, fontWeight: 500 }}>New Name</label>
                <input
                  className="input"
                  style={{ width: '100%', marginBottom: 6, fontSize: 13 }}
                  value={renameNewName}
                  onChange={e => setRenameNewName(e.target.value.toLowerCase().replace(/[^a-z0-9.-]/g, '-'))}
                  placeholder="new-bucket-name"
                  onKeyDown={e => e.key === 'Enter' && !renameLoading && handleRenameBucket()}
                  autoFocus
                />
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 18 }}>All objects will be copied to the new bucket. This may take a moment.</p>
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                  <button className="btn btn-secondary" onClick={() => setModal('none')}>Cancel</button>
                  <button className="btn btn-primary" onClick={handleRenameBucket}
                    disabled={renameLoading || !renameNewName.trim() || renameNewName.trim() === renamingBucket}>
                    {renameLoading ? <><div className="loading-spinner" style={{ width: 12, height: 12 }} />Renaming…</> : <><Pencil size={13} />Rename</>}
                  </button>
                </div>
              </>
            )}

            {/* Delete object */}
            {modal === 'delete-object' && deleteTarget?.key && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                  <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--danger-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <AlertTriangle size={18} style={{ color: 'var(--danger)' }} />
                  </div>
                  <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Delete Object</h3>
                </div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 18, lineHeight: 1.5 }}>
                  Delete <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>{deleteTarget.key}</strong>? This cannot be undone.
                </p>
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                  <button className="btn btn-secondary" onClick={() => setModal('none')}>Cancel</button>
                  <button className="btn btn-primary" style={{ background: 'var(--danger)', borderColor: 'var(--danger)' }} onClick={() => deleteObject(deleteTarget.bucket, deleteTarget.key!)}>Delete</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Toast ──────────────────────────────────────────────*/}
      {toast && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24, zIndex: 'var(--z-toast)',
          background: toast.type === 'ok' ? 'var(--success-dim)' : 'var(--danger-dim)',
          border: `1px solid ${toast.type === 'ok' ? 'var(--success)' : 'var(--danger)'}`,
          color: toast.type === 'ok' ? 'var(--success)' : 'var(--danger)',
          borderRadius: 8, padding: '10px 16px', fontSize: 13,
          display: 'flex', alignItems: 'center', gap: 8,
          boxShadow: 'var(--shadow-md)',
          animation: 'fadeIn 0.2s ease',
        }}>
          {toast.type === 'ok' ? <Check size={14} /> : <AlertTriangle size={14} />}
          {toast.msg}
        </div>
      )}
    </div>
  )
}
