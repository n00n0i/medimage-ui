import { useState, useEffect } from 'react'
import axios from 'axios'
import { Database, RefreshCw } from 'lucide-react'

const LS_API = '/api/ls'
const LS_TOKEN = '160d2644f4d45f84cd09f8931d20891e52f5e4cf'

interface Storage {
  id: number
  title: string
  bucket: string
  prefix: string
  status: string
  last_sync: string | null
  last_sync_count: number | null
  use_blob_urls: boolean
}

interface Dataset {
  id: number
  name: string
  images: number
  labeled: number
  status: string
  source: string
  lastUpdated: string
}

export default function Datasets() {
  const [loading, setLoading] = useState(true)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [toast, setToast] = useState<{msg: string; type: string} | null>(null)

  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    try {
      const res = await axios.get(`${LS_API}/storages/s3?project=1`, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
      })
      const storages: Storage[] = res.data

      const taskRes = await axios.get(`${LS_API}/tasks?project=1&page_size=200`, {
        headers: { Authorization: `Token ${LS_TOKEN}` },
      })
      const tasks = taskRes.data.tasks || []

      const mapped: Dataset[] = storages.map((s) => {
        const storageTasks = tasks.filter((t: any) => t.storage_id === s.id)
        const labeled = storageTasks.filter((t: any) => t.is_labeled).length
        return {
          id: s.id,
          name: s.title,
          images: storageTasks.length,
          labeled,
          status: s.status,
          source: `${s.bucket}/${s.prefix || ''}`,
          lastUpdated: s.last_sync ? new Date(s.last_sync).toLocaleString('th-TH') : '-',
        }
      })

      setDatasets(mapped)
    } catch (e: any) {
      showToast(`Error: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const triggerSync = async (storageId: number) => {
    showToast('Syncing storage...', 'info')
    try {
      void await axios.post(
        `/api/sync/${storageId}/`,
        {},
        { headers: { Authorization: `Token ${LS_TOKEN}` } }
      )
      showToast(`Sync triggered for storage ${storageId}`, 'success')
      setTimeout(fetchDatasets, 3000)
    } catch (e: any) {
      showToast('Sync not available via API (Redis RQ not running) — refresh to see current status', 'info')
      setTimeout(fetchDatasets, 2000)
    }
  }

  const showToast = (msg: string, type: string) => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  const statusBadge = (s: string) => {
    if (s === 'completed') return 'badge badge-success'
    if (s === 'failed') return 'badge badge-danger'
    if (s === 'pending') return 'badge badge-warning'
    return 'badge badge-primary'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner" />
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Datasets</h1>
          <p className="text-sm text-gray-500">จัดการและ sync ข้อมูลจาก MinIO S3 storage</p>
        </div>
        <button
          className="btn btn-secondary flex items-center gap-2"
          onClick={fetchDatasets}
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {toast && (
        <div className={`toast ${toast.type === 'error' ? 'toast-error' : 'toast-success'} mb-4`}>
          {toast.msg}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {datasets.map((ds) => (
          <div key={ds.id} className="card">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="font-semibold text-white">{ds.name}</h3>
                <p className="text-xs text-gray-500 font-mono mt-1 break-all">{ds.source}</p>
              </div>
              <span className={statusBadge(ds.status)}>{ds.status}</span>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-gray-900 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-indigo-400">{ds.images}</div>
                <div className="text-xs text-gray-500 mt-1">Images</div>
              </div>
              <div className="bg-gray-900 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-green-400">{ds.labeled}</div>
                <div className="text-xs text-gray-500 mt-1">Labeled</div>
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
              <span>Last sync: {ds.lastUpdated}</span>
              <span className="text-green-400">{ds.images > 0 ? `${Math.round((ds.labeled / ds.images) * 100)}% labeled` : '-'}</span>
            </div>

            <button
              className="btn btn-secondary w-full"
              onClick={() => triggerSync(ds.id)}
              disabled={ds.status === 'started'}
            >
              {ds.status === 'started' ? 'Syncing...' : '⟳ Sync Storage'}
            </button>
          </div>
        ))}
      </div>

      {datasets.length === 0 && (
        <div className="text-center py-16 text-gray-500">
          <Database size={48} className="mx-auto mb-4 opacity-50" />
          <p>ยังไม่มี dataset — ไปที่ Label Studio เพื่อสร้าง S3 storage ก่อน</p>
          <a href="http://100.68.221.236:8080" target="_blank" className="text-indigo-400 text-sm mt-2 inline-block">
            → เปิด Label Studio
          </a>
        </div>
      )}
    </div>
  )
}
