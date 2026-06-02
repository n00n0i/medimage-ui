import { useState, useEffect } from 'react'
import { Activity, Cpu, HardDrive, Loader } from 'lucide-react'

interface ClusterStatus {
  status: string
  cpus: number
  total_cpus: number
  gpus: number
  total_gpus: number
  memory_gb: number
  memory_total_gb: number
  autoscaling_status: string
  dashboard_url: string
  ray_head: string
}

interface ClusterNode {
  node_id: string
  alive: boolean
  cpu: number
  gpu: number
  memory: number
  node_name: string
  resources: Record<string, number>
}

interface NodesResponse {
  nodes: ClusterNode[]
  total: number
  active: number
}

export default function RayCluster() {
  const [status, setStatus] = useState<ClusterStatus | null>(null)
  const [nodes, setNodes] = useState<NodesResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/cluster/status').then(r => r.json()),
      fetch('/api/cluster/nodes').then(r => r.json()),
    ])
      .then(([s, n]) => {
        if (s.status !== 'error') setStatus(s)
        if (n.nodes) setNodes(n)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <Loader className="animate-spin text-indigo-400" size={32} />
    </div>
  )

  const isConnected = status && status.status !== 'error'

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Ray Cluster</h1>
          <p className="text-gray-400 text-sm">Monitor cluster nodes, resource utilisation, and Ray actors.</p>
        </div>
        <div className="flex gap-3">
          <a href={status?.dashboard_url || 'http://100.68.53.118:8265'} target="_blank" rel="noopener noreferrer">
            <button className="btn btn-secondary text-sm">Ray Dashboard</button>
          </a>
        </div>
      </div>

      {error && (
        <div className="card border border-red-900 bg-red-950/30 mb-6">
          <p className="text-red-400 text-sm">Failed to connect to cluster: {error}</p>
        </div>
      )}

      {!isConnected ? (
        <div className="card text-center py-16">
          <Activity className="mx-auto text-gray-600 mb-4" size={48} />
          <h3 className="text-xl font-semibold text-white mb-2">No cluster connected</h3>
          <p className="text-gray-500 text-sm mb-6">
            Connect to a Ray head node to start monitoring cluster resources and managing Ray actors.
          </p>
        </div>
      ) : (
        <>
          {/* Stat Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard
              icon={<Cpu size={20} className="text-indigo-400" />}
              label="TOTAL CPUs"
              value={status.total_cpus}
              sub={`${status.cpus} in use`}
              color="indigo"
            />
            <StatCard
              icon={<Activity size={20} className="text-green-400" />}
              label="TOTAL GPUs"
              value={status.total_gpus}
              sub={`${status.gpus} in use`}
              color="green"
            />
            <StatCard
              icon={<HardDrive size={20} className="text-cyan-400" />}
              label="TOTAL MEMORY"
              value={`${Math.round(status.memory_total_gb)} GB`}
              sub={`${Math.round(status.memory_gb)} GB used`}
              color="cyan"
            />
            <StatCard
              icon={<Activity size={20} className="text-amber-400" />}
              label="ACTIVE NODES"
              value={nodes?.active ?? 0}
              sub={`${nodes?.total ?? 0} total`}
              color="amber"
            />
          </div>

          {/* GPU Bar */}
          <div className="card mb-6">
            <h3 className="font-semibold text-white mb-4">GPU Utilisation</h3>
            <div className="mb-2 flex justify-between text-sm">
              <span className="text-gray-400">{status.gpus} / {status.total_gpus} GPUs</span>
              <span className="text-indigo-400 font-medium">{Math.round((status.gpus / status.total_gpus) * 100)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{
                width: `${(status.gpus / status.total_gpus) * 100}%`,
                background: 'linear-gradient(90deg, #6366f1, #34d399)',
              }} />
            </div>
          </div>

          {/* Node List */}
          {nodes?.nodes && nodes.nodes.length > 0 && (
            <div className="card">
              <h3 className="font-semibold text-white mb-4">Nodes</h3>
              <div className="space-y-3">
                {nodes.nodes.map((node) => (
                  <div key={node.node_id} className="flex items-center gap-4 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                    <div className={`w-3 h-3 rounded-full ${node.alive ? 'bg-green-500' : 'bg-red-500'}`} />
                    <div className="flex-1">
                      <div className="font-mono text-sm text-white">{node.node_id.slice(0, 16)}...</div>
                      <div className="text-xs text-gray-500">{node.node_name || 'Worker Node'}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-white">{node.cpu} CPUs</div>
                      <div className="text-xs text-gray-500">{node.gpu} GPUs</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Ray Head Info */}
          <div className="mt-4 p-3 rounded-lg bg-gray-900/50 border border-gray-800 flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs text-gray-400 font-mono">
              Head: {status.ray_head} | Dashboard: {status.dashboard_url}
            </span>
          </div>
        </>
      )}
    </div>
  )
}

function StatCard({ icon, label, value, sub, color }: {
  icon: React.ReactNode
  label: string
  value: string | number
  sub: string
  color: string
}) {
  const colors: Record<string, string> = {
    indigo: 'border-indigo-900/50',
    green: 'border-green-900/50',
    cyan: 'border-cyan-900/50',
    amber: 'border-amber-900/50',
  }
  return (
    <div className={`card border ${colors[color] ?? colors.indigo} flex flex-col`}>
      <div className="flex items-center gap-2 mb-3">{icon}</div>
      <div className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-1">{label}</div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      <div className="text-xs text-gray-500 mt-auto">{sub}</div>
    </div>
  )
}
