import { useState, useEffect, useRef } from 'react'
import { Cpu, MemoryStick, Thermometer, Zap } from 'lucide-react'

// ── Types ──────────────────────────────────────────────────────────────────────
interface GpuSnapshot {
  ts: number
  gpus: Array<{
    index: number                // globally unique across clusters (used as history key)
    localIndex: number           // the GPU's own index on its node (0, 1, 2…)
    nodeIndex: number            // which node this GPU is on
    nodeHost: string             // IP/hostname of the node
    cluster: 'on-prem' | 'modal' // which Ray cluster this GPU belongs to
    name: string
    utilGpu: number    // 0-100
    memUsedMiB: number
    memTotalMiB: number
    tempC: number
    powerW: number
  }>
  ramUsedGb: number
  ramTotalGb: number
  nodesTotal: number
}

// ── Sparkline (SVG, no deps) ───────────────────────────────────────────────────
function Sparkline({
  data, color, maxVal = 100, height = 48, width = 160,
}: {
  data: number[]
  color: string
  maxVal?: number
  height?: number
  width?: number
}) {
  const PAD = 2
  const pts = data.slice(-40)
  if (pts.length < 2) return <div style={{ height, width }} />

  const xs = pts.map((_, i) => PAD + (i / (pts.length - 1)) * (width - 2 * PAD))
  const ys = pts.map(v => height - PAD - ((v / maxVal) * (height - 2 * PAD)))

  const line = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  const area = `${line} L${xs[xs.length - 1].toFixed(1)},${height - PAD} L${PAD},${height - PAD} Z`
  const gradId = `g-${color.replace(/[^a-z0-9]/gi, '')}`

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gradId})`} />
      <path d={line} stroke={color} strokeWidth="1.5" fill="none" strokeLinejoin="round" strokeLinecap="round" />
      {/* last point dot */}
      <circle cx={xs[xs.length - 1]} cy={ys[ys.length - 1]} r="2.5" fill={color} />
    </svg>
  )
}

// ── GPU Card ──────────────────────────────────────────────────────────────────
function GpuCard({
  gpu, utilHistory, memHistory,
}: {
  gpu: GpuSnapshot['gpus'][number]
  utilHistory: number[]
  memHistory: number[]
}) {
  const memPct = gpu.memTotalMiB > 0 ? (gpu.memUsedMiB / gpu.memTotalMiB) * 100 : 0
  const memUsedGb = (gpu.memUsedMiB / 1024).toFixed(1)
  const memTotalGb = (gpu.memTotalMiB / 1024).toFixed(0)
  const utilColor = gpu.utilGpu > 80 ? 'var(--danger)' : gpu.utilGpu > 50 ? 'var(--warning)' : 'var(--success)'

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-default)',
      borderRadius: 12,
      padding: '14px 16px',
      minWidth: 0,
    }}>
      {/* GPU name + node host + cluster badge + temp + power */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
            <span style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 4,
              background: gpu.cluster === 'modal' ? '#8b5cf620' : '#f59e0b20',
              color:      gpu.cluster === 'modal' ? '#8b5cf6' : '#f59e0b',
              textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              {gpu.cluster}
            </span>
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              node {gpu.nodeIndex} · {gpu.nodeHost || '?'}
            </span>
          </div>
          <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            GPU {gpu.localIndex} · {gpu.name}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexShrink: 0 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 11, color: 'var(--text-muted)' }} title="Temperature">
            <Thermometer size={11} /> {gpu.tempC}°C
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 11, color: 'var(--text-muted)' }} title="Power draw">
            <Zap size={11} /> {gpu.powerW}W
          </span>
        </div>
      </div>

      {/* Utilization */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
            <Cpu size={11} /> SM Utilization
          </span>
          <span style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-mono)', color: utilColor }}>
            {gpu.utilGpu}%
          </span>
        </div>
        <div style={{ position: 'relative', height: 48 }}>
          <Sparkline data={utilHistory} color={utilColor === 'var(--danger)' ? '#ef4444' : utilColor === 'var(--warning)' ? '#f59e0b' : '#10b981'} maxVal={100} height={48} width={280} />
        </div>
      </div>

      {/* VRAM */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
            <MemoryStick size={11} /> VRAM
          </span>
          {gpu.memTotalMiB > 0 ? (
            <span style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              {memUsedGb} / {memTotalGb} GB
              <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 4 }}>({memPct.toFixed(0)}%)</span>
            </span>
          ) : (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>no metrics</span>
          )}
        </div>
        {gpu.memTotalMiB > 0 && (
          <>
            <div style={{ position: 'relative', height: 36 }}>
              <Sparkline data={memHistory} color="#6366f1" maxVal={gpu.memTotalMiB} height={36} width={280} />
            </div>
            {/* VRAM bar */}
            <div style={{ height: 4, borderRadius: 2, background: 'var(--bg-elevated)', marginTop: 6, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 2,
                width: `${memPct}%`,
                background: memPct > 85 ? 'var(--danger)' : memPct > 60 ? 'var(--warning)' : '#6366f1',
                transition: 'width 0.5s ease',
              }} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── RAM Card ──────────────────────────────────────────────────────────────────
function RamCard({ usedGb, totalGb, history }: { usedGb: number; totalGb: number; history: number[] }) {
  const pct = (usedGb / totalGb) * 100
  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border-default)', borderRadius: 12, padding: '14px 16px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
          <MemoryStick size={11} /> System RAM
        </span>
        <span style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
          {usedGb.toFixed(0)} / {totalGb.toFixed(0)} GB
          <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 4 }}>({pct.toFixed(0)}%)</span>
        </span>
      </div>
      <Sparkline data={history} color="#8b5cf6" maxVal={totalGb} height={36} width={280} />
      <div style={{ height: 4, borderRadius: 2, background: 'var(--bg-elevated)', marginTop: 6, overflow: 'hidden' }}>
        <div style={{ height: '100%', borderRadius: 2, width: `${pct}%`, background: '#8b5cf6', transition: 'width 0.5s ease' }} />
      </div>
    </div>
  )
}

// ── Main Export ───────────────────────────────────────────────────────────────
export default function GpuMonitor({ active }: { active: boolean }) {
  const [latest, setLatest] = useState<GpuSnapshot | null>(null)
  const [utilHist, setUtilHist] = useState<Record<number, number[]>>({})
  const [memHist, setMemHist] = useState<Record<number, number[]>>({})
  const [ramHist, setRamHist] = useState<number[]>([])
  const [error, setError] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const MAX_POINTS = 60

  /**
   * Pulls the Ray /nodes?view=summary from `endpointUrl` and appends
   * its GPUs to `acc`. Tag each GPU with the source cluster ("on-prem"
   * or "modal") so the UI can label cards correctly. Returns the
   * number of nodes that were actually present.
   */
  const _ingest = async (
    acc: { gpus: GpuSnapshot['gpus']; ramUsed: number; ramTotal: number; nodes: number },
    endpointUrl: string,
    clusterLabel: 'on-prem' | 'modal',
    globalOffset: number,
  ) => {
    const res = await fetch(endpointUrl)
    if (!res.ok) return
    const data = await res.json()
    const nodes: any[] = data?.data?.summary ?? []
    if (!nodes.length) return
    nodes.forEach((node: any, nIdx: number) => {
      const raylet = node.raylet || {}
      const nodeHost = raylet.hostname || raylet.nodeManagerAddress || node.ip || '?'
        for (const g of (node.gpus ?? [])) {
          const localIdx = g.index ?? 0
          acc.gpus.push({
            // Globally unique across both clusters. Modal GPUs get
            // +10_000 to avoid collisions with on-prem keys.
            index:       globalOffset + nIdx * 100 + localIdx,
            localIndex:  localIdx,
            nodeIndex:   nIdx,
            nodeHost,
            cluster:     clusterLabel,
            name:        g.name ?? 'GPU',
            utilGpu:     Math.round(g.utilizationGpu ?? 0),
            memUsedMiB:  g.memoryUsed  ?? 0,
            memTotalMiB: g.memoryTotal ?? 0,
            tempC:       Math.round(g.temperatureC ?? 0),
            powerW:      Math.round((g.powerMw ?? 0) / 1000),
          })
        }
      if (Array.isArray(node.mem) && node.mem.length >= 2) {
        acc.ramUsed += Math.max(0, (node.mem[0] ?? 0) - (node.mem[1] ?? 0))
        acc.ramTotal += (node.mem[0] ?? 0)
      }
    })
    acc.nodes += nodes.length
  }

  const poll = async () => {
    const acc = { gpus: [] as GpuSnapshot['gpus'], ramUsed: 0, ramTotal: 0, nodes: 0 }

    // Poll BOTH clusters in parallel. Either can 404 / fail; the
    // other still contributes. This is the fix for "I see 0% util
    // even though my Modal job is running" — the modal cluster's
    // Ray dashboard lives at a different URL than the on-prem one.
    const results = await Promise.allSettled([
      _ingest(acc, '/api/ray/nodes?view=summary',    'on-prem', 0),
      _ingest(acc, '/api/modal/nodes?view=summary', 'modal',   10_000),
    ])
    const anyOk = results.some(r => r.status === 'fulfilled')
    if (!anyOk || acc.gpus.length === 0) {
      setError(true)
      return
    }
    setError(false)

    const ramUsedGb  = acc.ramUsed / 1024 ** 3
    const ramTotalGb = acc.ramTotal / 1024 ** 3
    const snap: GpuSnapshot = {
      ts: Date.now(),
      gpus: acc.gpus,
      ramUsedGb, ramTotalGb,
      nodesTotal: acc.nodes,
    }
    setLatest(snap)

    setUtilHist(prev => {
      const next = { ...prev }
      for (const g of acc.gpus) {
        next[g.index] = [...(prev[g.index] ?? []), g.utilGpu].slice(-MAX_POINTS)
      }
      return next
    })
    setMemHist(prev => {
      const next = { ...prev }
      for (const g of acc.gpus) {
        next[g.index] = [...(prev[g.index] ?? []), g.memUsedMiB].slice(-MAX_POINTS)
      }
      return next
    })
    setRamHist(prev => [...prev, ramUsedGb].slice(-MAX_POINTS))
  }

  useEffect(() => {
    if (!active) {
      if (timerRef.current) clearInterval(timerRef.current)
      return
    }
    poll()
    timerRef.current = setInterval(poll, 2000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [active])

  if (error || !latest) {
    return (
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
        borderRadius: 12, padding: '16px 20px', fontSize: 12, color: 'var(--text-muted)',
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <Cpu size={14} /> {error ? 'Ray cluster ไม่ตอบสนอง' : 'กำลังโหลด GPU monitor...'}
      </div>
    )
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--success)', boxShadow: '0 0 6px var(--success)', animation: 'pulse 2s infinite' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>
          Live GPU / Memory &mdash; Ray Cluster
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>
          อัพเดตทุก 2s &middot; {latest.nodesTotal} node{latest.nodesTotal === 1 ? '' : 's'} &middot; {latest.gpus.length} GPU
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
        {latest.gpus.map(gpu => (
          <GpuCard
            key={gpu.index}
            gpu={gpu}
            utilHistory={utilHist[gpu.index] ?? []}
            memHistory={memHist[gpu.index] ?? []}
          />
        ))}
        <RamCard
          usedGb={latest.ramUsedGb}
          totalGb={latest.ramTotalGb}
          history={ramHist}
        />
      </div>
    </div>
  )
}
