import { NavLink } from 'react-router-dom'
import {
  FolderSync, Database, Brain, ListChecks, LayoutDashboard,
  ChevronLeft, ChevronRight, Activity, Server, HardDrive, BrainCircuit, BookOpen, FlaskConical
} from 'lucide-react'
import { useState } from 'react'

const navItems = [
  { to: '/projects',     label: 'Projects',     icon: FolderSync,       desc: 'Manage datasets' },
  { to: '/datasets',     label: 'Datasets',     icon: Database,         desc: 'Storage & sync' },
  { to: '/storage',      label: 'Storage',      icon: HardDrive,        desc: 'MinIO buckets' },
  { to: '/train',        label: 'Train',        icon: Brain,            desc: 'Model training' },
  { to: '/jobs',         label: 'Jobs',         icon: ListChecks,       desc: 'Running jobs' },
  { to: '/models',       label: 'Models',       icon: BrainCircuit,     desc: 'Trained models' },
  { to: '/playground',   label: 'Playground',   icon: FlaskConical,     desc: 'Test inference' },
  { to: '/ray-cluster',  label: 'Ray Cluster',  icon: Server,           desc: 'Cluster & GPUs' },
  { to: '/notebook',     label: 'Notebook',     icon: BookOpen,         desc: 'Jupyter Notebook' },
  { to: '/status',       label: 'Status',       icon: Activity,         desc: 'System health' },
  { to: '/dashboard',    label: 'Dashboard',    icon: LayoutDashboard,  desc: 'Analytics' },
]

interface SidebarProps {
  onCollapsedChange?: (collapsed: boolean) => void
}

export default function Sidebar({ onCollapsedChange }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className="sidebar"
      style={{
        width: collapsed ? 68 : 240,
        minHeight: '100vh',
        background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border-default)',
        display: 'flex',
        flexDirection: 'column',
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
        zIndex: 100,
        transition: 'width 0.25s cubic-bezier(0.4,0,0.2,1)',
        overflow: 'hidden',
      }}
    >
      {/* Logo */}
      <div
        style={{
          padding: collapsed ? '20px 0' : '20px 20px 16px',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          justifyContent: collapsed ? 'center' : 'flex-start',
          flexShrink: 0,
        }}
      >
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 8,
            background: 'var(--primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <Activity size={18} color="#fff" />
        </div>
        {!collapsed && (
          <div>
            <div style={{ fontWeight: 800, fontSize: 15, color: 'var(--text-primary)', letterSpacing: '-0.02em', lineHeight: 1.2 }}>
              MedImage
            </div>
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 1 }}>
              AI Platform
            </div>
          </div>
        )}
      </div>

      {/* Nav Items */}
      <nav style={{ flex: 1, padding: '12px 10px', display: 'flex', flexDirection: 'column', gap: 3, overflow: 'hidden' }}>
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: collapsed ? '10px 0' : '8px 10px',
              justifyContent: collapsed ? 'center' : 'flex-start',
              borderRadius: 8,
              textDecoration: 'none',
              transition: 'background 0.12s ease, color 0.12s ease',
              background: isActive ? 'var(--primary-dim)' : 'transparent',
              color: isActive ? 'var(--primary-hover)' : 'var(--text-secondary)',
            })}
            className="sidebar-nav-item"
          >
            <Icon size={17} style={{ flexShrink: 0, color: 'inherit' }} />
            {!collapsed && (
              <div style={{ overflow: 'hidden' }}>
                <div style={{ fontWeight: 500, fontSize: 13.5, color: 'inherit', whiteSpace: 'nowrap' }}>
                  {label}
                </div>
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom: Cluster Status + Collapse */}
      <div style={{
        padding: collapsed ? '14px 0' : '14px 16px',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        flexShrink: 0,
      }}>
        {/* Cluster Status */}
        {!collapsed && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 10px',
            borderRadius: 8,
            background: 'var(--bg-elevated)',
          }}>
            <div style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: 'var(--success)',
              flexShrink: 0,
            }} />
            <div style={{ overflow: 'hidden', flex: 1 }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-muted)' }}>
                Ray Cluster
              </div>
              <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                100.68.53.118
              </div>
            </div>
          </div>
        )}

        {/* Collapse Button */}
        <button
          onClick={() => setCollapsed(c => { const next = !c; onCollapsedChange?.(next); return next })}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            gap: 8,
            padding: collapsed ? '8px 0' : '8px 10px',
            background: 'transparent',
            border: 'none',
            borderRadius: 8,
            color: 'var(--text-muted)',
            cursor: 'pointer',
            transition: 'color 0.12s ease, background 0.12s ease',
            width: '100%',
            fontSize: 12.5,
            fontFamily: 'var(--font-ui)',
          }}
          className="collapse-btn"
        >
          {collapsed
            ? <ChevronRight size={16} />
            : <><ChevronLeft size={16} /><span>Collapse</span></>
          }
        </button>
      </div>

      <style>{`
        .sidebar-nav-item:hover:not([aria-current="page"]) {
          background: var(--bg-elevated) !important;
          color: var(--text-primary) !important;
        }
        .collapse-btn:hover {
          background: var(--bg-elevated) !important;
          color: var(--text-primary) !important;
        }
        .sidebar-nav-item:focus-visible,
        .collapse-btn:focus-visible {
          outline: 2px solid var(--primary);
          outline-offset: 2px;
        }
      `}</style>
    </aside>
  )
}
