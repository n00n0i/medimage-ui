import { NavLink } from 'react-router-dom'
import {
  FolderSync, Database, Brain, ListChecks, LayoutDashboard,
  ChevronLeft, ChevronRight, Activity, Server
} from 'lucide-react'
import { useState } from 'react'

const navItems = [
  { to: '/projects',     label: 'Projects',     icon: FolderSync,       desc: 'Manage datasets' },
  { to: '/datasets',     label: 'Datasets',    icon: Database,         desc: 'Storage & sync' },
  { to: '/train',        label: 'Train',        icon: Brain,            desc: 'Model training' },
  { to: '/jobs',         label: 'Jobs',         icon: ListChecks,       desc: 'Running jobs' },
  { to: '/ray-cluster',  label: 'Ray Cluster',  icon: Server,           desc: 'Cluster & GPUs' },
  { to: '/dashboard',    label: 'Dashboard',    icon: LayoutDashboard,  desc: 'Analytics' },
]

export default function Sidebar() {
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
            borderRadius: 10,
            background: 'linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            boxShadow: '0 2px 8px rgba(30,64,175,0.4)',
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
        {navItems.map(({ to, label, icon: Icon, desc }) => (
          <NavLink
            key={to}
            to={to}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: collapsed ? '10px 0' : '10px 12px',
              justifyContent: collapsed ? 'center' : 'flex-start',
              borderRadius: 10,
              textDecoration: 'none',
              transition: 'all 0.15s ease',
              position: 'relative',
              background: isActive ? 'var(--primary-glow)' : 'transparent',
              border: isActive ? '1px solid rgba(30,64,175,0.3)' : '1px solid transparent',
              color: isActive ? 'var(--primary-hover)' : 'var(--text-secondary)',
            })}
            className="sidebar-nav-item"
          >
            {({ isActive }) => (
              <>
                {/* Active indicator bar */}
                {isActive && (
                  <div style={{
                    position: 'absolute',
                    left: collapsed ? -10 : -10,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    width: 3,
                    height: 24,
                    borderRadius: 2,
                    background: 'var(--primary)',
                    boxShadow: '0 0 8px var(--primary)',
                  }} />
                )}
                <Icon size={18} style={{ flexShrink: 0, color: 'inherit' }} />
                {!collapsed && (
                  <div style={{ overflow: 'hidden' }}>
                    <div style={{ fontWeight: 600, fontSize: 13.5, color: 'inherit', whiteSpace: 'nowrap' }}>
                      {label}
                    </div>
                    <div style={{ fontSize: 10.5, color: 'var(--text-muted)', whiteSpace: 'nowrap', marginTop: 1 }}>
                      {desc}
                    </div>
                  </div>
                )}
              </>
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
            border: '1px solid var(--border-subtle)',
          }}>
            <div style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: 'var(--success)',
              boxShadow: '0 0 6px var(--success)',
              animation: 'pulse 2s ease-in-out infinite',
              flexShrink: 0,
            }} />
            <div style={{ overflow: 'hidden', flex: 1 }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' }}>
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
          onClick={() => setCollapsed(c => !c)}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            gap: 8,
            padding: collapsed ? '8px 0' : '8px 10px',
            background: 'transparent',
            border: '1px solid var(--border-subtle)',
            borderRadius: 8,
            color: 'var(--text-muted)',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
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
