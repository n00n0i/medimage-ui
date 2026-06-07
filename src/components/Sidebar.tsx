import { NavLink } from 'react-router-dom'
import {
  FolderSync, Database, Brain, ListChecks, LayoutDashboard,
  ChevronLeft, ChevronRight, Activity, Server, HardDrive, BrainCircuit, BookOpen, FlaskConical, Zap, Cloud, LogOut, User, Users, UserCircle, Cpu, MessageSquare,
} from 'lucide-react'
import { useState } from 'react'
import keycloak from '../keycloak'

const allNavItems = [
  { to: '/projects',     label: 'Projects',     icon: FolderSync,       adminOnly: false },
  { to: '/datasets',     label: 'Datasets',     icon: Database,         adminOnly: false },
  { to: '/storage',      label: 'Storage',      icon: HardDrive,        adminOnly: false },
  { to: '/jobs',         label: 'Jobs',         icon: ListChecks,       adminOnly: false },
  { to: '/models',       label: 'Models',       icon: BrainCircuit,     adminOnly: false },
  { to: '/playground',   label: 'Playground',   icon: FlaskConical,     adminOnly: false },
  { to: '/api-service',  label: 'API Service',  icon: Zap,              adminOnly: false },
  { to: '/ray-cluster',    label: 'Ray Cluster',    icon: Server,           adminOnly: false },
  { to: '/modal-cluster',  label: 'Modal',           icon: Cloud,            adminOnly: false },
  { to: '/notebook',     label: 'Notebook',     icon: BookOpen,         adminOnly: false },
  { to: '/status',       label: 'Status',       icon: Activity,         adminOnly: false },
  { to: '/dashboard',    label: 'Dashboard',    icon: LayoutDashboard,  adminOnly: false },
  { to: '/users',        label: 'Users',        icon: Users,            adminOnly: true  },
]

// Train submenu — separate from the main flat list so we can render
// it as a parent group with 2 indented children (Deep Learning +
// Large Language Model) instead of 2 top-level items.
const trainSubItems = [
  { to: '/train/deep-learning', label: 'Deep Learning',         icon: Cpu,           adminOnly: false },
  { to: '/train/llm',            label: 'Large Language Model',  icon: MessageSquare, adminOnly: false },
]

interface SidebarProps {
  onCollapsedChange?: (collapsed: boolean) => void
}

export default function Sidebar({ onCollapsedChange }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)
  const roles = (keycloak.tokenParsed as any)?.realm_access?.roles ?? []
  const isAdmin = roles.includes('platform-admin')
  const navItems = allNavItems.filter(item => !item.adminOnly || isAdmin)

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
              H-Forge
            </div>
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 1 }}>
              AI Platform
            </div>
          </div>
        )}
      </div>

      {/* Nav Items */}
      <nav style={{ flex: 1, padding: '12px 10px', display: 'flex', flexDirection: 'column', gap: 3, overflowY: 'auto', overflowX: 'hidden' }}>
        {navItems.map(({ to, label, icon: Icon }) => {
          // Inject the Train submenu group right after the
          // Storage item so the sidebar reads as: Storage,
          // then "Train → Deep Learning / LLM" as a parent
          // group, then Jobs / Models / ...
          const showTrainGroupAfter = to === '/storage'
          return (
            <div key={to}>
              <NavLink
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
              {showTrainGroupAfter && (
                <div style={{ marginTop: 2 }}>
                  {!collapsed && (
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: 8,
                      padding: '6px 10px 4px 10px',
                      color: 'var(--text-muted)',
                      fontSize: 11.5, fontWeight: 600,
                      textTransform: 'uppercase', letterSpacing: '0.06em',
                    }}>
                      <Brain size={12} />
                      <span>Train</span>
                    </div>
                  )}
                  {trainSubItems
                    .filter(s => !s.adminOnly || isAdmin)
                    .map(({ to: subTo, label: subLabel, icon: SubIcon }) => (
                      <NavLink
                        key={subTo}
                        to={subTo}
                        style={({ isActive }) => ({
                          display: 'flex',
                          alignItems: 'center',
                          gap: 10,
                          padding: collapsed ? '8px 0' : '6px 10px 6px 28px',
                          justifyContent: collapsed ? 'center' : 'flex-start',
                          borderRadius: 8,
                          textDecoration: 'none',
                          transition: 'background 0.12s ease, color 0.12s ease',
                          background: isActive ? 'var(--primary-dim)' : 'transparent',
                          color: isActive ? 'var(--primary-hover)' : 'var(--text-secondary)',
                          fontSize: 13,
                        })}
                        className="sidebar-nav-item"
                      >
                        <SubIcon size={15} style={{ flexShrink: 0, color: 'inherit' }} />
                        {!collapsed && (
                          <div style={{ overflow: 'hidden' }}>
                            <div style={{ fontWeight: 500, fontSize: 13, color: 'inherit', whiteSpace: 'nowrap' }}>
                              {subLabel}
                            </div>
                          </div>
                        )}
                      </NavLink>
                    ))}
                </div>
              )}
            </div>
          )
        })}
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

        {/* User + Logout */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: collapsed ? '8px 0' : '8px 10px',
          borderRadius: 8,
          background: 'var(--bg-elevated)',
          justifyContent: collapsed ? 'center' : 'flex-start',
        }}>
          <NavLink to="/profile" title="Profile" style={{ display: 'contents', textDecoration: 'none' }}>
          <div style={{ width: 26, height: 26, borderRadius: '50%',
            background: 'var(--primary)', display: 'flex',
            alignItems: 'center', justifyContent: 'center', flexShrink: 0, cursor: 'pointer',
          }}>
            <User size={13} color="#fff" />
          </div>
          </NavLink>
          {!collapsed && (
            <>
              <div style={{ flex: 1, overflow: 'hidden' }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {keycloak.tokenParsed?.preferred_username ?? keycloak.tokenParsed?.email ?? 'User'}
                </div>
                <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>Authenticated</div>
              </div>
              <NavLink
                to="/profile"
                title="Profile"
                style={{ display: 'flex', alignItems: 'center', padding: 4, borderRadius: 6,
                  color: 'var(--text-muted)', textDecoration: 'none', flexShrink: 0,
                  transition: 'color 0.12s ease',
                }}
                className="logout-btn"
              >
                <UserCircle size={14} />
              </NavLink>
              <button
                title="Sign out"
                onClick={() => keycloak.logout()}
                style={{
                  background: 'transparent', border: 'none', cursor: 'pointer',
                  color: 'var(--text-muted)', display: 'flex', alignItems: 'center',
                  padding: 4, borderRadius: 6, transition: 'color 0.12s ease',
                  flexShrink: 0,
                }}
                className="logout-btn"
              >
                <LogOut size={14} />
              </button>
            </>
          )}
          {collapsed && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center' }}>
              <NavLink
                to="/profile"
                title="Profile"
                style={{ display: 'flex', alignItems: 'center', padding: 4, borderRadius: 6,
                  color: 'var(--text-muted)', textDecoration: 'none',
                  transition: 'color 0.12s ease',
                }}
                className="logout-btn"
              >
                <UserCircle size={14} />
              </NavLink>
              <button
                title="Sign out"
                onClick={() => keycloak.logout()}
                style={{
                  background: 'transparent', border: 'none', cursor: 'pointer',
                  color: 'var(--text-muted)', display: 'flex', alignItems: 'center',
                  padding: 4, borderRadius: 6, transition: 'color 0.12s ease',
                }}
                className="logout-btn"
              >
                <LogOut size={14} />
              </button>
            </div>
          )}
        </div>

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
        .logout-btn:hover {
          color: var(--danger, #ef4444) !important;
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
