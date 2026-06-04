import { useState, useEffect } from 'react'
import {
  Users as UsersIcon, UserPlus, Edit2, Trash2, Key,
  ShieldOff, CheckCircle2, XCircle, Eye, EyeOff, Smartphone,
  RefreshCw, AlertCircle,
} from 'lucide-react'
import keycloak from '../keycloak'

interface KCUser {
  id: string
  username: string
  email: string
  firstName: string
  lastName: string
  enabled: boolean
  has_totp: boolean
  totp_credentials: { id: string; userLabel: string; createdDate: number | null }[]
  realmRoles: string[]
  createdTimestamp: number
}

// ─── Access Denied ────────────────────────────────────────────────────────────
function AccessDenied() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 400, gap: 16 }}>
      <ShieldOff size={48} style={{ color: 'var(--text-muted)' }} />
      <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-secondary)' }}>Access Denied</div>
      <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>Platform admin role required</div>
    </div>
  )
}

// ─── Modal Wrapper ────────────────────────────────────────────────────────────
function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24,
    }} onClick={onClose}>
      <div style={{
        background: 'var(--bg-surface)', borderRadius: 14, padding: '28px 32px',
        width: '100%', maxWidth: 480, boxShadow: '0 24px 60px rgba(0,0,0,0.35)',
        border: '1px solid var(--border-default)',
      }} onClick={e => e.stopPropagation()}>
        <div style={{ fontWeight: 700, fontSize: 16, color: 'var(--text-primary)', marginBottom: 20 }}>{title}</div>
        {children}
      </div>
    </div>
  )
}

// ─── Field helper ─────────────────────────────────────────────────────────────
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 5, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </label>
      {children}
    </div>
  )
}

const inputStyle: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box',
  padding: '9px 12px', borderRadius: 8,
  border: '1px solid var(--border-default)',
  background: 'var(--bg-base)', color: 'var(--text-primary)',
  fontSize: 13.5, outline: 'none',
}

// ─── Create User Modal ────────────────────────────────────────────────────────
function CreateUserModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [form, setForm] = useState({ username: '', email: '', firstName: '', lastName: '', password: '', confirm: '' })
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, [k]: e.target.value }))

  const submit = async () => {
    if (!form.username || !form.email || !form.password) { setError('Username, email and password are required'); return }
    if (form.password !== form.confirm) { setError('Passwords do not match'); return }
    setLoading(true); setError('')
    try {
      const res = await fetch('/api/admin/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: form.username, email: form.email, firstName: form.firstName, lastName: form.lastName, password: form.password }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      onCreated()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal title="Create User" onClose={onClose}>
      {error && <div style={{ color: 'var(--danger, #ef4444)', fontSize: 13, marginBottom: 12, padding: '8px 12px', background: 'rgba(239,68,68,0.08)', borderRadius: 6 }}>{error}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <Field label="First Name"><input style={inputStyle} value={form.firstName} onChange={set('firstName')} placeholder="First" /></Field>
        <Field label="Last Name"><input style={inputStyle} value={form.lastName} onChange={set('lastName')} placeholder="Last" /></Field>
      </div>
      <Field label="Username *"><input style={inputStyle} value={form.username} onChange={set('username')} placeholder="username" autoComplete="off" /></Field>
      <Field label="Email *"><input style={inputStyle} type="email" value={form.email} onChange={set('email')} placeholder="user@example.com" /></Field>
      <Field label="Password *">
        <div style={{ position: 'relative' }}>
          <input style={{ ...inputStyle, paddingRight: 40 }} type={showPw ? 'text' : 'password'} value={form.password} onChange={set('password')} autoComplete="new-password" />
          <button onClick={() => setShowPw(v => !v)} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
            {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
      </Field>
      <Field label="Confirm Password *">
        <input style={inputStyle} type={showPw ? 'text' : 'password'} value={form.confirm} onChange={set('confirm')} autoComplete="new-password" />
      </Field>
      <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 8 }}>
        <button onClick={onClose} style={{ padding: '8px 18px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 13.5 }}>Cancel</button>
        <button onClick={submit} disabled={loading} style={{ padding: '8px 18px', borderRadius: 8, border: 'none', background: 'var(--primary)', color: '#fff', cursor: loading ? 'not-allowed' : 'pointer', fontSize: 13.5, opacity: loading ? 0.7 : 1 }}>
          {loading ? 'Creating…' : 'Create User'}
        </button>
      </div>
    </Modal>
  )
}

// ─── Edit User Modal ──────────────────────────────────────────────────────────
function EditUserModal({ user, onClose, onSaved }: { user: KCUser; onClose: () => void; onSaved: () => void }) {
  const [form, setForm] = useState({ email: user.email, firstName: user.firstName, lastName: user.lastName, enabled: user.enabled })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, [k]: e.target.value }))

  const submit = async () => {
    setLoading(true); setError('')
    try {
      const res = await fetch(`/api/admin/users/${user.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      onSaved()
    } catch (e: any) { setError(e.message) } finally { setLoading(false) }
  }

  return (
    <Modal title={`Edit · ${user.username}`} onClose={onClose}>
      {error && <div style={{ color: 'var(--danger, #ef4444)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <Field label="First Name"><input style={inputStyle} value={form.firstName} onChange={set('firstName')} /></Field>
        <Field label="Last Name"><input style={inputStyle} value={form.lastName} onChange={set('lastName')} /></Field>
      </div>
      <Field label="Email"><input style={inputStyle} type="email" value={form.email} onChange={set('email')} /></Field>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
        <label style={{ fontSize: 13.5, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
          <input type="checkbox" checked={form.enabled} onChange={e => setForm(f => ({ ...f, enabled: e.target.checked }))} />
          Account enabled
        </label>
      </div>
      <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
        <button onClick={onClose} style={{ padding: '8px 18px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 13.5 }}>Cancel</button>
        <button onClick={submit} disabled={loading} style={{ padding: '8px 18px', borderRadius: 8, border: 'none', background: 'var(--primary)', color: '#fff', cursor: loading ? 'not-allowed' : 'pointer', fontSize: 13.5, opacity: loading ? 0.7 : 1 }}>
          {loading ? 'Saving…' : 'Save Changes'}
        </button>
      </div>
    </Modal>
  )
}

// ─── Reset Password Modal ─────────────────────────────────────────────────────
function ResetPasswordModal({ user, onClose, onReset }: { user: KCUser; onClose: () => void; onReset: () => void }) {
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async () => {
    if (!password) { setError('Password is required'); return }
    if (password !== confirm) { setError('Passwords do not match'); return }
    setLoading(true); setError('')
    try {
      const res = await fetch(`/api/admin/users/${user.id}/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      onReset()
    } catch (e: any) { setError(e.message) } finally { setLoading(false) }
  }

  return (
    <Modal title={`Reset Password · ${user.username}`} onClose={onClose}>
      {error && <div style={{ color: 'var(--danger, #ef4444)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
      <Field label="New Password">
        <div style={{ position: 'relative' }}>
          <input style={{ ...inputStyle, paddingRight: 40 }} type={showPw ? 'text' : 'password'} value={password} onChange={e => setPassword(e.target.value)} autoComplete="new-password" />
          <button onClick={() => setShowPw(v => !v)} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
            {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
      </Field>
      <Field label="Confirm Password"><input style={inputStyle} type={showPw ? 'text' : 'password'} value={confirm} onChange={e => setConfirm(e.target.value)} autoComplete="new-password" /></Field>
      <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 8 }}>
        <button onClick={onClose} style={{ padding: '8px 18px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 13.5 }}>Cancel</button>
        <button onClick={submit} disabled={loading} style={{ padding: '8px 18px', borderRadius: 8, border: 'none', background: 'var(--warning, #f59e0b)', color: '#fff', cursor: loading ? 'not-allowed' : 'pointer', fontSize: 13.5, opacity: loading ? 0.7 : 1 }}>
          {loading ? 'Resetting…' : 'Reset Password'}
        </button>
      </div>
    </Modal>
  )
}

// ─── Delete Confirm Modal ─────────────────────────────────────────────────────
function DeleteConfirmModal({ user, onClose, onDeleted }: { user: KCUser; onClose: () => void; onDeleted: () => void }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const confirm = async () => {
    setLoading(true); setError('')
    try {
      const res = await fetch(`/api/admin/users/${user.id}`, { method: 'DELETE' })
      if (!res.ok) {
        const d = await res.json().catch(() => ({}))
        throw new Error(d.detail || `HTTP ${res.status}`)
      }
      onDeleted()
    } catch (e: any) { setError(e.message) } finally { setLoading(false) }
  }

  return (
    <Modal title="Delete User" onClose={onClose}>
      {error && <div style={{ color: 'var(--danger, #ef4444)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
      <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 20 }}>
        Are you sure you want to delete <strong>{user.username}</strong>? This action cannot be undone.
      </p>
      <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
        <button onClick={onClose} style={{ padding: '8px 18px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 13.5 }}>Cancel</button>
        <button onClick={confirm} disabled={loading} style={{ padding: '8px 18px', borderRadius: 8, border: 'none', background: 'var(--danger, #ef4444)', color: '#fff', cursor: loading ? 'not-allowed' : 'pointer', fontSize: 13.5, opacity: loading ? 0.7 : 1 }}>
          {loading ? 'Deleting…' : 'Delete'}
        </button>
      </div>
    </Modal>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function Users() {
  const roles = keycloak.tokenParsed?.realm_access?.roles ?? []
  const isAdmin = roles.includes('platform-admin')

  const [users, setUsers] = useState<KCUser[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [modal, setModal] = useState<'create' | 'edit' | 'reset' | 'delete' | null>(null)
  const [selected, setSelected] = useState<KCUser | null>(null)
  const [toast, setToast] = useState<string | null>(null)

  const currentUserId = keycloak.tokenParsed?.sub

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 3000)
  }

  const fetchUsers = async () => {
    setLoading(true); setError(null)
    try {
      const res = await fetch('/api/admin/users')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setUsers(await res.json())
    } catch (e: any) { setError(e.message) } finally { setLoading(false) }
  }

  useEffect(() => { if (isAdmin) fetchUsers() }, [isAdmin])

  if (!isAdmin) return <AccessDenied />

  const filtered = users.filter(u =>
    !search || u.username.includes(search) || u.email?.includes(search) ||
    `${u.firstName} ${u.lastName}`.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <UsersIcon size={22} style={{ color: 'var(--primary)' }} />
            <h1 style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>User Management</h1>
          </div>
          <p style={{ fontSize: 13.5, color: 'var(--text-muted)', margin: '4px 0 0 32px' }}>
            {users.length} user{users.length !== 1 ? 's' : ''} in h-forge realm
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button onClick={fetchUsers} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 14px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 13 }}>
            <RefreshCw size={14} /> Refresh
          </button>
          <button onClick={() => setModal('create')} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', borderRadius: 8, border: 'none', background: 'var(--primary)', color: '#fff', cursor: 'pointer', fontWeight: 600, fontSize: 13.5 }}>
            <UserPlus size={15} /> Add User
          </button>
        </div>
      </div>

      {/* Search */}
      <div style={{ marginBottom: 18 }}>
        <input
          value={search} onChange={e => setSearch(e.target.value)}
          placeholder="Search by username, email, or name…"
          style={{ ...inputStyle, maxWidth: 360, padding: '9px 14px' }}
        />
      </div>

      {/* Error */}
      {error && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '12px 16px', borderRadius: 8, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: 'var(--danger, #ef4444)', marginBottom: 16 }}>
          <AlertCircle size={15} /> {error}
        </div>
      )}

      {/* Table */}
      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 200 }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>Loading users…</div>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                {['User', 'Email', 'Roles', '2FA', 'Status', 'Created', 'Actions'].map(h => (
                  <th key={h} style={{ padding: '12px 16px', textAlign: 'left', fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', whiteSpace: 'nowrap' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr><td colSpan={7} style={{ padding: '32px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13.5 }}>No users found</td></tr>
              ) : filtered.map(u => (
                <tr key={u.id} style={{ borderBottom: '1px solid var(--border-subtle)' }} className="user-row">
                  <td style={{ padding: '12px 16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{
                        width: 32, height: 32, borderRadius: '50%', flexShrink: 0,
                        background: u.id === currentUserId ? 'var(--primary)' : 'var(--bg-elevated)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 12, fontWeight: 700, color: u.id === currentUserId ? '#fff' : 'var(--text-secondary)',
                      }}>
                        {(u.firstName?.[0] || u.username[0] || '?').toUpperCase()}
                      </div>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: 13.5, color: 'var(--text-primary)' }}>
                          {u.username}
                          {u.id === currentUserId && <span style={{ marginLeft: 6, fontSize: 10, background: 'var(--primary-dim)', color: 'var(--primary-hover)', padding: '1px 6px', borderRadius: 4, fontWeight: 600 }}>You</span>}
                        </div>
                        {(u.firstName || u.lastName) && (
                          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{[u.firstName, u.lastName].filter(Boolean).join(' ')}</div>
                        )}
                      </div>
                    </div>
                  </td>
                  <td style={{ padding: '12px 16px', fontSize: 13, color: 'var(--text-secondary)' }}>{u.email || '—'}</td>
                  <td style={{ padding: '12px 16px' }}>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {u.realmRoles.filter(r => !r.startsWith('default-')).map(r => (
                        <span key={r} className="badge-primary" style={{ fontSize: 11, padding: '2px 7px', borderRadius: 4 }}>{r}</span>
                      ))}
                      {u.realmRoles.filter(r => !r.startsWith('default-')).length === 0 && (
                        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>—</span>
                      )}
                    </div>
                  </td>
                  <td style={{ padding: '12px 16px' }}>
                    {u.has_totp
                      ? <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--success, #22c55e)', fontWeight: 600 }}><Smartphone size={13} /> Enabled</span>
                      : <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>—</span>
                    }
                  </td>
                  <td style={{ padding: '12px 16px' }}>
                    {u.enabled
                      ? <span className="badge-success" style={{ fontSize: 11 }}><CheckCircle2 size={11} style={{ display: 'inline', marginRight: 4 }} />Active</span>
                      : <span className="badge-danger" style={{ fontSize: 11 }}><XCircle size={11} style={{ display: 'inline', marginRight: 4 }} />Disabled</span>
                    }
                  </td>
                  <td style={{ padding: '12px 16px', fontSize: 12, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {u.createdTimestamp ? new Date(u.createdTimestamp).toLocaleDateString('th-TH') : '—'}
                  </td>
                  <td style={{ padding: '12px 16px' }}>
                    <div style={{ display: 'flex', gap: 4 }}>
                      <button title="Edit user" onClick={() => { setSelected(u); setModal('edit') }} className="icon-action-btn">
                        <Edit2 size={14} />
                      </button>
                      <button title="Reset password" onClick={() => { setSelected(u); setModal('reset') }} className="icon-action-btn">
                        <Key size={14} />
                      </button>
                      <button title="Delete user" disabled={u.id === currentUserId} onClick={() => { setSelected(u); setModal('delete') }}
                        className="icon-action-btn danger"
                        style={{ opacity: u.id === currentUserId ? 0.3 : 1, cursor: u.id === currentUserId ? 'not-allowed' : 'pointer' }}>
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modals */}
      {modal === 'create' && (
        <CreateUserModal onClose={() => setModal(null)} onCreated={() => { setModal(null); fetchUsers(); showToast('User created successfully') }} />
      )}
      {modal === 'edit' && selected && (
        <EditUserModal user={selected} onClose={() => setModal(null)} onSaved={() => { setModal(null); fetchUsers(); showToast('User updated') }} />
      )}
      {modal === 'reset' && selected && (
        <ResetPasswordModal user={selected} onClose={() => setModal(null)} onReset={() => { setModal(null); showToast('Password reset successfully') }} />
      )}
      {modal === 'delete' && selected && (
        <DeleteConfirmModal user={selected} onClose={() => setModal(null)} onDeleted={() => { setModal(null); fetchUsers(); showToast('User deleted') }} />
      )}

      {/* Toast */}
      {toast && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24, zIndex: 2000,
          background: 'var(--success, #22c55e)', color: '#fff',
          padding: '10px 20px', borderRadius: 8, fontWeight: 600, fontSize: 13.5,
          boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
        }}>
          <CheckCircle2 size={14} style={{ display: 'inline', marginRight: 6 }} />
          {toast}
        </div>
      )}

      <style>{`
        .user-row:hover { background: var(--bg-elevated); }
        .icon-action-btn {
          display: flex; align-items: center; justify-content: center;
          width: 30px; height: 30px; border-radius: 6px;
          border: 1px solid var(--border-default);
          background: transparent; color: var(--text-muted);
          cursor: pointer; transition: all 0.12s;
        }
        .icon-action-btn:hover { background: var(--bg-elevated); color: var(--text-primary); border-color: var(--border-default); }
        .icon-action-btn.danger:hover { background: rgba(239,68,68,0.1); color: var(--danger, #ef4444); border-color: rgba(239,68,68,0.3); }
      `}</style>
    </div>
  )
}
