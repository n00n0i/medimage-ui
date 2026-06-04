import { useState, useEffect } from 'react'
import {
  UserCircle, Lock, Smartphone, Eye, EyeOff,
  CheckCircle2, AlertCircle, Trash2, QrCode, ShieldCheck, RefreshCw,
} from 'lucide-react'
import keycloak from '../keycloak'

interface TOTPCredential {
  id: string
  userLabel: string
  createdDate: number | null
}

interface TOTPSetupData {
  totpSecret: string
  totpSecretEncoded: string
  totpSecretQrCode: string   // base64 PNG
  totpSessionToken: string
  manualUrl?: string
}

// ─── Section Card ─────────────────────────────────────────────────────────────
function SectionCard({ icon, title, description, children }: {
  icon: React.ReactNode; title: string; description: string; children: React.ReactNode
}) {
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, marginBottom: 22 }}>
        <div style={{ width: 38, height: 38, borderRadius: 10, background: 'var(--primary-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          {icon}
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>{title}</div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 2 }}>{description}</div>
        </div>
      </div>
      {children}
    </div>
  )
}

// ─── Field ────────────────────────────────────────────────────────────────────
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: 'block', fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 5, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
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

const readonlyStyle: React.CSSProperties = {
  ...inputStyle,
  background: 'var(--bg-elevated)',
  color: 'var(--text-secondary)',
  cursor: 'default',
}

// ─── Alert ────────────────────────────────────────────────────────────────────
function Alert({ type, children }: { type: 'error' | 'success'; children: React.ReactNode }) {
  const isError = type === 'error'
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 8,
      padding: '10px 14px', borderRadius: 8, fontSize: 13, lineHeight: 1.5,
      background: isError ? 'rgba(239,68,68,0.08)' : 'rgba(34,197,94,0.08)',
      border: `1px solid ${isError ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.2)'}`,
      color: isError ? 'var(--danger, #ef4444)' : 'var(--success, #22c55e)',
      marginBottom: 14,
    }}>
      {isError ? <AlertCircle size={15} style={{ flexShrink: 0, marginTop: 1 }} /> : <CheckCircle2 size={15} style={{ flexShrink: 0, marginTop: 1 }} />}
      {children}
    </div>
  )
}

// ─── Change Password ──────────────────────────────────────────────────────────
function ChangePassword() {
  const [form, setForm] = useState({ current: '', next: '', confirm: '' })
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<{ type: 'error' | 'success'; msg: string } | null>(null)
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, [k]: e.target.value }))

  const submit = async () => {
    if (!form.current || !form.next) { setStatus({ type: 'error', msg: 'All fields are required' }); return }
    if (form.next !== form.confirm) { setStatus({ type: 'error', msg: 'New passwords do not match' }); return }
    if (form.next.length < 6) { setStatus({ type: 'error', msg: 'Password must be at least 6 characters' }); return }
    setLoading(true); setStatus(null)
    try {
      await keycloak.updateToken(30)
      const res = await fetch('/api/profile/change-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ currentPassword: form.current, newPassword: form.next }),
      })
      if (!res.ok) {
        const text = await res.text()
        let detail = 'Failed to change password'
        try { detail = JSON.parse(text).errorMessage || detail } catch { /* ignore */ }
        throw new Error(detail)
      }
      setStatus({ type: 'success', msg: 'Password changed successfully. You may need to log in again.' })
      setForm({ current: '', next: '', confirm: '' })
    } catch (e: any) {
      setStatus({ type: 'error', msg: e.message })
    } finally { setLoading(false) }
  }

  return (
    <SectionCard
      icon={<Lock size={18} style={{ color: 'var(--primary)' }} />}
      title="Change Password"
      description="Update your account password. You'll need your current password."
    >
      <div style={{ maxWidth: 400 }}>
        {status && <Alert type={status.type}>{status.msg}</Alert>}
        <Field label="Current Password">
          <input style={inputStyle} type={showPw ? 'text' : 'password'} value={form.current} onChange={set('current')} autoComplete="current-password" />
        </Field>
        <Field label="New Password">
          <div style={{ position: 'relative' }}>
            <input style={{ ...inputStyle, paddingRight: 40 }} type={showPw ? 'text' : 'password'} value={form.next} onChange={set('next')} autoComplete="new-password" />
            <button onClick={() => setShowPw(v => !v)} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
              {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
        </Field>
        <Field label="Confirm New Password">
          <input style={inputStyle} type={showPw ? 'text' : 'password'} value={form.confirm} onChange={set('confirm')} autoComplete="new-password" />
        </Field>
        <button onClick={submit} disabled={loading} style={{ marginTop: 4, padding: '9px 20px', borderRadius: 8, border: 'none', background: 'var(--primary)', color: '#fff', cursor: loading ? 'not-allowed' : 'pointer', fontWeight: 600, fontSize: 13.5, opacity: loading ? 0.7 : 1 }}>
          {loading ? 'Updating…' : 'Update Password'}
        </button>
      </div>
    </SectionCard>
  )
}

// ─── TOTP Management ──────────────────────────────────────────────────────────
function TOTPSection() {
  const [creds, setCreds] = useState<TOTPCredential[]>([])
  const [credsLoading, setCredsLoading] = useState(true)
  const [setupData, setSetupData] = useState<TOTPSetupData | null>(null)
  const [setupLoading, setSetupLoading] = useState(false)
  const [otpCode, setOtpCode] = useState('')
  const [otpLabel, setOtpLabel] = useState('My Phone')
  const [verifyLoading, setVerifyLoading] = useState(false)
  const [status, setStatus] = useState<{ type: 'error' | 'success'; msg: string } | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [useConsole, setUseConsole] = useState(false)

  const loadCreds = async () => {
    setCredsLoading(true)
    try {
      const res = await fetch('/api/profile/totp-credentials')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setCreds(await res.json())
    } catch { setCreds([]) } finally { setCredsLoading(false) }
  }

  useEffect(() => { loadCreds() }, [])

  const startSetup = async () => {
    setSetupLoading(true); setStatus(null); setSetupData(null)
    try {
      const res = await fetch('/api/profile/totp-setup')
      if (!res.ok) {
        // Fallback: link to Keycloak account console
        setUseConsole(true)
        return
      }
      setSetupData(await res.json())
      setUseConsole(false)
    } catch {
      setUseConsole(true)
    } finally { setSetupLoading(false) }
  }

  const verifyAndSave = async () => {
    if (!setupData || !otpCode || otpCode.length < 6) {
      setStatus({ type: 'error', msg: 'Enter the 6-digit code from your authenticator app' }); return
    }
    setVerifyLoading(true); setStatus(null)
    try {
      const res = await fetch('/api/profile/totp-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          totp: otpCode,
          totpSecret: setupData.totpSecret,
          totpSecretEncoded: setupData.totpSecretEncoded,
          userLabel: otpLabel || 'My Phone',
          totpSessionToken: setupData.totpSessionToken,
        }),
      })
      if (!res.ok) {
        const d = await res.json().catch(() => ({}))
        throw new Error(d.detail || 'Invalid code — check your authenticator and try again')
      }
      setStatus({ type: 'success', msg: 'Two-factor authentication enabled successfully!' })
      setSetupData(null); setOtpCode('')
      await loadCreds()
    } catch (e: any) {
      setStatus({ type: 'error', msg: e.message })
    } finally { setVerifyLoading(false) }
  }

  const deleteCred = async (id: string) => {
    setDeletingId(id)
    try {
      const res = await fetch(`/api/profile/totp-credentials/${id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      await loadCreds()
      setStatus({ type: 'success', msg: 'Authenticator removed' })
    } catch (e: any) {
      setStatus({ type: 'error', msg: e.message })
    } finally { setDeletingId(null) }
  }

  return (
    <SectionCard
      icon={<Smartphone size={18} style={{ color: 'var(--primary)' }} />}
      title="Two-Factor Authentication"
      description="Add an extra layer of security using Google Authenticator or any TOTP app."
    >
      {status && <Alert type={status.type}>{status.msg}</Alert>}

      {/* Existing authenticators */}
      {!credsLoading && creds.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 10 }}>Active Authenticators</div>
          {creds.map(c => (
            <div key={c.id} style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '10px 14px', borderRadius: 8, border: '1px solid var(--border-default)',
              background: 'var(--bg-elevated)', marginBottom: 8,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 32, height: 32, borderRadius: 8, background: 'rgba(34,197,94,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ShieldCheck size={16} style={{ color: 'var(--success, #22c55e)' }} />
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13.5, color: 'var(--text-primary)' }}>{c.userLabel || 'Authenticator'}</div>
                  {c.createdDate && (
                    <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                      Added {new Date(c.createdDate).toLocaleDateString('th-TH')}
                    </div>
                  )}
                </div>
              </div>
              <button
                title="Remove authenticator"
                onClick={() => deleteCred(c.id)}
                disabled={deletingId === c.id}
                style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '6px 12px', borderRadius: 6, border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.06)', color: 'var(--danger, #ef4444)', cursor: 'pointer', fontSize: 12.5, fontWeight: 500, opacity: deletingId === c.id ? 0.5 : 1 }}
              >
                <Trash2 size={12} /> {deletingId === c.id ? 'Removing…' : 'Remove'}
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Setup flow */}
      {!setupData && !useConsole && (
        <button
          onClick={startSetup}
          disabled={setupLoading}
          style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '9px 18px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-secondary)', cursor: setupLoading ? 'not-allowed' : 'pointer', fontSize: 13.5, fontWeight: 500 }}
        >
          {setupLoading ? <RefreshCw size={14} className="spin" /> : <QrCode size={15} />}
          {setupLoading ? 'Loading…' : creds.length > 0 ? 'Add Another Authenticator' : 'Enable 2FA'}
        </button>
      )}

      {/* Console fallback */}
      {useConsole && (
        <div style={{ padding: '14px 18px', borderRadius: 10, background: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}>
          <div style={{ fontSize: 13.5, color: 'var(--text-secondary)', marginBottom: 12 }}>
            TOTP setup is available in the Keycloak account console.
          </div>
          <a
            href="/kc/realms/h-forge/account/#/security/totp"
            target="_blank"
            rel="noopener noreferrer"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 16px', borderRadius: 8, background: 'var(--primary)', color: '#fff', textDecoration: 'none', fontSize: 13.5, fontWeight: 600 }}
          >
            <Smartphone size={14} /> Open Account Console
          </a>
          <button onClick={() => { setUseConsole(false); setSetupData(null) }} style={{ marginLeft: 10, padding: '8px 14px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 13 }}>
            Try Again
          </button>
        </div>
      )}

      {/* QR Code + verification step */}
      {setupData && (
        <div style={{ marginTop: 8 }}>
          <div style={{ padding: '20px', borderRadius: 12, border: '1px solid var(--border-default)', background: 'var(--bg-elevated)' }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', marginBottom: 16 }}>
              Scan with your authenticator app
            </div>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
              {/* QR Code */}
              <div style={{ flexShrink: 0 }}>
                {setupData.totpSecretQrCode ? (
                  <img
                    src={`data:image/png;base64,${setupData.totpSecretQrCode}`}
                    alt="QR Code"
                    style={{ width: 160, height: 160, borderRadius: 8, border: '4px solid white' }}
                  />
                ) : setupData.manualUrl ? (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', maxWidth: 200, wordBreak: 'break-all', fontFamily: 'var(--font-mono)' }}>
                    {setupData.manualUrl}
                  </div>
                ) : null}
              </div>
              {/* Instructions */}
              <div style={{ flex: 1, minWidth: 200 }}>
                <ol style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8, paddingLeft: 18, margin: '0 0 16px' }}>
                  <li>Open <strong>Google Authenticator</strong> or any TOTP app</li>
                  <li>Tap <strong>+</strong> and choose <strong>"Scan QR code"</strong></li>
                  <li>Scan the code, then enter the 6-digit code below</li>
                </ol>
                {setupData.totpSecretEncoded && (
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Manual entry key:</div>
                    <code style={{ fontSize: 12, color: 'var(--text-secondary)', background: 'var(--bg-base)', padding: '4px 8px', borderRadius: 6, letterSpacing: '0.1em' }}>
                      {setupData.totpSecretEncoded.match(/.{1,4}/g)?.join(' ')}
                    </code>
                  </div>
                )}
                <Field label="Device Name (optional)">
                  <input style={{ ...inputStyle, maxWidth: 220 }} value={otpLabel} onChange={e => setOtpLabel(e.target.value)} placeholder="My Phone" />
                </Field>
                <Field label="Verification Code">
                  <input
                    style={{ ...inputStyle, maxWidth: 180, fontFamily: 'var(--font-mono)', letterSpacing: '0.15em', fontSize: 18, textAlign: 'center' }}
                    value={otpCode}
                    onChange={e => setOtpCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                    placeholder="000000"
                    maxLength={6}
                    inputMode="numeric"
                    autoFocus
                  />
                </Field>
                <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                  <button
                    onClick={verifyAndSave}
                    disabled={verifyLoading || otpCode.length < 6}
                    style={{ padding: '8px 18px', borderRadius: 8, border: 'none', background: 'var(--primary)', color: '#fff', cursor: (verifyLoading || otpCode.length < 6) ? 'not-allowed' : 'pointer', fontWeight: 600, fontSize: 13.5, opacity: (verifyLoading || otpCode.length < 6) ? 0.6 : 1 }}
                  >
                    {verifyLoading ? 'Verifying…' : 'Verify & Enable'}
                  </button>
                  <button onClick={() => { setSetupData(null); setOtpCode(''); setStatus(null) }} style={{ padding: '8px 14px', borderRadius: 8, border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 13 }}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`.spin { animation: spin 1s linear infinite; } @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }`}</style>
    </SectionCard>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function Profile() {
  const tp = keycloak.tokenParsed as Record<string, any> | undefined
  const username = tp?.preferred_username ?? tp?.sub ?? 'User'
  const email = tp?.email ?? '—'
  const name = [tp?.given_name, tp?.family_name].filter(Boolean).join(' ') || '—'
  const roles = (tp?.realm_access?.roles ?? []) as string[]
  const isAdmin = roles.includes('platform-admin')

  return (
    <div style={{ maxWidth: 680 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 28 }}>
        <div style={{
          width: 56, height: 56, borderRadius: '50%',
          background: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 22, fontWeight: 700, color: '#fff', flexShrink: 0,
        }}>
          {username[0]?.toUpperCase()}
        </div>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>{username}</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
            <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>{email}</span>
            {isAdmin && <span className="badge-primary" style={{ fontSize: 11 }}>platform-admin</span>}
          </div>
        </div>
      </div>

      {/* Account Details */}
      <SectionCard
        icon={<UserCircle size={18} style={{ color: 'var(--primary)' }} />}
        title="Account Details"
        description="Your account information managed by H-Forge SSO."
      >
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, maxWidth: 500 }}>
          <Field label="Username"><input style={readonlyStyle} value={username} readOnly /></Field>
          <Field label="Full Name"><input style={readonlyStyle} value={name} readOnly /></Field>
          <Field label="Email" ><input style={readonlyStyle} value={email} readOnly /></Field>
          <Field label="Roles"><input style={readonlyStyle} value={roles.filter(r => !r.startsWith('default-')).join(', ') || 'user'} readOnly /></Field>
        </div>
      </SectionCard>

      {/* Change Password */}
      <ChangePassword />

      {/* 2FA / TOTP */}
      <TOTPSection />
    </div>
  )
}
