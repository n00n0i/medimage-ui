import { useEffect, useState, useCallback, useRef } from 'react'

// ─── Per-user preferences hook ────────────────────────────────────────────────
//
// Replaces all localStorage usage. All small UI state (filters, dashboard
// URLs, MinIO config, API keys, etc.) is stored server-side in the
// user_prefs table, keyed by the JWT subject.
//
// Reads: synchronous, served from the in-memory mirror. The mirror is
// populated on mount by GET /api/user-prefs and refreshed on focus.
// Writes: fire-and-forget PUT, mirror is updated immediately so the UI
// reflects the change without waiting for the round trip.

const _mirror: Record<string, string> = {}
const _subs = new Set<() => void>()
let _loaded = false
let _loadingPromise: Promise<void> | null = null

function _notify() {
  for (const cb of _subs) cb()
}

async function _loadFromServer(): Promise<void> {
  if (_loadingPromise) return _loadingPromise
  _loadingPromise = (async () => {
    try {
      const r = await fetch('/api/user-prefs')
      if (!r.ok) return
      const d = await r.json()
      const prefs: Record<string, string> = d?.prefs ?? {}
      for (const k of Object.keys(_mirror)) delete _mirror[k]
      Object.assign(_mirror, prefs)
      _loaded = true
      _notify()
    } catch { /* ignore — keep local defaults */ }
  })()
  return _loadingPromise
}

async function _put(key: string, value: string | null) {
  // Mirror update first so the UI sees it instantly
  if (value === null) delete _mirror[key]
  else _mirror[key] = value
  _notify()
  try {
    await fetch(`/api/user-prefs/${encodeURIComponent(key)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value }),
    })
  } catch { /* best-effort — mirror is already updated */ }
}

// Public API ──────────────────────────────────────────────────────────────

/** Synchronous read from the in-memory mirror. Returns undefined if not loaded yet. */
export function getPref(key: string): string | undefined {
  return _mirror[key]
}

/** Default value lookup: returns the pref if loaded, else the fallback. */
export function getPrefOr(key: string, fallback: string): string {
  const v = _mirror[key]
  return v !== undefined ? v : fallback
}

/** Delete a preference (PUT value: null). */
export function deletePref(key: string): Promise<void> {
  return _put(key, null)
}

/** Set a preference (PUT value: string). */
export function setPref(key: string, value: string): Promise<void> {
  return _put(key, value)
}

/** True if the mirror has been populated from the server at least once. */
export function isPrefsLoaded(): boolean { return _loaded }

/** Subscribe to pref changes (mirror updates or initial load). Returns unsubscribe. */
export function subscribePrefs(cb: () => void): () => void {
  _subs.add(cb)
  return () => { _subs.delete(cb) }
}

/** Force a refresh from the server. Returns a promise that resolves when done. */
export function refreshPrefs(): Promise<void> {
  _loaded = false
  return _loadFromServer()
}

/** React hook for reading a single preference with a fallback. */
export function useUserPref(key: string, fallback: string = ''): [string, (v: string) => void, () => void] {
  const [, force] = useState(0)
  useEffect(() => {
    // Trigger an initial load if we haven't yet
    if (!_loaded && !_loadingPromise) void _loadFromServer()
    return subscribePrefs(() => force(n => n + 1))
  }, [])

  // Re-subscribe if the key changes
  const keyRef = useRef(key)
  useEffect(() => { keyRef.current = key }, [key])

  const value = _mirror[key] !== undefined ? _mirror[key] : fallback
  const set = useCallback((v: string) => { void _put(key, v) }, [key])
  const remove = useCallback(() => { void _put(key, null) }, [key])
  return [value, set, remove]
}

/** React hook for the full prefs map (read-only). */
export function useAllPrefs(): Record<string, string> {
  const [, force] = useState(0)
  useEffect(() => {
    if (!_loaded && !_loadingPromise) void _loadFromServer()
    return subscribePrefs(() => force(n => n + 1))
  }, [])
  return _mirror
}

// Do NOT auto-load on first import — this module can be imported before
// keycloak is initialized, in which case the global fetch patch has no
// token to inject and the GET returns 401. The mirror stays empty and
// any module-level getPref() call (e.g. the bulk-run dismissed-flag
// check) returns undefined forever. Instead, callers MUST trigger the
// load via refreshPrefs() or the hooks (which run after mount, when
// keycloak is ready). The hooks all check `!_loaded` and call
// _loadFromServer() on first use.
