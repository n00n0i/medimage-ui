import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import axios from 'axios'
import './index.css'
import App from './App.tsx'
import keycloak from './keycloak.ts'

// Helper: return token, refreshing if needed
async function getValidToken(): Promise<string | undefined> {
  if (!keycloak.token) return undefined
  try {
    await keycloak.updateToken(30)
  } catch {
    // refresh failed — token may still be usable if not yet expired
  }
  return keycloak.token
}

// Patch window.fetch: inject Bearer token for /api/ calls (skip if caller already set Authorization)
const _origFetch = window.fetch.bind(window)
window.fetch = async (url: RequestInfo | URL, options?: RequestInit) => {
  const urlStr = typeof url === 'string' ? url : (url as URL).toString()
  const isApi = urlStr.startsWith('/api/') || urlStr.startsWith(window.location.origin + '/api/')
  const existingAuth = (options?.headers as Record<string, string>)?.['Authorization'] ??
    (options?.headers as Record<string, string>)?.['authorization']
  if (isApi && !existingAuth) {
    const token = await getValidToken()
    if (token) {
      options = {
        ...options,
        headers: { ...(options?.headers ?? {}), Authorization: `Bearer ${token}` },
      }
    }
  }
  return _origFetch(url, options)
}

// Axios interceptor: inject Bearer token for /api/ calls only when no auth header already set
axios.interceptors.request.use(async config => {
  if (config.url?.startsWith('/api/') && !config.headers.Authorization) {
    const token = await getValidToken()
    if (token) config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

keycloak
  .init({ onLoad: 'login-required', checkLoginIframe: false, pkceMethod: 'S256' })
  .then(authenticated => {
    if (authenticated) {
      createRoot(document.getElementById('root')!).render(
        <StrictMode>
          <App />
        </StrictMode>,
      )
    }
  })
  .catch(err => {
    console.error('[keycloak] init failed — rendering without auth:', err)
    createRoot(document.getElementById('root')!).render(
      <StrictMode>
        <App />
      </StrictMode>,
    )
  })

