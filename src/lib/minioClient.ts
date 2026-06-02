/**
 * MinIO S3-compatible client using AWS Signature V4 (Web Crypto API).
 * Signs requests in the browser and routes them through the /api/minio proxy.
 */

/* ── Config ──────────────────────────────────────────────────── */
export const MINIO_DEFAULTS = {
  apiUrl:     'http://localhost:9000',
  accessKey:  'minioadmin',
  secretKey:  'minioadmin',
  consoleUrl: 'http://localhost:9001',
}

export interface MinioConfig {
  apiUrl: string
  accessKey: string
  secretKey: string
  consoleUrl: string
}

const STALE_HOSTS = ['100.68.221.236', 'minio:9000', 'minio:9001']
const MINIO_KEYS  = ['minio_api_url', 'minio_access_key', 'minio_secret_key', 'minio_console_url'] as const

export function loadMinioConfig(): MinioConfig {
  // Migrate: clear any credential that points to a stale/internal host
  const stale = MINIO_KEYS.some(k => {
    const v = localStorage.getItem(k) ?? ''
    return STALE_HOSTS.some(h => v.includes(h))
  })
  if (stale) MINIO_KEYS.forEach(k => localStorage.removeItem(k))

  return {
    apiUrl:     localStorage.getItem('minio_api_url')     ?? MINIO_DEFAULTS.apiUrl,
    accessKey:  localStorage.getItem('minio_access_key')  ?? MINIO_DEFAULTS.accessKey,
    secretKey:  localStorage.getItem('minio_secret_key')  ?? MINIO_DEFAULTS.secretKey,
    consoleUrl: localStorage.getItem('minio_console_url') ?? MINIO_DEFAULTS.consoleUrl,
  }
}

export function saveMinioConfig(cfg: Partial<MinioConfig>): void {
  if (cfg.apiUrl     != null) localStorage.setItem('minio_api_url',     cfg.apiUrl)
  if (cfg.accessKey  != null) localStorage.setItem('minio_access_key',  cfg.accessKey)
  if (cfg.secretKey  != null) localStorage.setItem('minio_secret_key',  cfg.secretKey)
  if (cfg.consoleUrl != null) localStorage.setItem('minio_console_url', cfg.consoleUrl)
}

/* ── Crypto helpers ──────────────────────────────────────────── */
async function sha256(data: ArrayBuffer | string): Promise<ArrayBuffer> {
  const buf = typeof data === 'string' ? new TextEncoder().encode(data) : data
  return crypto.subtle.digest('SHA-256', buf)
}

function toHex(buf: ArrayBuffer): string {
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('')
}

async function hmac(key: ArrayBuffer | Uint8Array, msg: string): Promise<ArrayBuffer> {
  const rawKey: ArrayBuffer = key instanceof Uint8Array ? key.buffer as ArrayBuffer : key
  const k = await crypto.subtle.importKey(
    'raw', rawKey, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'],
  )
  return crypto.subtle.sign('HMAC', k, new TextEncoder().encode(msg))
}

async function signingKey(secret: string, date: string, region: string): Promise<ArrayBuffer> {
  let k = await hmac(new TextEncoder().encode('AWS4' + secret), date)
  k = await hmac(k, region)
  k = await hmac(k, 's3')
  return hmac(k, 'aws4_request')
}

function canonicalPath(path: string): string {
  // Encode each segment but preserve slashes between segments
  return path.split('/').map(s => (s === '' ? '' : encodeURIComponent(s))).join('/') || '/'
}

/* ── Signed fetch ────────────────────────────────────────────── */
export async function s3Fetch(
  method: string,
  path: string,                                // e.g. '/', '/bucket', '/bucket/key'
  query: Record<string, string> = {},
  body: ArrayBuffer | null = null,
  unsignedHeaders: Record<string, string> = {}, // e.g. content-type — not signed
  extraSignedHeaders: Record<string, string> = {}, // e.g. x-amz-copy-source — included in signature
): Promise<Response> {
  const { apiUrl, accessKey, secretKey } = loadMinioConfig()
  const minioHost = new URL(apiUrl).host
  const region = 'us-east-1'

  const now = new Date()
  // Format: 20240101T120000Z
  const amzDate = now.toISOString().replace(/[-:]/g, '').slice(0, 15) + 'Z'
  const dateStamp = amzDate.slice(0, 8)

  const payloadHash = toHex(await sha256(body ?? new ArrayBuffer(0)))

  // Canonical query string: sorted, URI-encoded
  const qs = Object.entries(query)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&')

  // Headers included in signature (must match what nginx forwards to MinIO)
  const toSign: Record<string, string> = {
    host:                   minioHost,
    'x-amz-content-sha256': payloadHash,
    'x-amz-date':           amzDate,
    ...Object.fromEntries(Object.entries(extraSignedHeaders).map(([k, v]) => [k.toLowerCase(), v])),
  }
  const sortedKeys      = Object.keys(toSign).sort()
  const canonicalHeaders = sortedKeys.map(k => `${k}:${toSign[k]}\n`).join('')
  const signedHeadersList = sortedKeys.join(';')

  const canonicalRequest = [
    method,
    canonicalPath(path),
    qs,
    canonicalHeaders,
    signedHeadersList,
    payloadHash,
  ].join('\n')

  const credentialScope = `${dateStamp}/${region}/s3/aws4_request`
  const stringToSign = [
    'AWS4-HMAC-SHA256',
    amzDate,
    credentialScope,
    toHex(await sha256(canonicalRequest)),
  ].join('\n')

  const sig = toHex(await hmac(await signingKey(secretKey, dateStamp, region), stringToSign))
  const authHeader = `AWS4-HMAC-SHA256 Credential=${accessKey}/${credentialScope}, SignedHeaders=${signedHeadersList}, Signature=${sig}`

  return fetch(`/api/minio${path}${qs ? '?' + qs : ''}`, {
    method,
    headers: {
      Authorization:            authHeader,
      'x-amz-date':             amzDate,
      'x-amz-content-sha256':   payloadHash,
      ...extraSignedHeaders,
      ...unsignedHeaders,
    },
    body: body ?? undefined,
  })
}

/* ── Types ───────────────────────────────────────────────────── */
export interface BucketInfo {
  name:         string
  creationDate: string
  objectCount:  number | null   // null while loading
  sizeBytes:    number | null
}

export interface ObjectInfo {
  key:          string
  size:         number
  lastModified: string
  etag:         string
}

/* ── XML helper ──────────────────────────────────────────────── */
function xml(text: string): Document {
  return new DOMParser().parseFromString(text, 'text/xml')
}

/* ── S3 operations ───────────────────────────────────────────── */

/** List all buckets (name + creation date only). */
export async function listBuckets(): Promise<Pick<BucketInfo, 'name' | 'creationDate'>[]> {
  const res = await s3Fetch('GET', '/')
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    const code = xml(text).querySelector('Code')?.textContent ?? ''
    throw new Error(code || `ListBuckets ${res.status}`)
  }
  const doc = xml(await res.text())
  return [...doc.querySelectorAll('Bucket')].map(b => ({
    name:         b.querySelector('Name')?.textContent ?? '',
    creationDate: b.querySelector('CreationDate')?.textContent ?? '',
  }))
}

/** List objects in a bucket (handles pagination up to 10 000 objects). */
export async function listObjects(bucket: string, prefix = ''): Promise<ObjectInfo[]> {
  const all: ObjectInfo[] = []
  let token = ''
  let pages = 0

  do {
    const q: Record<string, string> = { 'list-type': '2', 'max-keys': '1000' }
    if (prefix) q.prefix = prefix
    if (token)  q['continuation-token'] = token

    const res = await s3Fetch('GET', `/${bucket}`, q)
    if (!res.ok) throw new Error(`ListObjects ${res.status}`)
    const doc = xml(await res.text())

    for (const c of doc.querySelectorAll('Contents')) {
      all.push({
        key:          c.querySelector('Key')?.textContent ?? '',
        size:         parseInt(c.querySelector('Size')?.textContent ?? '0', 10),
        lastModified: c.querySelector('LastModified')?.textContent ?? '',
        etag:         (c.querySelector('ETag')?.textContent ?? '').replace(/"/g, ''),
      })
    }

    const truncated = doc.querySelector('IsTruncated')?.textContent?.toLowerCase() === 'true'
    token = truncated ? (doc.querySelector('NextContinuationToken')?.textContent ?? '') : ''
    pages++
  } while (token && pages < 10)

  return all
}

/** Create a new bucket. */
export async function createBucket(name: string): Promise<void> {
  const res = await s3Fetch('PUT', `/${name}`)
  if (!res.ok) {
    const code = xml(await res.text()).querySelector('Code')?.textContent
    throw new Error(code ?? `CreateBucket ${res.status}`)
  }
}

/** Delete an empty bucket. */
export async function deleteBucket(name: string): Promise<void> {
  const res = await s3Fetch('DELETE', `/${name}`)
  if (!res.ok) {
    const code = xml(await res.text()).querySelector('Code')?.textContent
    throw new Error(code ?? `DeleteBucket ${res.status}`)
  }
}

/** Copy an object within or between buckets (used for bucket rename). */
export async function copyObject(srcBucket: string, srcKey: string, destBucket: string, destKey: string): Promise<void> {
  const encodedSrc  = srcKey.split('/').map(encodeURIComponent).join('/')
  const encodedDest = destKey.split('/').map(encodeURIComponent).join('/')
  const res = await s3Fetch(
    'PUT',
    `/${destBucket}/${encodedDest}`,
    {},
    null,
    {},
    { 'x-amz-copy-source': `/${srcBucket}/${encodedSrc}` },
  )
  if (!res.ok) {
    const code = xml(await res.text()).querySelector('Code')?.textContent
    throw new Error(code ?? `CopyObject ${res.status}`)
  }
}

/** Delete a single object. */
export async function deleteObject(bucket: string, key: string): Promise<void> {
  const encodedKey = key.split('/').map(encodeURIComponent).join('/')
  const res = await s3Fetch('DELETE', `/${bucket}/${encodedKey}`)
  if (!res.ok) throw new Error(`DeleteObject ${res.status}`)
}

/** Upload a File to a bucket. */
export async function uploadObject(bucket: string, key: string, file: File): Promise<void> {
  const buf = await file.arrayBuffer()
  const encodedKey = key.split('/').map(encodeURIComponent).join('/')
  const res = await s3Fetch(
    'PUT',
    `/${bucket}/${encodedKey}`,
    {},
    buf,
    { 'content-type': file.type || 'application/octet-stream' },
  )
  if (!res.ok) throw new Error(`Upload ${res.status}`)
}

/* ── Admin API ───────────────────────────────────────────────── */
export interface DiskInfo {
  totalBytes: number
  usedBytes:  number
  freeBytes:  number
}

/** Get disk capacity from MinIO Prometheus metrics (public, no auth). Returns null if unavailable. */
export async function fetchDiskInfo(): Promise<DiskInfo | null> {
  try {
    const res = await fetch('/api/minio-metrics')
    if (!res.ok) return null
    const text = await res.text()

    // Parse Prometheus text format
    function metric(name: string): number {
      const m = text.match(new RegExp(`^${name}(?:{[^}]*})? ([\\d.e+]+)`, 'm'))
      return m ? parseFloat(m[1]) : 0
    }

    const totalBytes = metric('minio_cluster_capacity_usable_total_bytes')
    const freeBytes  = metric('minio_cluster_capacity_usable_free_bytes')
    const usedBytes  = metric('minio_cluster_usage_total_bytes') || (totalBytes - freeBytes)

    if (totalBytes === 0) return null
    return { totalBytes, usedBytes, freeBytes }
  } catch {
    return null
  }
}
