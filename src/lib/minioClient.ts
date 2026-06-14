/**
 * MinIO S3-compatible client using AWS Signature V4.
 * Uses @noble/hashes (pure JS) so it works in both secure and non-secure contexts.
 * Signs requests in the browser and routes them through the /api/minio proxy.
 *
 * Config is persisted server-side in the user_prefs table (per-user)
 * via the shared userPrefs hook — no more localStorage.
 */
import { sha256 as nobleSha256 } from '@noble/hashes/sha2.js'
import { hmac as nobleHmac } from '@noble/hashes/hmac.js'
import { md5 as nobleMd5 } from '@noble/hashes/legacy.js'
import { getPrefOr, deletePref } from './userPrefs'

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
  // (defensive — also done if a user manually copies a stale URL into
  // a server-side pref)
  const stale = MINIO_KEYS.some(k => {
    const v = getPrefOr(k, '')
    return STALE_HOSTS.some(h => v.includes(h))
  })
  if (stale) {
    MINIO_KEYS.forEach(k => { void deletePref(k) })
  }

  return {
    apiUrl:     getPrefOr('minio_api_url',     MINIO_DEFAULTS.apiUrl),
    accessKey:  getPrefOr('minio_access_key',  MINIO_DEFAULTS.accessKey),
    secretKey:  getPrefOr('minio_secret_key',  MINIO_DEFAULTS.secretKey),
    consoleUrl: getPrefOr('minio_console_url', MINIO_DEFAULTS.consoleUrl),
  }
}

export function saveMinioConfig(cfg: Partial<MinioConfig>): void {
  // Persist via the shared hook — each call PUTs to /api/user-prefs/{key}
  // and updates the in-memory mirror so loadMinioConfig() sees it instantly.
  void import('./userPrefs').then(() => {
    // We can't call useUserPref from a non-React module, so we use the
    // mirror mutator directly:
    const writes: Array<[string, string | null]> = []
    if (cfg.apiUrl     != null) writes.push(['minio_api_url',     cfg.apiUrl])
    if (cfg.accessKey  != null) writes.push(['minio_access_key',  cfg.accessKey])
    if (cfg.secretKey  != null) writes.push(['minio_secret_key',  cfg.secretKey])
    if (cfg.consoleUrl != null) writes.push(['minio_console_url', cfg.consoleUrl])
    for (const [key, value] of writes) {
      void fetch(`/api/user-prefs/${encodeURIComponent(key)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value }),
      })
    }
  })
}

/* ── Crypto helpers ──────────────────────────────────────────── */
function sha256(data: ArrayBuffer | string): ArrayBuffer {
  const buf = typeof data === 'string' ? new TextEncoder().encode(data) : new Uint8Array(data)
  return nobleSha256(buf).buffer as ArrayBuffer
}

function toHex(buf: ArrayBuffer): string {
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('')
}

function hmac(key: ArrayBuffer | Uint8Array, msg: string): ArrayBuffer {
  const k = key instanceof ArrayBuffer ? new Uint8Array(key) : key
  return nobleHmac(nobleSha256, k, new TextEncoder().encode(msg)).buffer as ArrayBuffer
}

/** Base64-encode a raw byte array (used for S3's Content-MD5 header).
 *  SubtleCrypto can't MD5 in modern browsers, so we compute MD5 via
 *  @noble/hashes/legacy and base64-encode here. */
function md5B64(data: ArrayBuffer | Uint8Array): string {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data
  const digest = nobleMd5(bytes)
  let bin = ''
  for (let i = 0; i < digest.length; i++) bin += String.fromCharCode(digest[i])
  return btoa(bin)
}

function signingKey(secret: string, date: string, region: string): ArrayBuffer {
  let k = hmac(new TextEncoder().encode('AWS4' + secret), date)
  k = hmac(k, region)
  k = hmac(k, 's3')
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

  const payloadHash = toHex(sha256(body ?? new ArrayBuffer(0)))

  // Canonical query string: sorted, URI-encoded
  const qs = Object.entries(query)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&')

  // S3's multi-object delete (POST /<bucket>?delete=) requires a Content-MD5
  // header per the S3 spec. SubtleCrypto can't MD5 (it's not in the FIPS
  // allowlist), so we compute it with @noble/hashes/legacy. The header
  // MUST be included in the signature — otherwise MinIO rejects the request
  // with MissingContentMD5 even when the body is otherwise valid.
  const contentMd5 = body ? md5B64(body) : ''

  // Headers included in signature (must match what nginx forwards to MinIO)
  const toSign: Record<string, string> = {
    host:                   minioHost,
    'x-amz-content-sha256': payloadHash,
    'x-amz-date':           amzDate,
    ...(contentMd5 ? { 'content-md5': contentMd5 } : {}),
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
    toHex(sha256(canonicalRequest)),
  ].join('\n')

  const sig = toHex(hmac(signingKey(secretKey, dateStamp, region), stringToSign))
  const authHeader = `AWS4-HMAC-SHA256 Credential=${accessKey}/${credentialScope}, SignedHeaders=${signedHeadersList}, Signature=${sig}`

  return fetch(`/api/minio${path}${qs ? '?' + qs : ''}`, {
    method,
    headers: {
      Authorization:            authHeader,
      'x-amz-date':             amzDate,
      'x-amz-content-sha256':   payloadHash,
      ...(contentMd5 ? { 'Content-MD5': contentMd5 } : {}),
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

export interface DirInfo {
  prefix: string
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

/** List objects in a bucket (handles pagination up to 10 000 objects).
 *  When delimiter is set, also returns common prefixes (subdirectories). */
export async function listObjects(bucket: string, prefix = '', delimiter = ''): Promise<{ objects: ObjectInfo[]; dirs: DirInfo[] }> {
  const all: ObjectInfo[] = []
  const dirs: DirInfo[] = []
  let token = ''
  let pages = 0

  do {
    const q: Record<string, string> = { 'list-type': '2', 'max-keys': '1000' }
    if (prefix)    q.prefix = prefix
    if (delimiter) q.delimiter = delimiter
    if (token)     q['continuation-token'] = token

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

    for (const cp of doc.querySelectorAll('CommonPrefixes Prefix')) {
      const p = cp.textContent ?? ''
      if (p && p !== prefix) dirs.push({ prefix: p })
    }

    const truncated = doc.querySelector('IsTruncated')?.textContent?.toLowerCase() === 'true'
    token = truncated ? (doc.querySelector('NextContinuationToken')?.textContent ?? '') : ''
    pages++
  } while (token && pages < 10)

  return { objects: all, dirs }
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

/** Delete multiple objects in a single S3 request (max 1000 keys per call). */
export async function deleteObjects(bucket: string, keys: string[]): Promise<void> {
  if (keys.length === 0) return
  // S3 multi-object delete caps at 1000 keys per request.
  for (let i = 0; i < keys.length; i += 1000) {
    const chunk = keys.slice(i, i + 1000)
    const objects = chunk.map(k => `<Object><Key>${k.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</Key></Object>`).join('')
    const body = `<Delete xmlns="http://s3.amazonaws.com/doc/2006-03-01/">${objects}</Delete>`
    const res = await s3Fetch(
      'POST',
      `/${bucket}`,
      { delete: '' },
      new TextEncoder().encode(body).buffer as ArrayBuffer,
      { 'content-type': 'application/xml' },
      {},
    )
    if (!res.ok) {
      const doc = xml(await res.text().catch(() => '<x/>'))
      const code = doc.querySelector('Code')?.textContent
      throw new Error(code ?? `DeleteObjects ${res.status}`)
    }
  }
}

/** Recursively delete every object under a directory prefix.
 *  S3 has no "delete directory" primitive — we list with delimiter='' to
 *  flatten the subtree, then chunked-delete the resulting keys. */
export async function deletePrefix(bucket: string, prefix: string): Promise<number> {
  const normalized = prefix.endsWith('/') ? prefix : prefix + '/'
  const { objects } = await listObjects(bucket, normalized, '')
  const keys = objects.map(o => o.key)
  await deleteObjects(bucket, keys)
  return keys.length
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
