import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const BACKEND_URL = env.BACKEND_URL || 'http://localhost:8000'
  const LS_URL      = env.LS_URL      || 'http://localhost:8085'
  const SYNC_URL    = env.SYNC_URL    || 'http://localhost:8084'
  const RAY_URL     = env.RAY_URL     || 'http://100.68.53.118:8265'
  const MINIO_URL   = env.MINIO_URL   || 'http://localhost:9000'

  return {
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    host: '0.0.0.0',
    port: 5177,
    proxy: {
      // Ray Dashboard REST API — must come before generic /api catch-all
      '/api/ray': {
        target: RAY_URL,
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/ray/, ''),
      },
      // Label Studio API — must come before generic /api catch-all
      '/api/ls': {
        target: LS_URL,
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/ls/, '/api'),
      },
      // Sync webhook server
      '/api/sync': {
        target: SYNC_URL,
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/sync\/(\d+)\/?$/, '/sync/$1'),
      },
      // MinIO S3 API — must come before generic /api catch-all
      '/api/minio': {
        target: MINIO_URL,
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/minio/, ''),
      },
      // Backend API (projects, jobs, cluster, train …)
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
    },
  },
  }
})