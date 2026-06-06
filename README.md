# MedImage UI

> MLOps Control Center for Medical Imaging AI Teams

MedImage UI is a modern React-based web application that serves as a centralized interface for managing medical imaging machine learning workflows. It integrates Label Studio annotation, MinIO storage, Ray cluster computing, and custom training pipelines into a unified platform.

## Features

### 🏥 Medical Imaging MLOps Platform
- **Label Studio Integration**: Create and manage annotation projects for medical images
- **Dataset Management**: Import from HuggingFace, local sources, and manage MinIO S3 buckets
- **Model Training**: Configure and submit training jobs to Ray cluster
- **Model Registry**: Version control, download, and deploy trained models
- **Inference Playground**: Real-time model inference with bounding box annotation
- **Ray Cluster Monitoring**: Track cluster health, resource usage, and job status
- **API Service Deployment**: Deploy inference endpoints via Ray Serve or Modal

### 🔐 Authentication & Authorization
- **Keycloak Integration**: OIDC/PKCE authentication flow
- **Role-Based Access**: Admin and user role management
- **Auto Token Refresh**: Seamless session management

### 📊 Monitoring & Visualization
- **GPU Metrics**: Real-time GPU utilization and memory tracking
- **Job Pipeline**: Visual stepper for training job progress
- **Storage Browser**: MinIO S3 file management interface
- **System Health**: Service status monitoring dashboard

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| **Frontend Framework** | React | 19.2.6 |
| **Language** | TypeScript | 6.0.2 |
| **Build Tool** | Vite | 8.0.12 |
| **Routing** | React Router DOM | 7.15.1 |
| **Styling** | Tailwind CSS | 4.3.0 |
| **Authentication** | Keycloak | 26.2.4 |
| **Icons** | Lucide React | 1.16.0 |
| **HTTP Client** | Axios | 1.16.1 |

## Project Structure

```
src/
├── App.tsx                 # Root component with routing
├── main.tsx                # Entry point (Keycloak initialization)
├── keycloak.ts             # Keycloak configuration
├── components/
│   ├── Sidebar.tsx         # Main navigation sidebar
│   ├── Navbar.tsx          # Secondary navbar
│   └── GpuMonitor.tsx      # GPU metrics widget
├── pages/                  # Route components
│   ├── Dashboard.tsx       # System overview
│   ├── Projects.tsx        # Label Studio project management
│   ├── Datasets.tsx        # Dataset import/export
│   ├── TrainModel.tsx      # Training job configuration
│   ├── Jobs.tsx            # Job monitoring
│   ├── Models.tsx          # Model registry
│   ├── Playground.tsx      # Model inference interface
│   ├── RayCluster.tsx      # Ray cluster monitoring
│   ├── Storage.tsx         # MinIO S3 browser
│   ├── ApiService.tsx      # API service deployment
│   ├── Notebook.tsx        # Jupyter integration
│   ├── Status.tsx          # System health monitoring
│   ├── Profile.tsx         # User profile settings
│   └── Users.tsx           # User management (admin)
└── lib/
    └── minioClient.ts      # MinIO S3 client
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Docker and Docker Compose (for full stack)
- Access to Keycloak instance
- Ray cluster (optional for local development)

### Environment Variables

Create a `.env` file in the project root for the **frontend (medimage-ui)**:

```env
BACKEND_URL=http://medimage-api:8000
LS_URL=http://localhost:8085
SYNC_URL=http://localhost:8084
RAY_URL=http://localhost:8265
LS_USER=admin@medimage.local
LS_PASSWORD=YourPassword
```

### Backend (`medimage-api`) Environment Variables

The backend reads several env vars that **must match the rest of the stack** or you will get 401 / DNS / auto-login failures. These are set in `docker-compose.yml` under the `medimage-api` service:

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `LS_API_URL` | LS URL from inside the docker network | `http://label-studio:8080` | Server-to-server calls |
| `LS_PUBLIC_URL` | LS URL the user's browser will hit | `LS_API_URL` (broken) | **Must be set to external URL** like `http://100.68.3.42:8085` — used by `/api/ls-goto/{id}` redirects |
| `LS_USER` | LS login email | `admin@medimage.local` | Must match an actual LS user |
| `LS_PASSWORD` | LS login password | `admin` | **Must match the password the LS user was created with** — env var is only honored on first container start, not on subsequent restarts |
| `LS_TOKEN` | LS API token (for direct REST calls) | `''` | Used by dataset sync |
| `KEYCLOAK_JWKS_URL` | Keycloak JWKS endpoint for JWT verification | `http://medimage-keycloak-1:8080/realms/h-forge/protocol/openid-connect/certs` | **MUST include `/kc`** prefix when Keycloak runs with `KC_HTTP_RELATIVE_PATH=/kc` (the default in this compose) — the correct value is `http://medimage-keycloak-1:8080/kc/realms/h-forge/protocol/openid-connect/certs` |
| `KEYCLOAK_ENABLED` | Toggle JWT auth | `true` | Set `false` to bypass auth for single-user/internal use |
| `MINIO_URL` | MinIO S3 endpoint (API server → MinIO) | `http://minio:9000` | Internal Docker hostname — used by the API server only |
| `MINIO_PUBLIC_URL` | MinIO S3 endpoint reachable from **outside** the Docker network (Ray workers, browsers) | `MINIO_URL` | See [MinIO URL for remote Ray workers](#minio-url-for-remote-ray-workers) below |
| `MINIO_HOST_IP` | Explicit IP/host the Ray cluster should use to reach MinIO on port 9000 | _empty_ | Optional override. If unset, the host is derived from `RAY_URL` |
| `RAY_URL` | Ray Dashboard URL | `http://100.68.53.118:8265` | Used as the host fallback for `MINIO_HOST_IP` when the Ray cluster is on the same machine as MinIO |
| `DB_PATH` | SQLite DB path | `/data/jobs.db` | Persisted via `api-data` volume |

> ⚠️ **Critical gotcha:** `LS_PASSWORD` must match the password that the LS user `admin@medimage.local` was *originally* created with. Changing the env var on an already-initialized LS container has **no effect** — LS only reads it on first start. If you change it, the API's `/api/ls-goto/{id}` will silently fail to log the user in and redirect them to the LS login page. See the [Troubleshooting](#troubleshooting) section.

### Development

Install dependencies:
```bash
npm install
```

Start development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5177`

### MinIO URL for remote Ray workers

The MinIO service runs inside the `medimage` Docker network. Its docker-compose
service name `minio` is **not resolvable from Ray workers** that run on a
different host. Training and model-deploy jobs must therefore be told the
*external* address of MinIO.

The API resolves this via `_resolve_minio_url_for_ray()` in `backend/main.py`,
with this precedence:

1. `MINIO_PUBLIC_URL` — if it does **not** contain the docker service name
   `minio` (i.e. it already points at a real external host), it is used as-is.
2. `MINIO_HOST_IP` — explicit IP/host override. **Set this** when the Ray
   cluster is on a different host than docker-compose.
3. Host parsed from `RAY_URL`, port 9000 — works when Ray and MinIO are on
   the same machine (the common case in this stack).
4. `host.docker.internal` — last-ditch fallback.

Example for a split setup (MinIO on `10.0.0.5`, Ray on `10.0.0.6`):
```env
MINIO_HOST_IP=10.0.0.5
RAY_URL=http://10.0.0.6:8265
```

If the Ray cluster is on the same host as MinIO, **no extra env var is
needed** — the host is derived from `RAY_URL` automatically.

### Build

Build for production:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

### Docker Deployment

Run the full stack with Docker Compose:
```bash
docker-compose up -d
```

Services:
- **medimage-ui**: Frontend application (`:8083`)
- **medimage-api**: Backend API (`:8000`)
- **label-studio**: Annotation platform (`:8085`)
- **minio**: S3-compatible storage (`:9000`, `:9001`)
- **jupyter**: Notebook environment

## API Integration

The application integrates with multiple backend services via Vite proxy:

| Endpoint | Service | Description |
|----------|---------|-------------|
| `/api/` | Backend API | Projects, jobs, training, models |
| `/api/ls/` | Label Studio | Annotation projects and tasks |
| `/api/ray/` | Ray Dashboard | Cluster status and monitoring |
| `/api/minio/` | MinIO S3 | Object storage operations |
| `/api/sync/` | Sync Server | Webhook synchronization |

## Design System

MedImage UI follows the **H-Forge Instrument-Panel** design philosophy:

- **Information Density**: Show what matters without clutter
- **State as Signal**: Color and motion communicate status, not decoration
- **Clinical Clarity**: Industry-standard affordances for medical AI workflows
- **OKLCH Color Palette**: Semantic colors for consistent theming
- **Inter + Fira Code**: Professional typography for UI and code

## Authentication Flow

1. Application initializes Keycloak client with PKCE method
2. User redirects to Keycloak login if unauthenticated
3. JWT token automatically injected in all `/api/*` requests
4. Token auto-refreshes when <30 seconds remaining
5. Role-based navigation (platform-admin sees Users page)

## Key Routes

- `/projects` - Label Studio project management
- `/datasets` - Dataset import and synchronization
- `/train` - Training job configuration
- `/jobs` - Job monitoring and status
- `/models` - Model registry and versioning
- `/playground` - Model inference interface
- `/ray-cluster` - Ray cluster monitoring
- `/storage` - MinIO S3 file browser
- `/api-service` - API service deployment
- `/status` - System health dashboard
- `/profile` - User profile and settings
- `/users` - User management (admin only)

## Code Style

- **TypeScript Strict Mode**: Enabled for type safety
- **ESLint**: Code quality enforcement
- **React Hooks**: Functional components with hooks pattern
- **Local State**: Component-level state management (no Redux/Zustand)
- **Polling Pattern**: Real-time updates via setInterval (no WebSocket)

## Performance

- **Code Splitting**: Automatic route-based splitting via Vite
- **Tree Shaking**: ESM modules for optimal bundle size
- **Lazy Loading**: Components load on-demand
- **Module Caching**: Page-level caching for faster navigation

## Security

- ✅ OIDC/PKCE authentication (no client secret in frontend)
- ✅ JWT token in secure Keycloak storage
- ✅ S3 Signature V4 computed browser-side
- ✅ CORS handled via nginx reverse proxy
- ✅ Input validation on forms

## Scripts

```json
{
  "dev": "vite",                    // Start dev server
  "build": "tsc -b && vite build",  // Build for production
  "lint": "eslint .",               // Run ESLint
  "preview": "vite preview"         // Preview production build
}
```

## Troubleshooting

Common errors seen after fresh deploys or container rebuilds:

### `Failed to load jobs: HTTP 401` on Jobs/Models pages

**Cause:** JWT auth middleware is rejecting every `/api/*` call.
**Fix:** Set `KEYCLOAK_JWKS_URL` in `medimage-api` env to include the `/kc` prefix:
```yaml
- KEYCLOAK_JWKS_URL=http://medimage-keycloak-1:8080/kc/realms/h-forge/protocol/openid-connect/certs
```
If you don't want auth at all, set `KEYCLOAK_ENABLED=false`.

### Open in LS fails with `ERR_NAME_NOT_RESOLVED (-105)` pointing to `http://label-studio:8080/...`

**Cause:** `LS_PUBLIC_URL` is not set, so the API falls back to the internal docker hostname which the browser cannot resolve.
**Fix:** Add `LS_PUBLIC_URL=http://<your-external-host>:8085` to `medimage-api` env.

### LS auto-login redirects back to `/user/login/` instead of the project

**Cause:** `LS_PASSWORD` env var doesn't match the password the LS user was created with on first container start. The login silently fails and a useless anonymous sessionid is set.
**Fix:** Use the actual password (default first-start password is `admin`):
```yaml
- LS_USER=admin@medimage.local
- LS_PASSWORD=admin
```

### Playground image goes black after modal inference

**Cause:** The `<img>` element's `display: 'block' → 'none'` toggle (when result type changes) re-decodes the image, briefly setting `naturalWidth=0`. The useEffect then waits for a `load` event that already fired.
**Fix:** Already fixed in `src/pages/Playground.tsx` — image is pre-loaded into a separate `Image()` ref, decoupled from the visible `<img>`.

### Training job fails with `Failed to resolve 'minio'` / `Name or service not known`

**Cause:** A Ray worker tried to download the dataset from
`http://minio:9000/...`. The docker-compose service name `minio` only resolves
inside the `medimage` network, not on a remote Ray cluster.
**Fix:** Set `MINIO_HOST_IP` (or a fully-qualified `MINIO_PUBLIC_URL`) in
`medimage-api` env so the worker reaches MinIO by external IP. See
[MinIO URL for remote Ray workers](#minio-url-for-remote-ray-workers).

### Jobs/Models show "Failed to load" but Playground inference works

**Cause:** The Jobs/Models pages need a valid Keycloak JWT, but Playground might be using a cached model list from before the API was rebuilt. The new API container needs `KEYCLOAK_JWKS_URL` configured (see above) to verify tokens.

## Contributing

1. Follow TypeScript strict mode guidelines
2. Use functional components with React Hooks
3. Maintain design system consistency
4. Write self-documenting code with clear types
5. Test integrations with backend services

## Bug Reports & Issue Tracking

All bugs are tracked as **GitHub Issues** on the [issue tracker](https://github.com/n00n0i/medimage-ui/issues). When you encounter a bug:

1. **Search existing issues first** to avoid duplicates.
2. **Open a new issue** with:
   - Repro steps (URL, what you clicked, what you expected, what you got)
   - Browser + console logs / network tab capture
   - Docker container logs: `docker logs medimage-medimage-api-1 --tail 100`
   - Environment: which env vars are set in `medimage-api`
3. Label it `bug` (and `area:frontend` / `area:backend` / `area:auth` / `area:deployment` if relevant).
4. Trello is used for **project-wide status updates** (e.g. "3 bugs fixed in this sprint") and references issue numbers from GitHub.

## License

Proprietary - Internal use only

## Support

For issues or questions, contact the MedImage development team.

---

**Built with ❤️ by the H-Forge team**
