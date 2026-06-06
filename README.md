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

Create a `.env` file in the project root:

```env
BACKEND_URL=http://medimage-api:8000
LS_URL=http://localhost:8085
SYNC_URL=http://localhost:8084
RAY_URL=http://localhost:8265
LS_USER=admin@medimage.local
LS_PASSWORD=YourPassword
```

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

## Contributing

1. Follow TypeScript strict mode guidelines
2. Use functional components with React Hooks
3. Maintain design system consistency
4. Write self-documenting code with clear types
5. Test integrations with backend services

## License

Proprietary - Internal use only

## Support

For issues or questions, contact the MedImage development team.

---

**Built with ❤️ by the H-Forge team**
