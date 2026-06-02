import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import TrainModel from './pages/TrainModel'
import Datasets from './pages/Datasets'
import Projects from './pages/Projects'
import Jobs from './pages/Jobs'
import Models from './pages/Models'
import RayCluster from './pages/RayCluster'
import Storage from './pages/Storage'
import Sidebar from './components/Sidebar'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <div style={{
        display: 'flex',
        minHeight: '100vh',
        background: 'var(--bg-base)',
        width: '100vw',
        overflow: 'hidden',
      }}>
        <Sidebar />
        <main style={{
          flex: 1,
          marginLeft: 240,
          padding: '32px 40px',
          minHeight: '100vh',
          width: 'calc(100vw - 240px)',
          overflowX: 'hidden',
          boxSizing: 'border-box',
        }}>
          <Routes>
            <Route path="/" element={<Navigate to="/projects" replace />} />
            <Route path="/projects"  element={<Projects />} />
            <Route path="/datasets"  element={<Datasets />} />
            <Route path="/train"     element={<TrainModel />} />
            <Route path="/jobs"      element={<Jobs />} />
            <Route path="/models"    element={<Models />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/ray-cluster" element={<RayCluster />} />
            <Route path="/storage" element={<Storage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
