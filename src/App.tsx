import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useState } from 'react'
import Dashboard from './pages/Dashboard'
import TrainModel from './pages/TrainModel'
import Datasets from './pages/Datasets'
import Projects from './pages/Projects'
import Jobs from './pages/Jobs'
import Models from './pages/Models'
import Notebook from './pages/Notebook'
import RayCluster from './pages/RayCluster'
import Storage from './pages/Storage'
import Sidebar from './components/Sidebar'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <AppInner />
    </BrowserRouter>
  )
}

function AppInner() {
  const location = useLocation()
  const isNotebook = location.pathname === '/notebook'
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const sidebarWidth = sidebarCollapsed ? 68 : 240

  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      background: 'var(--bg-base)',
      width: '100vw',
      overflow: 'hidden',
    }}>
      <Sidebar onCollapsedChange={setSidebarCollapsed} />
      <main style={{
        flex: 1,
        marginLeft: sidebarWidth,
        padding: isNotebook ? 0 : '32px 40px',
        minHeight: '100vh',
        width: `calc(100vw - ${sidebarWidth}px)`,
        overflowX: 'hidden',
        boxSizing: 'border-box',
        transition: 'margin-left 0.25s cubic-bezier(0.4,0,0.2,1), width 0.25s cubic-bezier(0.4,0,0.2,1)',
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
            <Route path="/notebook" element={<Notebook />} />
          </Routes>
        </main>
      </div>
  )
}
