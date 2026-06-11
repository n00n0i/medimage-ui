import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useState } from 'react'
import Dashboard from './pages/Dashboard'
import TrainDeepLearning from './pages/TrainDeepLearning'
import TrainLLM from './pages/TrainLLM'
import Datasets from './pages/Datasets'
import Projects from './pages/Projects'
import Jobs from './pages/Jobs'
import Models from './pages/Models'
import Notebook from './pages/Notebook'
import RayCluster from './pages/RayCluster'
import ModalCluster from './pages/ModalCluster'
import Storage from './pages/Storage'
import Status from './pages/Status'
import Playground from './pages/Playground'
import ApiService from './pages/ApiService'
import TestAllModels from './pages/TestAllModels'
import DeployModels from './pages/DeployModels'
import Users from './pages/Users'
import Profile from './pages/Profile'
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
            <Route path="/train"     element={<Navigate to="/train/deep-learning" replace />} />
            <Route path="/train/deep-learning" element={<TrainDeepLearning />} />
            <Route path="/train/llm"  element={<TrainLLM />} />
            <Route path="/jobs"      element={<Jobs />} />
            <Route path="/models"    element={<Models />} />
            <Route path="/models/zero-shot" element={<DeployModels />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/ray-cluster" element={<RayCluster />} />
            <Route path="/modal-cluster" element={<ModalCluster />} />
            <Route path="/storage" element={<Storage />} />
            <Route path="/notebook" element={<Notebook />} />
            <Route path="/status" element={<Status />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/test-all" element={<TestAllModels />} />
            <Route path="/api-service" element={<ApiService />} />
            <Route path="/users" element={<Users />} />
            <Route path="/profile" element={<Profile />} />
          </Routes>
        </main>
      </div>
  )
}
