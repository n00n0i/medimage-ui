import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Folder, Database, Brain, ListChecks, LayoutDashboard } from 'lucide-react'

const navItems = [
  { to: '/projects',  label: 'Projects',  icon: Folder },
  { to: '/datasets',  label: 'Datasets',  icon: Database },
  { to: '/train',     label: 'Train',     icon: Brain },
  { to: '/jobs',      label: 'Jobs',      icon: ListChecks },
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
]

export default function Navbar() {
  const [active] = useState('datasets')
  void active

  return (
    <nav className="bg-gray-900 border-b border-gray-800 px-6 py-3">
      <div className="flex items-center gap-1">
        <div className="mr-6 flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
            <span className="text-white font-bold text-sm">MI</span>
          </div>
          <span className="font-semibold text-gray-100">MedImage</span>
        </div>
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              }`
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
        <div className="ml-auto flex items-center gap-3">
          <div className="text-right">
            <div className="text-xs text-gray-400">Ray Cluster</div>
            <div className="text-sm font-mono text-indigo-400">100.68.53.118</div>
          </div>
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" title="Connected" />
        </div>
      </div>
    </nav>
  )
}