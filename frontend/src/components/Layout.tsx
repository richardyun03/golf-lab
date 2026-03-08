import { Link, Outlet, useLocation } from "react-router-dom";
import { Upload, History } from "lucide-react";

const NAV = [
  { to: "/", label: "Analyze", icon: Upload },
  { to: "/history", label: "History", icon: History },
];

export default function Layout() {
  const { pathname } = useLocation();

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/" className="text-xl font-bold tracking-tight text-white">
            Golf Lab
          </Link>
          <nav className="flex gap-1">
            {NAV.map(({ to, label, icon: Icon }) => {
              const active = pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    active
                      ? "bg-emerald-600/20 text-emerald-400"
                      : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                  }`}
                >
                  <Icon size={16} />
                  {label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <Outlet />
      </main>
    </div>
  );
}
