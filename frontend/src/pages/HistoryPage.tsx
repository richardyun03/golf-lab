import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Clock, ArrowRight, Loader2 } from "lucide-react";
import type { SessionSummary } from "../lib/api";
import { listSessions } from "../lib/api";

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listSessions()
      .then(setSessions)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 size={32} className="text-emerald-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-12">
      <h1 className="text-2xl font-bold text-white mb-6">Analysis History</h1>

      {error && (
        <p className="text-red-400 bg-red-400/10 rounded-lg px-4 py-3 mb-4">
          {error}
        </p>
      )}

      {sessions.length === 0 ? (
        <div className="text-center py-16">
          <Clock size={48} className="mx-auto text-gray-700 mb-4" />
          <p className="text-gray-500 text-lg">No analyses yet</p>
          <Link
            to="/"
            className="inline-block mt-4 text-emerald-400 hover:text-emerald-300"
          >
            Upload your first video
          </Link>
        </div>
      ) : (
        <div className="space-y-2">
          {sessions.map((s) => (
            <Link
              key={s.session_id}
              to={`/analysis/${s.session_id}`}
              className="flex items-center justify-between bg-gray-900 hover:bg-gray-800 rounded-xl px-5 py-4 transition-colors group"
            >
              <div>
                <p className="text-white font-medium text-sm">
                  Session {s.session_id.slice(0, 8)}
                </p>
                <p className="text-gray-500 text-xs mt-0.5">
                  {new Date(s.created_at).toLocaleString()}
                </p>
              </div>
              <ArrowRight
                size={16}
                className="text-gray-600 group-hover:text-gray-300 transition-colors"
              />
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
