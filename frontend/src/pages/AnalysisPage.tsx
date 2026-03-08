import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft, Loader2 } from "lucide-react";
import type { AnalysisResponse, SwingPhase } from "../lib/api";
import { getSession } from "../lib/api";
import ScoreRing from "../components/ScoreRing";
import PhaseTimeline from "../components/PhaseTimeline";
import MetricsPanel from "../components/MetricsPanel";
import FaultsList from "../components/FaultsList";
import VideoPlayer from "../components/VideoPlayer";

export default function AnalysisPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activePhase, setActivePhase] = useState<SwingPhase | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    getSession(sessionId)
      .then(setData)
      .catch((err) => setError(err.message));
  }, [sessionId]);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto px-6 py-24 text-center">
        <p className="text-red-400 text-lg">{error}</p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 mt-4 text-gray-400 hover:text-white"
        >
          <ArrowLeft size={16} /> Back to upload
        </Link>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 size={32} className="text-emerald-500 animate-spin" />
      </div>
    );
  }

  // For now, video URL won't be available from API (we'd need to serve it)
  // We'll show a placeholder. In production, the API would return a video URL.
  const videoUrl = `/api/v1/analysis/${sessionId}/video`;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <Link
          to="/"
          className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
        >
          <ArrowLeft size={20} />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">Swing Analysis</h1>
          <p className="text-gray-500 text-sm">
            Session {sessionId?.slice(0, 8)} &middot;{" "}
            {data.video_duration_seconds.toFixed(1)}s &middot; {data.fps.toFixed(0)}{" "}
            fps &middot; {data.frame_count} frames
          </p>
        </div>
        <ScoreRing score={data.overall_score} />
      </div>

      {/* Phase timeline */}
      <div className="mb-8">
        <PhaseTimeline
          phases={data.swing_phases}
          frameCount={data.frame_count}
          activePhase={activePhase}
          onSelect={setActivePhase}
        />
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Video */}
        <div className="lg:col-span-2">
          <div className="bg-gray-900/50 rounded-2xl p-4">
            <VideoPlayer
              videoUrl={videoUrl}
              fps={data.fps}
              phases={data.swing_phases}
              frameCount={data.frame_count}
              activePhase={activePhase}
              onPhaseSelect={setActivePhase}
            />
          </div>

          {/* Summary */}
          <div className="mt-6 bg-gray-900/50 rounded-2xl p-6">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Summary
            </h2>
            <p className="text-gray-300 leading-relaxed">{data.summary}</p>
          </div>

          {/* Faults */}
          <div className="mt-6">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Faults ({data.faults.length})
            </h2>
            <FaultsList faults={data.faults} />
          </div>
        </div>

        {/* Right: Metrics */}
        <div>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">
            Metrics
          </h2>
          <MetricsPanel metrics={data.metrics} />
        </div>
      </div>
    </div>
  );
}
