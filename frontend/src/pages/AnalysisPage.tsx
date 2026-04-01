import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft, Loader2, Film, Camera, Users } from "lucide-react";
import type { AnalysisResponse, SwingPhase } from "../lib/api";
import { getSession, CLUB_LABELS, type ClubType } from "../lib/api";
import ScoreRing from "../components/ScoreRing";
import PhaseTimeline from "../components/PhaseTimeline";
import MetricsPanel from "../components/MetricsPanel";
import FaultsList from "../components/FaultsList";
import VideoPlayer from "../components/VideoPlayer";
import SwingVideoPlayer from "../components/SwingVideoPlayer";
import ProComparison from "../components/ProComparison";
import PhaseScoreCard from "../components/PhaseScoreCard";

type ViewMode = "frames" | "video";
type RightTab = "metrics" | "pro";

export default function AnalysisPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activePhase, setActivePhase] = useState<SwingPhase | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("frames");
  const [rightTab, setRightTab] = useState<RightTab>("metrics");

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
            Session {sessionId?.slice(0, 8)}
            {data.club_type && (
              <> &middot; {CLUB_LABELS[data.club_type as ClubType] ?? data.club_type}</>
            )}
            {" "}&middot; {data.video_duration_seconds.toFixed(1)}s &middot;{" "}
            {data.fps.toFixed(0)} fps &middot; {data.frame_count} frames
          </p>
        </div>
        <ScoreRing score={data.overall_score} />
      </div>

      {/* Phase timeline */}
      <div className="mb-4">
        <PhaseTimeline
          phases={data.swing_phases}
          frameCount={data.frame_count}
          activePhase={activePhase}
          onSelect={setActivePhase}
        />
      </div>

      {/* Phase grades */}
      <div className="mb-8">
        <PhaseScoreCard phaseScores={data.phase_scores} />
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left column */}
        <div className="lg:col-span-2">
          {/* View mode toggle */}
          <div className="flex items-center gap-1 mb-3 bg-gray-900 rounded-lg p-1 w-fit">
            <button
              onClick={() => setViewMode("frames")}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === "frames"
                  ? "bg-gray-700 text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              <Camera size={15} />
              Phase Frames
            </button>
            <button
              onClick={() => setViewMode("video")}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === "video"
                  ? "bg-gray-700 text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              <Film size={15} />
              Video
            </button>
          </div>

          {/* Player area */}
          <div className="bg-gray-900/50 rounded-2xl p-4 max-w-xl">
            {viewMode === "frames" ? (
              <VideoPlayer
                sessionId={sessionId!}
                phases={data.swing_phases}
                activePhase={activePhase}
                onPhaseSelect={setActivePhase}
              />
            ) : (
              <SwingVideoPlayer
                videoUrl={videoUrl}
                fps={data.fps}
                phases={data.swing_phases}
                frameCount={data.frame_count}
                activePhase={activePhase}
                onPhaseSelect={setActivePhase}
              />
            )}
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
            <FaultsList faults={data.faults} sessionId={sessionId} />
          </div>
        </div>

        {/* Right column: Metrics / Pro Comparison tabs */}
        <div>
          <div className="flex items-center gap-1 mb-3 bg-gray-900 rounded-lg p-1">
            <button
              onClick={() => setRightTab("metrics")}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                rightTab === "metrics"
                  ? "bg-gray-700 text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              Metrics
            </button>
            <button
              onClick={() => setRightTab("pro")}
              className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                rightTab === "pro"
                  ? "bg-gray-700 text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              <Users size={14} />
              Pro Compare
            </button>
          </div>

          {rightTab === "metrics" ? (
            <MetricsPanel metrics={data.metrics} idealRanges={data.ideal_ranges} clubType={data.club_type} />
          ) : (
            <ProComparison sessionId={sessionId!} phases={data.swing_phases} />
          )}
        </div>
      </div>
    </div>
  );
}
