import { useEffect, useState } from "react";
import { Loader2, ChevronLeft, ChevronRight, ChevronDown, ChevronUp, ArrowRight } from "lucide-react";
import type {
  ComparisonResult,
  SwingMatchResult,
  TourMetricComparison,
  SwingPhase,
} from "../lib/api";
import {
  getProComparison,
  getTourComparison,
  getComparisonFrameUrl,
  PHASE_ORDER,
  PHASE_LABELS,
} from "../lib/api";

interface Props {
  sessionId: string;
  phases: Record<SwingPhase, number>;
}

function pct(val: number) {
  return Math.round(val * 100);
}

// ── Match Card (top-level list) ─────────────────────────────────────

function MatchCard({
  match,
  rank,
  onSelect,
}: {
  match: SwingMatchResult;
  rank: number;
  onSelect: () => void;
}) {
  const score = pct(match.similarity_score);
  const barColor =
    score >= 75
      ? "bg-emerald-500"
      : score >= 55
        ? "bg-yellow-500"
        : "bg-orange-500";

  return (
    <button
      onClick={onSelect}
      className="w-full bg-gray-800/60 rounded-xl flex items-center gap-3 p-3 text-left hover:bg-gray-800/80 transition-colors group"
    >
      <span className="text-xl font-bold text-gray-600 w-7 text-center shrink-0">
        {rank}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-white font-semibold text-sm truncate">
            {match.pro.name}
          </span>
          <span className="text-[10px] text-gray-500 shrink-0">
            {match.pro.tour}
          </span>
        </div>
        <p className="text-[11px] text-gray-500 mt-0.5">{match.pro.swing_style}</p>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <div className="w-16">
          <div className="text-xs text-white font-medium text-right mb-0.5">{score}%</div>
          <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
            <div className={`h-full rounded-full ${barColor}`} style={{ width: `${score}%` }} />
          </div>
        </div>
        <ArrowRight size={14} className="text-gray-600 group-hover:text-gray-400 transition-colors" />
      </div>
    </button>
  );
}

// ── Detailed Pro View ───────────────────────────────────────────────

function ProDetail({
  match,
  sessionId,
  phases,
  onBack,
}: {
  match: SwingMatchResult;
  sessionId: string;
  phases: Record<SwingPhase, number>;
  onBack: () => void;
}) {
  const [phaseIdx, setPhaseIdx] = useState(0);
  const current = PHASE_ORDER[phaseIdx];

  const prev = () => setPhaseIdx(Math.max(0, phaseIdx - 1));
  const next = () => setPhaseIdx(Math.min(PHASE_ORDER.length - 1, phaseIdx + 1));

  const comparisonUrl = getComparisonFrameUrl(sessionId, match.pro.pro_id, current);
  const originalUrl = `/api/v1/analysis/${sessionId}/frames/${current}`;
  const [showComparison, setShowComparison] = useState(true);

  const score = pct(match.similarity_score);

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
        >
          <ChevronLeft size={18} />
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-white font-semibold text-sm">{match.pro.name}</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-900/40 text-emerald-400">
              {score}% match
            </span>
          </div>
          <p className="text-[11px] text-gray-500">
            {match.pro.tour} &middot; {match.pro.swing_style}
          </p>
        </div>
      </div>

      {/* Known for tags */}
      <div className="flex flex-wrap gap-1.5">
        {match.pro.known_for.map((trait) => (
          <span
            key={trait}
            className="text-[10px] px-2 py-0.5 rounded-full bg-gray-800 text-gray-400"
          >
            {trait}
          </span>
        ))}
      </div>

      {/* Phase frame viewer */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-500 uppercase tracking-wider font-semibold">
            {PHASE_LABELS[current]}
          </span>
          <div className="flex items-center gap-1 bg-gray-800 rounded-md p-0.5">
            <button
              onClick={() => setShowComparison(true)}
              className={`text-[10px] px-2 py-1 rounded transition-colors ${
                showComparison ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"
              }`}
            >
              vs Pro
            </button>
            <button
              onClick={() => setShowComparison(false)}
              className={`text-[10px] px-2 py-1 rounded transition-colors ${
                !showComparison ? "bg-gray-700 text-white" : "text-gray-500 hover:text-gray-300"
              }`}
            >
              Original
            </button>
          </div>
        </div>

        <div className="relative rounded-xl overflow-hidden bg-black">
          <img
            key={`${showComparison ? "comp" : "orig"}-${current}`}
            src={showComparison ? comparisonUrl : originalUrl}
            alt={`${PHASE_LABELS[current]} - ${showComparison ? "comparison" : "original"}`}
            className="w-full"
          />

          {phaseIdx > 0 && (
            <button
              onClick={prev}
              className="absolute left-1.5 top-1/2 -translate-y-1/2 p-1.5 rounded-full bg-black/50 hover:bg-black/70 text-white transition-colors"
            >
              <ChevronLeft size={20} />
            </button>
          )}
          {phaseIdx < PHASE_ORDER.length - 1 && (
            <button
              onClick={next}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1.5 rounded-full bg-black/50 hover:bg-black/70 text-white transition-colors"
            >
              <ChevronRight size={20} />
            </button>
          )}
        </div>

        {/* Phase thumbnails */}
        <div className="grid grid-cols-8 gap-1 mt-2">
          {PHASE_ORDER.map((phase, i) => (
            <button
              key={phase}
              onClick={() => setPhaseIdx(i)}
              className={`relative rounded-md overflow-hidden transition-all ${
                phaseIdx === i
                  ? "ring-2 ring-emerald-500 ring-offset-1 ring-offset-gray-950"
                  : "opacity-50 hover:opacity-80"
              }`}
            >
              <img
                src={
                  showComparison
                    ? getComparisonFrameUrl(sessionId, match.pro.pro_id, phase)
                    : `/api/v1/analysis/${sessionId}/frames/${phase}`
                }
                alt={PHASE_LABELS[phase]}
                className="w-full aspect-[3/4] object-cover"
              />
              <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-0.5 py-0.5">
                <span className="text-[8px] text-white leading-tight block truncate">
                  {PHASE_LABELS[phase]}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Similarities & Differences */}
      <div className="grid grid-cols-2 gap-3">
        {match.key_similarities.length > 0 && (
          <div className="bg-gray-800/60 rounded-lg p-3">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5 font-semibold">
              Similar
            </p>
            <ul className="space-y-1">
              {match.key_similarities.map((s, i) => (
                <li key={i} className="text-[11px] text-emerald-400">{s}</li>
              ))}
            </ul>
          </div>
        )}
        {match.key_differences.length > 0 && (
          <div className="bg-gray-800/60 rounded-lg p-3">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5 font-semibold">
              Different
            </p>
            <ul className="space-y-1">
              {match.key_differences.map((d, i) => (
                <li key={i} className="text-[11px] text-orange-400">{d}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Matching phases */}
      {match.matching_phases.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">
            Matching phases:
          </span>
          {match.matching_phases.map((phase) => (
            <span
              key={phase}
              className="text-[10px] px-2 py-0.5 rounded bg-emerald-900/40 text-emerald-400"
            >
              {phase}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Tour Comparison Chart ───────────────────────────────────────────

function TourComparisonChart({
  data,
}: {
  data: Record<string, TourMetricComparison>;
}) {
  const entries = Object.values(data);
  if (entries.length === 0) return null;

  return (
    <div className="bg-gray-800/60 rounded-xl p-4">
      <h3 className="text-xs text-gray-500 uppercase tracking-wider font-semibold mb-3">
        vs Tour Average
      </h3>
      <div className="space-y-3">
        {entries.map((item) => {
          const maxVal = Math.max(item.user, item.tour_avg) * 1.2 || 1;
          const userPct = (item.user / maxVal) * 100;
          const tourPct = (item.tour_avg / maxVal) * 100;
          const isClose = Math.abs(item.diff) / item.tour_avg < 0.1;

          return (
            <div key={item.label}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-gray-400">{item.label}</span>
                <span className={isClose ? "text-emerald-400" : "text-orange-400"}>
                  {item.diff > 0 ? "+" : ""}{item.diff.toFixed(1)}
                </span>
              </div>
              <div className="relative h-3 flex gap-0.5">
                <div className="flex-1 bg-gray-700 rounded-sm overflow-hidden">
                  <div className="h-full bg-blue-500/70 rounded-sm" style={{ width: `${userPct}%` }} />
                </div>
                <div className="flex-1 bg-gray-700 rounded-sm overflow-hidden">
                  <div className="h-full bg-gray-500/70 rounded-sm" style={{ width: `${tourPct}%` }} />
                </div>
              </div>
              <div className="flex items-center justify-between text-[10px] mt-0.5">
                <span className="text-blue-400">You: {item.user}</span>
                <span className="text-gray-500">Tour: {item.tour_avg}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────

export default function ProComparison({ sessionId, phases }: Props) {
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [tourData, setTourData] = useState<Record<string, TourMetricComparison> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPro, setSelectedPro] = useState<SwingMatchResult | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([getProComparison(sessionId), getTourComparison(sessionId)])
      .then(([comp, tour]) => {
        setComparison(comp);
        setTourData(tour);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [sessionId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 size={24} className="text-emerald-500 animate-spin" />
      </div>
    );
  }

  if (error || !comparison) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500 text-sm">{error || "No comparison data available"}</p>
      </div>
    );
  }

  // Detailed pro view
  if (selectedPro) {
    return (
      <ProDetail
        match={selectedPro}
        sessionId={sessionId}
        phases={phases}
        onBack={() => setSelectedPro(null)}
      />
    );
  }

  // Overview with all matches
  return (
    <div className="space-y-4">
      {/* Archetype */}
      <div className="bg-gray-800/60 rounded-xl p-4 flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-emerald-900/50 flex items-center justify-center text-emerald-400 text-lg font-bold shrink-0">
          {pct(comparison.primary_match.similarity_score)}
        </div>
        <div>
          <p className="text-white font-semibold text-sm">
            Your swing type:{" "}
            <span className="text-emerald-400">{comparison.swing_archetype}</span>
          </p>
          <p className="text-xs text-gray-500">
            Closest match: {comparison.primary_match.pro.name}
          </p>
        </div>
      </div>

      {/* Pro list — click to drill in */}
      <div>
        <p className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-2">
          Select a pro to compare
        </p>
        <div className="space-y-1.5">
          {comparison.top_matches.map((match, i) => (
            <MatchCard
              key={match.pro.pro_id}
              match={match}
              rank={i + 1}
              onSelect={() => setSelectedPro(match)}
            />
          ))}
        </div>
      </div>

      {/* Tour comparison */}
      {tourData && <TourComparisonChart data={tourData} />}
    </div>
  );
}
