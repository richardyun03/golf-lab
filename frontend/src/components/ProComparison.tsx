import { useEffect, useState } from "react";
import { Loader2, ChevronDown, ChevronUp } from "lucide-react";
import type {
  ComparisonResult,
  SwingMatchResult,
  TourMetricComparison,
} from "../lib/api";
import { getProComparison, getTourComparison } from "../lib/api";

interface Props {
  sessionId: string;
}

function pct(val: number) {
  return Math.round(val * 100);
}

function MatchCard({
  match,
  rank,
  expanded,
  onToggle,
}: {
  match: SwingMatchResult;
  rank: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  const score = pct(match.similarity_score);
  const barColor =
    score >= 75
      ? "bg-emerald-500"
      : score >= 55
        ? "bg-yellow-500"
        : "bg-orange-500";

  return (
    <div className="bg-gray-800/60 rounded-xl overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-4 p-4 text-left hover:bg-gray-800/80 transition-colors"
      >
        <span className="text-2xl font-bold text-gray-600 w-8 text-center shrink-0">
          {rank}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-white font-semibold truncate">
              {match.pro.name}
            </span>
            <span className="text-xs text-gray-500 shrink-0">
              {match.pro.tour}
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-0.5">{match.pro.swing_style}</p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <div className="w-24">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-gray-500">Match</span>
              <span className="text-white font-medium">{score}%</span>
            </div>
            <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${barColor}`}
                style={{ width: `${score}%` }}
              />
            </div>
          </div>
          {expanded ? (
            <ChevronUp size={16} className="text-gray-500" />
          ) : (
            <ChevronDown size={16} className="text-gray-500" />
          )}
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 pt-1 border-t border-gray-700/50">
          {/* Known for */}
          <div className="flex flex-wrap gap-1.5 mb-3">
            {match.pro.known_for.map((trait) => (
              <span
                key={trait}
                className="text-[10px] px-2 py-0.5 rounded-full bg-gray-700 text-gray-300"
              >
                {trait}
              </span>
            ))}
          </div>

          {/* Matching phases */}
          {match.matching_phases.length > 0 && (
            <div className="mb-3">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">
                Matching phases
              </p>
              <div className="flex gap-1.5">
                {match.matching_phases.map((phase) => (
                  <span
                    key={phase}
                    className="text-xs px-2 py-0.5 rounded bg-emerald-900/40 text-emerald-400"
                  >
                    {phase}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            {/* Similarities */}
            {match.key_similarities.length > 0 && (
              <div>
                <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">
                  Similar
                </p>
                <ul className="space-y-1">
                  {match.key_similarities.map((s, i) => (
                    <li key={i} className="text-xs text-emerald-400">
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {/* Differences */}
            {match.key_differences.length > 0 && (
              <div>
                <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">
                  Different
                </p>
                <ul className="space-y-1">
                  {match.key_differences.map((d, i) => (
                    <li key={i} className="text-xs text-orange-400">
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

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
                <span
                  className={
                    isClose ? "text-emerald-400" : "text-orange-400"
                  }
                >
                  {item.diff > 0 ? "+" : ""}
                  {item.diff.toFixed(1)}
                </span>
              </div>
              <div className="relative h-3 flex gap-0.5">
                <div className="flex-1 bg-gray-700 rounded-sm overflow-hidden">
                  <div
                    className="h-full bg-blue-500/70 rounded-sm"
                    style={{ width: `${userPct}%` }}
                  />
                </div>
                <div className="flex-1 bg-gray-700 rounded-sm overflow-hidden">
                  <div
                    className="h-full bg-gray-500/70 rounded-sm"
                    style={{ width: `${tourPct}%` }}
                  />
                </div>
              </div>
              <div className="flex items-center justify-between text-[10px] mt-0.5">
                <span className="text-blue-400">You: {item.user}</span>
                <span className="text-gray-500">
                  Tour: {item.tour_avg}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ProComparison({ sessionId }: Props) {
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [tourData, setTourData] = useState<Record<
    string,
    TourMetricComparison
  > | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedIdx, setExpandedIdx] = useState<number>(0);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getProComparison(sessionId),
      getTourComparison(sessionId),
    ])
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
        <p className="text-gray-500 text-sm">
          {error || "No comparison data available"}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Archetype badge */}
      <div className="bg-gray-800/60 rounded-xl p-4 flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-emerald-900/50 flex items-center justify-center text-emerald-400 text-lg font-bold shrink-0">
          {pct(comparison.primary_match.similarity_score)}
        </div>
        <div>
          <p className="text-white font-semibold text-sm">
            Your swing type:{" "}
            <span className="text-emerald-400">
              {comparison.swing_archetype}
            </span>
          </p>
          <p className="text-xs text-gray-500">
            Closest match: {comparison.primary_match.pro.name}
          </p>
        </div>
      </div>

      {/* Pro matches */}
      <div className="space-y-2">
        {comparison.top_matches.map((match, i) => (
          <MatchCard
            key={match.pro.pro_id}
            match={match}
            rank={i + 1}
            expanded={expandedIdx === i}
            onToggle={() => setExpandedIdx(expandedIdx === i ? -1 : i)}
          />
        ))}
      </div>

      {/* Tour comparison */}
      {tourData && <TourComparisonChart data={tourData} />}
    </div>
  );
}
