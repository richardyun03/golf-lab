import { PHASE_LABELS, PHASE_ORDER, type SwingPhase } from "../lib/api";

interface Props {
  phaseScores?: Record<string, number>;
}

function grade(score: number): { letter: string; color: string } {
  if (score >= 85) return { letter: "A", color: "text-emerald-400 bg-emerald-400/10 border-emerald-400/30" };
  if (score >= 70) return { letter: "B", color: "text-emerald-300 bg-emerald-300/10 border-emerald-300/20" };
  if (score >= 55) return { letter: "C", color: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20" };
  if (score >= 40) return { letter: "D", color: "text-orange-400 bg-orange-400/10 border-orange-400/20" };
  return { letter: "F", color: "text-red-400 bg-red-400/10 border-red-400/20" };
}

export default function PhaseScoreCard({ phaseScores }: Props) {
  if (!phaseScores || Object.keys(phaseScores).length === 0) return null;

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl px-4 py-3">
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2.5">
        Phase Grades
      </p>
      <div className="flex gap-1.5 overflow-x-auto">
        {PHASE_ORDER.map((phase) => {
          const score = phaseScores[phase];
          if (score === undefined) return null;
          const g = grade(score);
          return (
            <div
              key={phase}
              className={`flex flex-col items-center px-2.5 py-1.5 rounded-lg border min-w-[4rem] ${g.color}`}
            >
              <span className="text-lg font-bold leading-tight">{g.letter}</span>
              <span className="text-[10px] opacity-70 mt-0.5 whitespace-nowrap">
                {PHASE_LABELS[phase as SwingPhase]}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
