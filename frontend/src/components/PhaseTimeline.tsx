import type { SwingPhase } from "../lib/api";
import { PHASE_ORDER, PHASE_LABELS } from "../lib/api";

interface Props {
  phases: Record<SwingPhase, number>;
  frameCount: number;
  activePhase: SwingPhase | null;
  onSelect: (phase: SwingPhase) => void;
}

const PHASE_COLORS: Record<SwingPhase, string> = {
  address: "bg-blue-500",
  takeaway: "bg-cyan-500",
  backswing: "bg-teal-500",
  top: "bg-emerald-500",
  downswing: "bg-yellow-500",
  impact: "bg-orange-500",
  follow_through: "bg-red-500",
  finish: "bg-purple-500",
};

const PHASE_BORDER_COLORS: Record<SwingPhase, string> = {
  address: "border-blue-500",
  takeaway: "border-cyan-500",
  backswing: "border-teal-500",
  top: "border-emerald-500",
  downswing: "border-yellow-500",
  impact: "border-orange-500",
  follow_through: "border-red-500",
  finish: "border-purple-500",
};

export default function PhaseTimeline({
  phases,
  frameCount,
  activePhase,
  onSelect,
}: Props) {
  return (
    <div>
      {/* Bar */}
      <div className="relative h-10 bg-gray-800 rounded-lg overflow-hidden flex">
        {PHASE_ORDER.map((phase, i) => {
          const start = phases[phase];
          const next =
            i < PHASE_ORDER.length - 1 ? phases[PHASE_ORDER[i + 1]] : frameCount;
          const width = ((next - start) / frameCount) * 100;

          return (
            <button
              key={phase}
              onClick={() => onSelect(phase)}
              className={`relative h-full transition-opacity ${PHASE_COLORS[phase]} ${
                activePhase && activePhase !== phase ? "opacity-40" : "opacity-80"
              } hover:opacity-100`}
              style={{ width: `${Math.max(width, 1)}%` }}
              title={`${PHASE_LABELS[phase]} (frame ${start})`}
            />
          );
        })}
      </div>

      {/* Labels */}
      <div className="flex mt-2 gap-1">
        {PHASE_ORDER.map((phase) => (
          <button
            key={phase}
            onClick={() => onSelect(phase)}
            className={`flex-1 text-center text-[11px] py-1.5 rounded-md transition-colors ${
              activePhase === phase
                ? `border ${PHASE_BORDER_COLORS[phase]} bg-gray-800 text-white`
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            {PHASE_LABELS[phase]}
          </button>
        ))}
      </div>
    </div>
  );
}
