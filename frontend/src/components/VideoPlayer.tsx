import { useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import type { SwingPhase } from "../lib/api";
import { PHASE_ORDER, PHASE_LABELS } from "../lib/api";

interface Props {
  sessionId: string;
  phases: Record<SwingPhase, number>;
  activePhase: SwingPhase | null;
  onPhaseSelect: (phase: SwingPhase) => void;
}

export default function VideoPlayer({
  sessionId,
  phases,
  activePhase,
  onPhaseSelect,
}: Props) {
  const currentIndex = activePhase ? PHASE_ORDER.indexOf(activePhase) : 0;
  const current = activePhase ?? PHASE_ORDER[0];

  const prev = () => {
    const i = Math.max(0, currentIndex - 1);
    onPhaseSelect(PHASE_ORDER[i]);
  };

  const next = () => {
    const i = Math.min(PHASE_ORDER.length - 1, currentIndex + 1);
    onPhaseSelect(PHASE_ORDER[i]);
  };

  const frameUrl = `/api/v1/analysis/${sessionId}/frames/${current}`;

  return (
    <div>
      {/* Main frame display */}
      <div className="relative rounded-xl overflow-hidden bg-black">
        <img
          src={frameUrl}
          alt={`${PHASE_LABELS[current]} phase`}
          className="w-full"
        />
        {/* Phase label overlay */}
        <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-sm text-white text-sm font-medium px-3 py-1.5 rounded-full">
          {PHASE_LABELS[current]}
        </div>
        {/* Frame number */}
        <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm text-white text-xs font-mono px-3 py-1.5 rounded-full">
          frame {phases[current]}
        </div>

        {/* Arrow navigation */}
        {currentIndex > 0 && (
          <button
            onClick={prev}
            className="absolute left-2 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/50 hover:bg-black/70 text-white transition-colors"
          >
            <ChevronLeft size={24} />
          </button>
        )}
        {currentIndex < PHASE_ORDER.length - 1 && (
          <button
            onClick={next}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/50 hover:bg-black/70 text-white transition-colors"
          >
            <ChevronRight size={24} />
          </button>
        )}
      </div>

      {/* Phase thumbnails */}
      <div className="grid grid-cols-8 gap-1.5 mt-3">
        {PHASE_ORDER.map((phase) => (
          <button
            key={phase}
            onClick={() => onPhaseSelect(phase)}
            className={`relative rounded-lg overflow-hidden transition-all ${
              current === phase
                ? "ring-2 ring-emerald-500 ring-offset-2 ring-offset-gray-950"
                : "opacity-60 hover:opacity-100"
            }`}
          >
            <img
              src={`/api/v1/analysis/${sessionId}/frames/${phase}`}
              alt={PHASE_LABELS[phase]}
              className="w-full aspect-[3/4] object-cover"
            />
            <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-1 py-1">
              <span className="text-[10px] text-white leading-tight block truncate">
                {PHASE_LABELS[phase]}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
