import type { SwingFault, SwingPhase } from "../lib/api";
import { PHASE_LABELS } from "../lib/api";
import { AlertTriangle, ChevronDown, ChevronUp, Dumbbell } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { getTrainingPlan } from "../lib/drills";

interface Props {
  faults: SwingFault[];
  sessionId?: string;
}

function severityColor(s: number) {
  if (s >= 0.7) return "text-red-400 bg-red-400/10 border-red-400/20";
  if (s >= 0.4) return "text-yellow-400 bg-yellow-400/10 border-yellow-400/20";
  return "text-blue-400 bg-blue-400/10 border-blue-400/20";
}

function severityLabel(s: number) {
  if (s >= 0.7) return "High";
  if (s >= 0.4) return "Medium";
  return "Low";
}

export default function FaultsList({ faults, sessionId }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (faults.length === 0) {
    return (
      <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-6 text-center">
        <p className="text-emerald-400 font-medium">No major faults detected</p>
        <p className="text-gray-500 text-sm mt-1">
          Your swing looks solid. Keep it up!
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {faults.map((fault, i) => {
        const open = expanded === i;
        return (
          <div
            key={i}
            className={`border rounded-xl overflow-hidden transition-colors ${severityColor(fault.severity)}`}
          >
            <button
              onClick={() => setExpanded(open ? null : i)}
              className="w-full flex items-center gap-3 px-4 py-3 text-left"
            >
              <AlertTriangle size={16} className="shrink-0" />
              <div className="flex-1 min-w-0">
                <span className="font-medium text-sm">
                  {fault.fault_type.replace(/_/g, " ")}
                </span>
                <span className="text-xs opacity-60 ml-2">
                  {PHASE_LABELS[fault.phase as SwingPhase] ?? fault.phase}
                </span>
              </div>
              <span className="text-xs font-medium opacity-70 mr-2">
                {severityLabel(fault.severity)}
              </span>
              {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {open && (
              <div className="px-4 pb-4 space-y-2 text-sm">
                <p className="opacity-80">{fault.description}</p>
                <div className="bg-black/20 rounded-lg p-3">
                  <p className="text-xs font-semibold uppercase tracking-wider opacity-50 mb-1">
                    How to fix
                  </p>
                  <p className="opacity-90">{fault.correction}</p>
                </div>
                {/* Severity bar */}
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs opacity-50">Severity</span>
                  <div className="flex-1 h-1.5 bg-black/20 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-current transition-all"
                      style={{ width: `${fault.severity * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono">
                    {(fault.severity * 100).toFixed(0)}%
                  </span>
                </div>
                {getTrainingPlan(fault.fault_type) && (
                  <Link
                    to={`/training?fault=${fault.fault_type}${sessionId ? `&session=${sessionId}` : ""}`}
                    className="mt-2 inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400 hover:text-emerald-300 transition-colors"
                  >
                    <Dumbbell size={12} />
                    Practice drills for this fault
                  </Link>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
