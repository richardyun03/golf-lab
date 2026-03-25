import { useSearchParams, Link } from "react-router-dom";
import { useState } from "react";
import {
  Dumbbell,
  ChevronRight,
  Clock,
  Wrench,
  RotateCcw,
  Lightbulb,
  Check,
  ArrowLeft,
} from "lucide-react";
import {
  FAULT_TRAINING,
  getAllFaultTypes,
  getTrainingPlan,
  type Drill,
  type Difficulty,
  type FaultTrainingPlan,
} from "../lib/drills";

const DIFFICULTY_COLORS: Record<Difficulty, string> = {
  beginner: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
  intermediate: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20",
  advanced: "text-orange-400 bg-orange-400/10 border-orange-400/20",
};

function DrillCard({ drill, index }: { drill: Drill; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());

  const toggleStep = (stepIdx: number) => {
    setCompletedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepIdx)) next.delete(stepIdx);
      else next.add(stepIdx);
      return next;
    });
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-4 px-5 py-4 text-left hover:bg-gray-800/50 transition-colors"
      >
        <div className="w-8 h-8 rounded-lg bg-gray-800 flex items-center justify-center text-sm font-bold text-gray-400 shrink-0">
          {index + 1}
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-white">{drill.name}</div>
          <div className="flex items-center gap-3 mt-1">
            <span
              className={`text-xs px-2 py-0.5 rounded-full border ${DIFFICULTY_COLORS[drill.difficulty]}`}
            >
              {drill.difficulty}
            </span>
            <span className="text-xs text-gray-500 flex items-center gap-1">
              <Clock size={11} />
              {drill.duration}
            </span>
          </div>
        </div>
        <ChevronRight
          size={16}
          className={`text-gray-500 transition-transform ${expanded ? "rotate-90" : ""}`}
        />
      </button>

      {expanded && (
        <div className="px-5 pb-5 space-y-4 border-t border-gray-800 pt-4">
          {/* Equipment */}
          {drill.equipment.length > 0 && (
            <div className="flex items-start gap-2">
              <Wrench size={14} className="text-gray-500 mt-0.5 shrink-0" />
              <div className="text-sm text-gray-400">
                {drill.equipment.join(", ")}
              </div>
            </div>
          )}

          {/* Steps */}
          <div className="space-y-2">
            {drill.steps.map((step, si) => (
              <button
                key={si}
                onClick={() => toggleStep(si)}
                className="w-full flex items-start gap-3 text-left group"
              >
                <div
                  className={`w-5 h-5 rounded-md border mt-0.5 shrink-0 flex items-center justify-center transition-colors ${
                    completedSteps.has(si)
                      ? "bg-emerald-500 border-emerald-500"
                      : "border-gray-700 group-hover:border-gray-500"
                  }`}
                >
                  {completedSteps.has(si) && (
                    <Check size={12} className="text-white" />
                  )}
                </div>
                <span
                  className={`text-sm ${completedSteps.has(si) ? "text-gray-500 line-through" : "text-gray-300"}`}
                >
                  {step}
                </span>
              </button>
            ))}
          </div>

          {/* Why it works */}
          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
              <Lightbulb size={12} />
              Why it works
            </div>
            <p className="text-sm text-gray-400">{drill.why_it_works}</p>
          </div>

          {/* Reps */}
          <div className="flex items-center gap-2">
            <RotateCcw size={13} className="text-gray-500" />
            <span className="text-sm text-gray-400">{drill.reps}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function FaultSection({ plan }: { plan: FaultTrainingPlan }) {
  return (
    <div className="space-y-6">
      {/* Summary */}
      <p className="text-gray-400">{plan.summary}</p>

      {/* Key feels */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 mb-3">
          Key Feels
        </h3>
        <ul className="space-y-2">
          {plan.key_feels.map((feel, i) => (
            <li key={i} className="flex items-start gap-2 text-sm">
              <span className="text-emerald-400 mt-1 shrink-0">&#x2022;</span>
              <span className="text-gray-300">{feel}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Drills */}
      <div>
        <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500 mb-3">
          Practice Drills
        </h3>
        <div className="space-y-3">
          {plan.drills.map((drill, i) => (
            <DrillCard key={drill.id} drill={drill} index={i} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function TrainingPage() {
  const [params, setParams] = useSearchParams();
  const faultParam = params.get("fault");
  const sessionId = params.get("session");

  const allFaults = getAllFaultTypes();
  const activeFault = faultParam && getTrainingPlan(faultParam) ? faultParam : null;
  const activePlan = activeFault ? getTrainingPlan(activeFault) : null;

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex items-center gap-3 mb-8">
        {sessionId ? (
          <Link
            to={`/analysis/${sessionId}`}
            className="p-2 rounded-lg hover:bg-gray-800 transition-colors text-gray-400 hover:text-white"
          >
            <ArrowLeft size={18} />
          </Link>
        ) : (
          <div className="p-2">
            <Dumbbell size={20} className="text-emerald-400" />
          </div>
        )}
        <div>
          <h1 className="text-2xl font-bold text-white">Practice & Drills</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Targeted exercises to fix common swing faults
          </p>
        </div>
      </div>

      {/* Fault selector pills */}
      <div className="flex flex-wrap gap-2 mb-8">
        {allFaults.map((ft) => {
          const plan = FAULT_TRAINING[ft];
          const active = ft === activeFault;
          return (
            <button
              key={ft}
              onClick={() => {
                const next = new URLSearchParams(params);
                if (active) next.delete("fault");
                else next.set("fault", ft);
                setParams(next);
              }}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                active
                  ? "bg-emerald-600/20 text-emerald-400 border border-emerald-500/30"
                  : "bg-gray-900 text-gray-400 border border-gray-800 hover:border-gray-700 hover:text-gray-200"
              }`}
            >
              {plan.title}
            </button>
          );
        })}
      </div>

      {/* Content */}
      {activePlan ? (
        <div>
          <h2 className="text-xl font-bold text-white mb-4">
            {activePlan.title}
          </h2>
          <FaultSection plan={activePlan} />
        </div>
      ) : (
        <div className="space-y-6">
          <p className="text-gray-400">
            Select a fault above to see targeted drills, or browse all training
            plans below.
          </p>
          {allFaults.map((ft) => {
            const plan = FAULT_TRAINING[ft];
            return (
              <button
                key={ft}
                onClick={() => {
                  const next = new URLSearchParams(params);
                  next.set("fault", ft);
                  setParams(next);
                }}
                className="w-full bg-gray-900 border border-gray-800 rounded-xl p-5 text-left hover:border-gray-700 transition-colors group"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-white font-semibold group-hover:text-emerald-400 transition-colors">
                      {plan.title}
                    </h3>
                    <p className="text-sm text-gray-500 mt-1 line-clamp-2">
                      {plan.summary}
                    </p>
                  </div>
                  <ChevronRight
                    size={16}
                    className="text-gray-600 shrink-0 ml-4"
                  />
                </div>
                <div className="flex gap-2 mt-3">
                  {plan.drills.map((d) => (
                    <span
                      key={d.id}
                      className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded"
                    >
                      {d.name}
                    </span>
                  ))}
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
