import type { SwingMetrics } from "../lib/api";

interface Props {
  metrics: SwingMetrics;
}

interface MetricRowProps {
  label: string;
  value: number | null;
  unit: string;
  good?: [number, number]; // [min, max] range for "good" indicator
}

function MetricRow({ label, value, unit, good }: MetricRowProps) {
  if (value === null) return null;

  let indicator = "text-gray-400";
  if (good) {
    if (value >= good[0] && value <= good[1]) indicator = "text-emerald-400";
    else indicator = "text-yellow-400";
  }

  return (
    <div className="flex items-center justify-between py-2.5 border-b border-gray-800 last:border-0">
      <span className="text-gray-400 text-sm">{label}</span>
      <span className={`font-mono font-medium ${indicator}`}>
        {typeof value === "number" ? value.toFixed(1) : value}
        <span className="text-gray-600 text-xs ml-1">{unit}</span>
      </span>
    </div>
  );
}

export default function MetricsPanel({ metrics }: Props) {
  return (
    <div className="space-y-6">
      {/* Rotation */}
      <div>
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
          Rotation
        </h3>
        <div className="bg-gray-900 rounded-xl px-4">
          <MetricRow
            label="Shoulder rotation"
            value={metrics.shoulder_rotation_degrees}
            unit="deg"
            good={[80, 110]}
          />
          <MetricRow
            label="Hip rotation"
            value={metrics.hip_rotation_degrees}
            unit="deg"
            good={[35, 55]}
          />
          <MetricRow
            label="X-Factor"
            value={metrics.x_factor_degrees}
            unit="deg"
            good={[35, 55]}
          />
        </div>
      </div>

      {/* Spine */}
      <div>
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
          Spine & Posture
        </h3>
        <div className="bg-gray-900 rounded-xl px-4">
          <MetricRow
            label="Spine tilt at address"
            value={metrics.spine_tilt_address_degrees}
            unit="deg"
          />
          <MetricRow
            label="Spine tilt change"
            value={metrics.spine_tilt_change_degrees}
            unit="deg"
          />
        </div>
      </div>

      {/* Lower body */}
      <div>
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
          Lower Body
        </h3>
        <div className="bg-gray-900 rounded-xl px-4">
          <MetricRow
            label="Lead knee flex (address)"
            value={metrics.lead_knee_flex_address_degrees}
            unit="deg"
          />
          <MetricRow
            label="Lead knee flex (impact)"
            value={metrics.lead_knee_flex_impact_degrees}
            unit="deg"
          />
        </div>
      </div>

      {/* Tempo */}
      <div>
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
          Tempo
        </h3>
        <div className="bg-gray-900 rounded-xl px-4">
          <MetricRow
            label="Tempo ratio"
            value={metrics.tempo_ratio}
            unit=":1"
            good={[2.5, 3.5]}
          />
          <MetricRow
            label="Backswing"
            value={metrics.backswing_duration_seconds}
            unit="s"
          />
          <MetricRow
            label="Downswing"
            value={metrics.downswing_duration_seconds}
            unit="s"
          />
          <MetricRow
            label="Total swing"
            value={metrics.total_swing_duration_seconds}
            unit="s"
          />
        </div>
      </div>
    </div>
  );
}
