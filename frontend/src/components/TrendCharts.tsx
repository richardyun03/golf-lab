import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import type { SessionTrendPoint } from "../lib/api";

interface Props {
  data: SessionTrendPoint[];
}

const METRIC_OPTIONS = [
  { key: "overall_score", label: "Overall Score", color: "#34d399" },
  { key: "tempo_ratio", label: "Tempo Ratio", color: "#60a5fa" },
  { key: "x_factor_degrees", label: "X-Factor", color: "#f59e0b" },
  { key: "hip_rotation_degrees", label: "Hip Rotation", color: "#a78bfa" },
  { key: "shoulder_rotation_degrees", label: "Shoulder Rotation", color: "#f472b6" },
  { key: "spine_tilt_change_degrees", label: "Spine Tilt Change", color: "#fb923c" },
] as const;

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-gray-400 mb-1">{label}</p>
      {payload.map((entry: any) => (
        <p key={entry.dataKey} style={{ color: entry.color }}>
          {entry.name}: <span className="font-semibold">{Number(entry.value).toFixed(1)}</span>
        </p>
      ))}
    </div>
  );
}

function ScoreChart({ data }: { data: SessionTrendPoint[] }) {
  const chartData = data.map((d) => ({
    date: formatDate(d.created_at),
    score: d.overall_score,
  }));

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-400 mb-3">Score Over Time</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData}>
          <XAxis
            dataKey="date"
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
            width={35}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="score"
            name="Score"
            stroke="#34d399"
            strokeWidth={2}
            dot={{ fill: "#34d399", r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function MetricsChart({ data }: { data: SessionTrendPoint[] }) {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(["overall_score", "tempo_ratio"])
  );

  const toggle = (key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const chartData = data.map((d) => {
    const point: Record<string, any> = { date: formatDate(d.created_at) };
    for (const opt of METRIC_OPTIONS) {
      if (opt.key === "overall_score") {
        point[opt.key] = d.overall_score;
      } else {
        point[opt.key] = (d.metrics as any)[opt.key] ?? null;
      }
    }
    return point;
  });

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-400 mb-3">Metrics Trend</h3>
      <div className="flex flex-wrap gap-1.5 mb-3">
        {METRIC_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            onClick={() => toggle(opt.key)}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors border ${
              selected.has(opt.key)
                ? "border-current opacity-100"
                : "border-gray-700 opacity-40 hover:opacity-60"
            }`}
            style={{ color: opt.color }}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData}>
          <XAxis
            dataKey="date"
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
            width={35}
          />
          <Tooltip content={<CustomTooltip />} />
          {METRIC_OPTIONS.filter((o) => selected.has(o.key)).map((opt) => (
            <Line
              key={opt.key}
              type="monotone"
              dataKey={opt.key}
              name={opt.label}
              stroke={opt.color}
              strokeWidth={2}
              dot={{ fill: opt.color, r: 3 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function FaultFrequencyChart({ data }: { data: SessionTrendPoint[] }) {
  const counts: Record<string, number> = {};
  for (const d of data) {
    for (const ft of d.fault_types) {
      counts[ft] = (counts[ft] || 0) + 1;
    }
  }

  const chartData = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({
      name: name.replace(/_/g, " "),
      count,
    }));

  if (chartData.length === 0) return null;

  const BAR_COLORS = ["#f87171", "#fb923c", "#fbbf24", "#a78bfa", "#60a5fa", "#34d399", "#f472b6", "#94a3b8"];

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-400 mb-3">Most Common Faults</h3>
      <ResponsiveContainer width="100%" height={Math.max(120, chartData.length * 36)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 10 }}>
          <XAxis
            type="number"
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
            allowDecimals={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={120}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="count" name="Occurrences" radius={[0, 4, 4, 0]} barSize={20}>
            {chartData.map((_, i) => (
              <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} fillOpacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function TrendCharts({ data }: Props) {
  if (data.length < 2) {
    return (
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 text-center">
        <p className="text-gray-500 text-sm">
          Complete at least 2 analyses to see trends.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
        <ScoreChart data={data} />
      </div>
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
        <MetricsChart data={data} />
      </div>
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
        <FaultFrequencyChart data={data} />
      </div>
    </div>
  );
}
