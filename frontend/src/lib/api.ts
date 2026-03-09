export interface SwingMetrics {
  hip_rotation_degrees: number | null;
  shoulder_rotation_degrees: number | null;
  x_factor_degrees: number | null;
  spine_tilt_address_degrees: number | null;
  spine_tilt_change_degrees: number | null;
  lead_knee_flex_address_degrees: number | null;
  lead_knee_flex_impact_degrees: number | null;
  tempo_ratio: number | null;
  backswing_duration_seconds: number | null;
  downswing_duration_seconds: number | null;
  total_swing_duration_seconds: number | null;
}

export interface SwingFault {
  fault_type: string;
  phase: string;
  description: string;
  severity: number;
  correction: string;
}

export type SwingPhase =
  | "address"
  | "takeaway"
  | "backswing"
  | "top"
  | "downswing"
  | "impact"
  | "follow_through"
  | "finish";

export const PHASE_LABELS: Record<SwingPhase, string> = {
  address: "Address",
  takeaway: "Takeaway",
  backswing: "Backswing",
  top: "Top",
  downswing: "Downswing",
  impact: "Impact",
  follow_through: "Follow-through",
  finish: "Finish",
};

export const PHASE_ORDER: SwingPhase[] = [
  "address",
  "takeaway",
  "backswing",
  "top",
  "downswing",
  "impact",
  "follow_through",
  "finish",
];

export interface AnalysisResponse {
  session_id: string;
  video_duration_seconds: number;
  fps: number;
  swing_phases: Record<SwingPhase, number>;
  metrics: SwingMetrics;
  faults: SwingFault[];
  overall_score: number;
  summary: string;
  frame_count: number;
}

export interface SessionSummary {
  session_id: string;
  created_at: string;
}

const BASE = "/api/v1/analysis";

export async function uploadVideo(file: File): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("video", file);
  const res = await fetch(BASE + "/", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || `Error ${res.status}`);
  }
  return res.json();
}

export async function getSession(id: string): Promise<AnalysisResponse> {
  const res = await fetch(`${BASE}/${id}`);
  if (!res.ok) throw new Error("Session not found");
  return res.json();
}

export async function listSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${BASE}/sessions`);
  if (!res.ok) throw new Error("Failed to load sessions");
  return res.json();
}

// ── Pro Comparison ──────────────────────────────────────────────────

export interface ProProfile {
  pro_id: string;
  name: string;
  tour: string;
  swing_style: string;
  known_for: string[];
  thumbnail_url?: string;
}

export interface SwingMatchResult {
  pro: ProProfile;
  similarity_score: number;
  matching_phases: string[];
  key_similarities: string[];
  key_differences: string[];
}

export interface ComparisonResult {
  session_id: string;
  top_matches: SwingMatchResult[];
  primary_match: SwingMatchResult;
  swing_archetype: string;
}

export interface TourMetricComparison {
  user: number;
  tour_avg: number;
  diff: number;
  label: string;
}

const COMPARISON_BASE = "/api/v1/comparison";

export async function getProComparison(sessionId: string): Promise<ComparisonResult> {
  const res = await fetch(`${COMPARISON_BASE}/${sessionId}`);
  if (!res.ok) throw new Error("Comparison not available");
  return res.json();
}

export async function getTourComparison(
  sessionId: string,
  tour: string = "PGA"
): Promise<Record<string, TourMetricComparison>> {
  const res = await fetch(`${COMPARISON_BASE}/${sessionId}/tour/${tour}`);
  if (!res.ok) throw new Error("Tour comparison not available");
  return res.json();
}
