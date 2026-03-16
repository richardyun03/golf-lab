import math
import numpy as np
from app.core.config import Settings
from app.schemas.analysis import SwingFault, SwingPhase, SwingMetrics, PoseKeypoint


MIN_CONFIDENCE = 0.4  # Skip keypoints below this threshold


def _angle_deg(a: tuple, b: tuple, c: tuple) -> float:
    """Compute the angle at point B formed by points A-B-C. Returns degrees [0, 180]."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba * mag_bc == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _kp_dict(keypoints: list[PoseKeypoint]) -> dict[str, PoseKeypoint]:
    return {kp.name: kp for kp in keypoints}


def _pt(kp: PoseKeypoint) -> tuple[float, float]:
    return (kp.x, kp.y)


def _midpoint_kp(a: PoseKeypoint, b: PoseKeypoint) -> tuple[float, float]:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _confident(kps: dict[str, PoseKeypoint], *names: str) -> bool:
    """Check that all named keypoints exist and exceed the confidence threshold."""
    return all(name in kps and kps[name].confidence >= MIN_CONFIDENCE for name in names)


def _dist(a: tuple, b: tuple) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _angle_3d(a: tuple, b: tuple, c: tuple) -> float:
    """Angle in degrees at point B formed by points A-B-C in 3D. Returns [0, 180]."""
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)
    if mag_ba * mag_bc < 1e-9:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


MIN_SEGMENT_LENGTH = 0.03  # Skip angle checks if arm/leg segments are shorter than this (foreshortened)


class PhaseFrames:
    """
    For each swing phase, selects a representative frame and a multi-frame
    window for averaging measurements.

    Representative frame: the frame within the phase window that has the
    highest average keypoint confidence (i.e., MediaPipe was most sure).

    Multi-frame window: N frames centered on the representative, used to
    average geometric measurements and reduce per-frame noise.
    """

    def __init__(
        self,
        keypoints_by_frame: list[list[PoseKeypoint]],
        swing_phases: dict[SwingPhase, int],
        window_radius: int = 2,
        max_phase_window: int = 15,
    ):
        self.keypoints = keypoints_by_frame
        self.phases = swing_phases
        self.window_radius = window_radius
        self.max_phase_window = max_phase_window
        self.n = len(keypoints_by_frame)

        self._rep_frames: dict[SwingPhase, int] = {}
        self._compute_representative_frames()

    def _phase_range(self, phase: SwingPhase) -> tuple[int, int]:
        """
        Return (start, end) frame indices for a phase.

        Capped to a maximum of max_phase_window frames so that long phases
        (like follow-through or finish) don't search hundreds of frames
        for the representative.
        """
        ordered = sorted(self.phases.items(), key=lambda x: x[1])
        for i, (p, start) in enumerate(ordered):
            if p == phase:
                next_phase_start = ordered[i + 1][1] if i + 1 < len(ordered) else self.n
                end = min(next_phase_start, start + self.max_phase_window)
                return start, end
        return 0, 0

    def _compute_representative_frames(self):
        for phase in self.phases:
            start, end = self._phase_range(phase)
            if start >= end:
                self._rep_frames[phase] = start
                continue

            # Score = confidence - proximity penalty.
            # This biases toward frames near the phase start so fast phases
            # (like impact) don't drift into the next phase visually.
            phase_len = end - start
            best_frame = start
            best_score = -1.0
            for i in range(start, min(end, self.n)):
                kps = self.keypoints[i]
                if not kps:
                    continue
                avg_conf = sum(kp.confidence for kp in kps) / len(kps)
                # Penalty grows linearly: 0 at start, 0.1 at end of window
                proximity_penalty = 0.1 * (i - start) / max(phase_len, 1)
                score = avg_conf - proximity_penalty
                if score > best_score:
                    best_score = score
                    best_frame = i

            self._rep_frames[phase] = best_frame

    def rep_frame(self, phase: SwingPhase) -> int:
        """The single best frame for this phase."""
        return self._rep_frames.get(phase, self.phases.get(phase, 0))

    def rep_kps(self, phase: SwingPhase) -> dict[str, PoseKeypoint] | None:
        """Keypoint dict for the representative frame."""
        idx = self.rep_frame(phase)
        if 0 <= idx < self.n and self.keypoints[idx]:
            return _kp_dict(self.keypoints[idx])
        return None

    def window_frames(self, phase: SwingPhase) -> list[int]:
        """Frame indices for the multi-frame averaging window."""
        center = self.rep_frame(phase)
        start = max(0, center - self.window_radius)
        end = min(self.n, center + self.window_radius + 1)
        return [i for i in range(start, end) if self.keypoints[i]]

    def avg_measurement(self, phase: SwingPhase, measure_fn) -> float | None:
        """
        Average a measurement function across the multi-frame window.

        measure_fn(kps_dict) -> float | None
        Frames where measure_fn returns None are skipped.
        """
        frames = self.window_frames(phase)
        values = []
        for i in frames:
            kps = _kp_dict(self.keypoints[i])
            val = measure_fn(kps)
            if val is not None:
                values.append(val)
        if not values:
            return None
        return float(np.mean(values))


class FaultDetector:
    """
    Detects common swing faults using:
    - Representative frames (highest confidence) per phase
    - Multi-frame averaging to reduce noise
    - Confidence gating to skip unreliable keypoints
    - Relative geometry (joint angles, normalized distances)

    Assumes a right-handed golfer (lead side = left, trail side = right).
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def detect(
        self,
        keypoints_by_frame: list[list[PoseKeypoint]],
        swing_phases: dict[SwingPhase, int],
        metrics: SwingMetrics,
    ) -> list[SwingFault]:
        if not swing_phases or not keypoints_by_frame:
            return []

        pf = PhaseFrames(keypoints_by_frame, swing_phases)
        faults: list[SwingFault] = []

        checks = [
            self._check_sway,
            self._check_early_extension,
            self._check_chicken_wing,
            self._check_casting,
            self._check_head_movement,
        ]

        for check in checks:
            fault = check(pf)
            if fault is not None:
                faults.append(fault)

        return faults

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _torso_height(kps: dict[str, PoseKeypoint]) -> float | None:
        """
        Torso height: distance from shoulder midpoint to hip midpoint.
        Visible in every camera angle, unlike hip/shoulder width which
        collapse in face-on or down-the-line views.
        """
        if not _confident(kps, "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            return None
        sm = _midpoint_kp(kps["left_shoulder"], kps["right_shoulder"])
        hm = _midpoint_kp(kps["left_hip"], kps["right_hip"])
        return _dist(sm, hm)

    # ------------------------------------------------------------------
    # Individual fault checks
    # ------------------------------------------------------------------

    def _check_sway(self, pf: PhaseFrames) -> SwingFault | None:
        """
        Detect lateral hip sway during the backswing.

        Hip midpoint X shift from address to top, normalized by torso height.
        """
        def hip_x(kps):
            if not _confident(kps, "left_hip", "right_hip"):
                return None
            return _midpoint_kp(kps["left_hip"], kps["right_hip"])[0]

        addr_hip_x = pf.avg_measurement(SwingPhase.ADDRESS, hip_x)
        top_hip_x = pf.avg_measurement(SwingPhase.TOP, hip_x)
        torso = pf.avg_measurement(SwingPhase.ADDRESS, self._torso_height)

        if addr_hip_x is None or top_hip_x is None or not torso or torso < 0.05:
            return None

        normalized_shift = abs(top_hip_x - addr_hip_x) / torso

        if normalized_shift > 0.06:
            severity = min(1.0, (normalized_shift - 0.06) / 0.10)
            return SwingFault(
                fault_type="sway",
                phase=SwingPhase.BACKSWING,
                description=f"Lateral hip sway detected during backswing ({normalized_shift:.0%} of torso height). Hips are sliding instead of rotating.",
                severity=severity,
                correction="Keep your weight on the inside of your trail foot. Feel like your trail hip rotates behind you rather than sliding. Drill: place an alignment stick against your trail hip at address and maintain contact.",
            )
        return None

    def _check_early_extension(self, pf: PhaseFrames) -> SwingFault | None:
        """
        Detect early extension (loss of spine angle through impact).

        Compares the spine tilt angle at address vs impact.
        """
        def spine_tilt(kps):
            if not _confident(kps, "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
                return None
            shoulder_mid = _midpoint_kp(kps["left_shoulder"], kps["right_shoulder"])
            hip_mid = _midpoint_kp(kps["left_hip"], kps["right_hip"])
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            return math.degrees(math.atan2(abs(dx), abs(dy)))

        def hip_y(kps):
            if not _confident(kps, "left_hip", "right_hip"):
                return None
            return _midpoint_kp(kps["left_hip"], kps["right_hip"])[1]

        addr_spine = pf.avg_measurement(SwingPhase.ADDRESS, spine_tilt)
        impact_spine = pf.avg_measurement(SwingPhase.IMPACT, spine_tilt)
        addr_hip_y = pf.avg_measurement(SwingPhase.ADDRESS, hip_y)
        impact_hip_y = pf.avg_measurement(SwingPhase.IMPACT, hip_y)

        if addr_spine is None or impact_spine is None:
            return None

        spine_loss = addr_spine - impact_spine
        hip_rise = (addr_hip_y - impact_hip_y) if (addr_hip_y and impact_hip_y) else 0

        if spine_loss > 5 or hip_rise > 0.02:
            severity = min(1.0, max(spine_loss / 15.0, hip_rise / 0.05))
            return SwingFault(
                fault_type="early_extension",
                phase=SwingPhase.DOWNSWING,
                description=f"Early extension detected — spine angle decreased by {spine_loss:.1f} degrees from address to impact. Hips are thrusting toward the ball.",
                severity=severity,
                correction="Maintain your tush line through impact. Feel your hips rotate in place rather than pushing forward. Drill: set up with your glutes touching a chair and keep contact through the swing.",
            )
        return None

    def _check_chicken_wing(self, pf: PhaseFrames) -> SwingFault | None:
        """
        Detect chicken wing (lead arm collapse post-impact).

        Lead elbow angle at follow-through. Extended = ~170-180 deg.
        Skipped when the arm is foreshortened (segments shorter than 20%
        of torso height — arm is pointing toward/away from camera).
        """
        torso = pf.avg_measurement(SwingPhase.FOLLOW_THROUGH, self._torso_height)
        min_seg = max(MIN_SEGMENT_LENGTH, (torso or 0) * 0.40)

        def lead_elbow_angle(kps):
            if not _confident(kps, "left_shoulder", "left_elbow", "left_wrist"):
                return None
            s, e, w = _pt(kps["left_shoulder"]), _pt(kps["left_elbow"]), _pt(kps["left_wrist"])
            if _dist(s, e) < min_seg or _dist(e, w) < min_seg:
                return None
            return _angle_deg(s, e, w)

        angle = pf.avg_measurement(SwingPhase.FOLLOW_THROUGH, lead_elbow_angle)
        if angle is None:
            return None

        if angle < 155:
            severity = min(1.0, (155 - angle) / 35.0)
            return SwingFault(
                fault_type="chicken_wing",
                phase=SwingPhase.FOLLOW_THROUGH,
                description=f"Chicken wing detected — lead elbow angle is {angle:.0f} degrees at follow-through (should be 170+). Lead arm is collapsing instead of extending.",
                severity=severity,
                correction="Focus on extending both arms through the ball. Feel like you're pushing the club toward the target with your lead arm after impact. Drill: hit punch shots focusing on arm extension.",
            )
        return None

    def _check_casting(self, pf: PhaseFrames) -> SwingFault | None:
        """
        Detect casting (early release of wrist lag in the downswing).

        Uses 3D world coordinates for the trail (right) wrist hinge angle
        because wrist cock is primarily a depth rotation that collapses in 2D.
        The trail hand is used because the lead hand is occluded at the top.

        Measures: right_elbow → right_WRIST → right_index (3D angle).
        At the top, a properly maintained wrist should be ~160-175 deg (in 3D).
        At impact it naturally drops to ~100-130 deg as the club releases.
        Casting = this drop happens too early (by mid-downswing instead of impact).

        Secondary signal: trail elbow straightening as supporting evidence.
        """
        WRIST_HINGE_MIN_CONF = 0.5  # Trail hand has good visibility

        def trail_wrist_hinge_3d(kps):
            """Trail wrist hinge using 3D world coordinates."""
            re = kps.get("right_elbow")
            rw = kps.get("right_wrist")
            # Try index first, fall back to pinky
            rh = None
            for name in ("right_index", "right_pinky"):
                kp = kps.get(name)
                if kp and kp.confidence >= WRIST_HINGE_MIN_CONF:
                    rh = kp
                    break
            if not re or not rw or not rh:
                return None
            if re.confidence < WRIST_HINGE_MIN_CONF or rw.confidence < WRIST_HINGE_MIN_CONF:
                return None
            if re.z is None or rw.z is None or rh.z is None:
                return None
            return _angle_3d(
                (re.x, re.y, re.z),
                (rw.x, rw.y, rw.z),
                (rh.x, rh.y, rh.z),
            )

        torso = pf.avg_measurement(SwingPhase.TOP, self._torso_height)
        min_seg = max(MIN_SEGMENT_LENGTH, (torso or 0) * 0.30)

        def trail_elbow_angle(kps):
            if not _confident(kps, "right_shoulder", "right_elbow", "right_wrist"):
                return None
            s, e, w = _pt(kps["right_shoulder"]), _pt(kps["right_elbow"]), _pt(kps["right_wrist"])
            if _dist(s, e) < min_seg or _dist(e, w) < min_seg:
                return None
            return _angle_deg(s, e, w)

        # Measure 3D wrist hinge at top, downswing, and impact
        top_wrist = pf.avg_measurement(SwingPhase.TOP, trail_wrist_hinge_3d)
        ds_wrist = pf.avg_measurement(SwingPhase.DOWNSWING, trail_wrist_hinge_3d)
        impact_wrist = pf.avg_measurement(SwingPhase.IMPACT, trail_wrist_hinge_3d)

        # Trail elbow (secondary)
        top_elbow = pf.avg_measurement(SwingPhase.TOP, trail_elbow_angle)
        ds_elbow = pf.avg_measurement(SwingPhase.DOWNSWING, trail_elbow_angle)

        # Primary: 3D wrist hinge loss from top to downswing
        # A good swing holds the hinge — the big drop should happen at/after impact.
        # Casting = the drop happens by mid-downswing.
        if top_wrist is not None and ds_wrist is not None:
            early_release = top_wrist - ds_wrist  # positive = wrist unhinging early

            # Also check how much total release happens (top to impact)
            total_release = (top_wrist - impact_wrist) if impact_wrist is not None else None

            # Casting: significant release by downswing relative to total release
            if total_release is not None and total_release > 15:
                # What fraction of the total release happened by mid-downswing?
                early_fraction = early_release / total_release if total_release > 0 else 0
                # More than 50% of the release by mid-downswing = casting
                if early_fraction > 0.50 and early_release > 15:
                    severity = min(1.0, (early_fraction - 0.50) / 0.35)

                    # Boost if elbow also releasing
                    elbow_release = (ds_elbow - top_elbow) if (top_elbow and ds_elbow) else None
                    if elbow_release is not None and elbow_release > 15:
                        severity = min(1.0, severity + 0.15)

                    return SwingFault(
                        fault_type="casting",
                        phase=SwingPhase.DOWNSWING,
                        description=(
                            f"Casting detected — {early_fraction:.0%} of wrist release happened by mid-downswing "
                            f"(wrist angle: top {top_wrist:.0f}°, downswing {ds_wrist:.0f}°, impact {impact_wrist:.0f}°). "
                            f"Releasing lag before impact costs clubhead speed."
                        ),
                        severity=severity,
                        correction="Maintain wrist hinge longer into the downswing. Feel the butt of the club pointing at the ball until your hands pass your trail thigh. Drill: slow-motion swings pausing at hip height — check that wrists are still fully hinged.",
                    )

            # Fallback: if we don't have impact data, use absolute threshold
            elif early_release > 25:
                severity = min(1.0, (early_release - 25) / 30.0)
                return SwingFault(
                    fault_type="casting",
                    phase=SwingPhase.DOWNSWING,
                    description=(
                        f"Casting detected — wrist angle dropped {early_release:.0f}° by mid-downswing "
                        f"(top: {top_wrist:.0f}°, downswing: {ds_wrist:.0f}°). "
                        f"Releasing lag before impact costs clubhead speed."
                    ),
                    severity=severity,
                    correction="Maintain wrist hinge longer into the downswing. Feel the butt of the club pointing at the ball until your hands pass your trail thigh. Drill: slow-motion swings pausing at hip height — check that wrists are still fully hinged.",
                )

        # Last resort: elbow-only with strict threshold (no 3D wrist data)
        elbow_release = None
        if top_elbow is not None and ds_elbow is not None:
            elbow_release = ds_elbow - top_elbow
        if top_wrist is None and elbow_release is not None and elbow_release > 30:
            severity = min(1.0, (elbow_release - 30) / 30.0) * 0.8
            return SwingFault(
                fault_type="casting",
                phase=SwingPhase.DOWNSWING,
                description=(
                    f"Possible casting — trail arm straightened {elbow_release:.0f}° by mid-downswing "
                    f"(top: {top_elbow:.0f}°, downswing: {ds_elbow:.0f}°). "
                    f"This may indicate early lag release."
                ),
                severity=severity,
                correction="Maintain wrist hinge longer into the downswing. Feel the butt of the club pointing at the ball until your hands pass your trail thigh. Drill: slow-motion swings pausing at hip height — check that wrists are still fully hinged.",
            )

        return None

    def _check_head_movement(self, pf: PhaseFrames) -> SwingFault | None:
        """
        Detect excessive head movement from address through impact.

        Nose position shift normalized by torso height.
        """
        def nose_pos(kps):
            if not _confident(kps, "nose"):
                return None
            return _pt(kps["nose"])

        addr_frames = pf.window_frames(SwingPhase.ADDRESS)
        impact_frames = pf.window_frames(SwingPhase.IMPACT)

        addr_positions = [nose_pos(_kp_dict(pf.keypoints[i])) for i in addr_frames]
        impact_positions = [nose_pos(_kp_dict(pf.keypoints[i])) for i in impact_frames]

        addr_positions = [p for p in addr_positions if p is not None]
        impact_positions = [p for p in impact_positions if p is not None]

        if not addr_positions or not impact_positions:
            return None

        addr_nose = (
            np.mean([p[0] for p in addr_positions]),
            np.mean([p[1] for p in addr_positions]),
        )
        impact_nose = (
            np.mean([p[0] for p in impact_positions]),
            np.mean([p[1] for p in impact_positions]),
        )

        torso = pf.avg_measurement(SwingPhase.ADDRESS, self._torso_height)
        if not torso or torso < 0.05:
            return None

        lateral_shift = abs(impact_nose[0] - addr_nose[0]) / torso
        vertical_shift = abs(impact_nose[1] - addr_nose[1]) / torso
        total_movement = math.sqrt(lateral_shift ** 2 + vertical_shift ** 2)

        if total_movement > 0.15:
            severity = min(1.0, (total_movement - 0.15) / 0.25)
            direction = "laterally" if lateral_shift > vertical_shift else "vertically (dipping/rising)"

            return SwingFault(
                fault_type="excessive_head_movement",
                phase=SwingPhase.DOWNSWING,
                description=f"Excessive head movement detected — head moved {direction} by {total_movement:.0%} of torso height from address to impact.",
                severity=severity,
                correction="Keep your head steady as a pivot point for the swing. A slight rotation is fine, but avoid lateral slides or vertical dips. Drill: have someone hold a hand on your head while you make slow swings.",
            )
        return None
