import math
import numpy as np
from app.schemas.analysis import SwingMetrics, SwingPhase, PoseKeypoint
from ml.swing_analysis.fault_detector import PhaseFrames, _confident, _kp_dict, _pt, _midpoint_kp, _angle_deg, MIN_CONFIDENCE


def _has_z(*keypoints: PoseKeypoint) -> bool:
    """Check that all keypoints have valid z (world) coordinates."""
    return all(kp.z is not None for kp in keypoints)


def compute_metrics(
    keypoints_by_frame: list[list[PoseKeypoint]],
    swing_phases: dict[SwingPhase, int],
    fps: float,
) -> SwingMetrics:
    """
    Compute biomechanical swing metrics from keypoints and phase boundaries.

    Uses 3D world coordinates (x, z from MediaPipe world landmarks) for
    rotation metrics, and 2D image coordinates for spine tilt and knee flex.
    """
    if not swing_phases or not keypoints_by_frame:
        return SwingMetrics()

    pf = PhaseFrames(keypoints_by_frame, swing_phases)

    hip_rot = _hip_rotation(pf)
    shoulder_rot = _shoulder_rotation(pf)
    x_factor = (shoulder_rot - hip_rot) if (shoulder_rot is not None and hip_rot is not None) else None

    spine_addr = _spine_tilt_3d(pf, SwingPhase.ADDRESS)
    spine_impact = _spine_tilt_3d(pf, SwingPhase.IMPACT)
    spine_change = (spine_addr - spine_impact) if (spine_addr is not None and spine_impact is not None) else None

    knee_addr = _lead_knee_flex(pf, SwingPhase.ADDRESS)
    knee_impact = _lead_knee_flex(pf, SwingPhase.IMPACT)

    tempo = _tempo(swing_phases, fps)

    return SwingMetrics(
        hip_rotation_degrees=_round(hip_rot),
        shoulder_rotation_degrees=_round(shoulder_rot),
        x_factor_degrees=_round(x_factor),
        spine_tilt_address_degrees=_round(spine_addr),
        spine_tilt_change_degrees=_round(spine_change),
        lead_knee_flex_address_degrees=_round(knee_addr),
        lead_knee_flex_impact_degrees=_round(knee_impact),
        **tempo,
    )


def _round(val, decimals=1):
    return round(val, decimals) if val is not None else None


# ---------------------------------------------------------------------------
# Rotation metrics (3D world coordinates)
# ---------------------------------------------------------------------------

def _rotation_vector_3d(kps: dict[str, PoseKeypoint], left_name: str, right_name: str) -> tuple[float, float] | None:
    """
    Return the (dx, dz) vector of a body line in the horizontal plane
    using image x and world z coordinates.
    """
    if not _confident(kps, left_name, right_name):
        return None
    left = kps[left_name]
    right = kps[right_name]
    if not _has_z(left, right):
        return None
    dx = right.x - left.x
    dz = right.z - left.z
    return (dx, dz)


def _rotation_from_address_3d(pf: PhaseFrames, left_name: str, right_name: str) -> float | None:
    """
    Compute rotation at top of backswing relative to address using 3D coords.

    Uses dot product between the address and top body-line vectors to get the
    actual angle between them. This avoids atan2 wraparound issues when
    shoulders rotate past 90° and left/right swap apparent positions.
    """
    def measure(kps):
        return _rotation_vector_3d(kps, left_name, right_name)

    # Collect vectors from multi-frame windows and average components
    addr_frames = pf.window_frames(SwingPhase.ADDRESS)
    top_frames = pf.window_frames(SwingPhase.TOP)

    addr_vecs = [measure(_kp_dict(pf.keypoints[i])) for i in addr_frames]
    top_vecs = [measure(_kp_dict(pf.keypoints[i])) for i in top_frames]
    addr_vecs = [v for v in addr_vecs if v is not None]
    top_vecs = [v for v in top_vecs if v is not None]

    if not addr_vecs or not top_vecs:
        return None

    # Average vectors
    ax = np.mean([v[0] for v in addr_vecs])
    az = np.mean([v[1] for v in addr_vecs])
    tx = np.mean([v[0] for v in top_vecs])
    tz = np.mean([v[1] for v in top_vecs])

    # Dot product angle
    dot = ax * tx + az * tz
    mag_a = math.sqrt(ax ** 2 + az ** 2)
    mag_t = math.sqrt(tx ** 2 + tz ** 2)

    if mag_a < 1e-6 or mag_t < 1e-6:
        return None

    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_t)))
    angle = math.degrees(math.acos(cos_angle))

    # Discard implausible values — monocular z estimation can be noisy.
    # Max realistic rotation: ~120° shoulders, ~80° hips.
    if angle > 120:
        return None
    return angle


def _hip_rotation(pf: PhaseFrames) -> float | None:
    return _rotation_from_address_3d(pf, "left_hip", "right_hip")


def _shoulder_rotation(pf: PhaseFrames) -> float | None:
    return _rotation_from_address_3d(pf, "left_shoulder", "right_shoulder")


# ---------------------------------------------------------------------------
# Spine tilt (3D)
# ---------------------------------------------------------------------------

def _spine_tilt_3d(pf: PhaseFrames, phase: SwingPhase) -> float | None:
    """
    Spine tilt from vertical using 3D world coordinates.

    Uses world y (vertical) and world z (depth) + image x (lateral) to
    compute the angle of the spine vector from true vertical.
    0 = perfectly upright, larger = more forward tilt.
    """
    def measure(kps):
        if not _confident(kps, "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            return None
        ls, rs = kps["left_shoulder"], kps["right_shoulder"]
        lh, rh = kps["left_hip"], kps["right_hip"]

        if not _has_z(ls, rs, lh, rh):
            # Fallback to 2D
            shoulder_mid = _midpoint_kp(ls, rs)
            hip_mid = _midpoint_kp(lh, rh)
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            return math.degrees(math.atan2(abs(dx), abs(dy)))

        # 3D spine vector: shoulder midpoint → hip midpoint
        # World coords: x=lateral, y=vertical, z=depth (all in meters)
        smx = (ls.x + rs.x) / 2.0
        smy = (ls.y + rs.y) / 2.0  # image y (proxy for vertical)
        smz = (ls.z + rs.z) / 2.0

        hmx = (lh.x + rh.x) / 2.0
        hmy = (lh.y + rh.y) / 2.0
        hmz = (lh.z + rh.z) / 2.0

        # Spine vector (shoulder - hip, in 3D)
        sx = smx - hmx
        sy = smy - hmy  # vertical component (image y, negative = above)
        sz = smz - hmz

        # Angle from vertical: vertical axis is along y
        # Horizontal displacement = sqrt(sx^2 + sz^2)
        horiz = math.sqrt(sx ** 2 + sz ** 2)
        vert = abs(sy)

        if vert < 0.001:
            return 90.0  # nearly horizontal spine

        return math.degrees(math.atan2(horiz, vert))

    return pf.avg_measurement(phase, measure)


# ---------------------------------------------------------------------------
# Knee flex
# ---------------------------------------------------------------------------

def _lead_knee_flex(pf: PhaseFrames, phase: SwingPhase) -> float | None:
    """
    Lead knee flex angle (left hip → left knee → left ankle).

    Straight leg = ~180 degrees. More flex = lower angle.
    """
    def measure(kps):
        if not _confident(kps, "left_hip", "left_knee", "left_ankle"):
            return None
        return _angle_deg(
            _pt(kps["left_hip"]),
            _pt(kps["left_knee"]),
            _pt(kps["left_ankle"]),
        )

    return pf.avg_measurement(phase, measure)


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------

def _tempo(swing_phases: dict[SwingPhase, int], fps: float) -> dict:
    """
    Compute tempo metrics from phase frame indices.

    Tempo ratio = backswing duration / downswing duration.
    Pro average is roughly 3:1.
    """
    result = {
        "tempo_ratio": None,
        "backswing_duration_seconds": None,
        "downswing_duration_seconds": None,
        "total_swing_duration_seconds": None,
    }

    if fps <= 0:
        return result

    address = swing_phases.get(SwingPhase.ADDRESS)
    takeaway = swing_phases.get(SwingPhase.TAKEAWAY)
    top = swing_phases.get(SwingPhase.TOP)
    impact = swing_phases.get(SwingPhase.IMPACT)
    finish = swing_phases.get(SwingPhase.FINISH)

    if takeaway is not None and top is not None:
        bs_frames = top - takeaway
        result["backswing_duration_seconds"] = round(bs_frames / fps, 2)

    if top is not None and impact is not None:
        ds_frames = impact - top
        result["downswing_duration_seconds"] = round(ds_frames / fps, 2)

    if result["backswing_duration_seconds"] and result["downswing_duration_seconds"]:
        if result["downswing_duration_seconds"] > 0:
            result["tempo_ratio"] = round(
                result["backswing_duration_seconds"] / result["downswing_duration_seconds"], 2
            )

    if address is not None and finish is not None:
        result["total_swing_duration_seconds"] = round((finish - address) / fps, 2)

    return result
