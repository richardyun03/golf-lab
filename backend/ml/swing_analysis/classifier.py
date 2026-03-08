import numpy as np
from app.core.config import Settings
from app.schemas.analysis import SwingPhase, PoseKeypoint


def _smooth(signal: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(signal)]


def _velocity(signal: np.ndarray) -> np.ndarray:
    """Frame-to-frame velocity (first derivative). Returns array of same length (first element = 0)."""
    return np.diff(signal, prepend=signal[0])


class SwingPhaseClassifier:
    """
    Segments a swing into its canonical phases by frame index.

    Strategy (anchor-based):
    1. Use body velocity to find the SWING WINDOW — the main burst of activity
    2. Find IMPACT as the peak body velocity within that window
    3. Find TOP as the minimum wrist y (hands highest) before impact
    4. Derive address, takeaway, backswing from the pre-top region
    5. Derive downswing, follow-through, finish from the post-top region
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def classify(self, keypoints_by_frame: list[list[PoseKeypoint]]) -> dict[SwingPhase, int]:
        if not keypoints_by_frame or not any(keypoints_by_frame):
            return {}

        wrist_y, body_vel = self._extract_signals(keypoints_by_frame)
        if wrist_y is None:
            return {}

        n = len(wrist_y)
        wrist_y_smooth = _smooth(wrist_y, window=7)
        wrist_y_light = _smooth(wrist_y, 3)
        body_vel_smooth = _smooth(body_vel, window=9)

        # ---- Step 1: Find TOP — wrist Y minimum followed by velocity burst ----
        # The top of backswing = hands at highest point (min wrist Y) that is
        # followed by a strong velocity increase (the downswing). This
        # distinguishes it from follow-through (hands also high but velocity
        # decreasing) and pre-swing motion.
        top_frame = self._find_top(wrist_y_smooth, body_vel_smooth, n)

        # ---- Step 2: Find IMPACT — hands at lowest point after top ----
        # Search a limited window after top (max 30 frames = 1s at 30fps)
        # to avoid latching onto standing-around frames far after the swing.
        impact_search_end = min(n, top_frame + 30)
        post_top_wrist = wrist_y_light[top_frame:impact_search_end]
        if len(post_top_wrist) > 0:
            impact_frame = top_frame + int(np.argmax(post_top_wrist))
        else:
            impact_frame = top_frame + 1

        # ---- Step 3: Find ADDRESS — last quiet period before the backswing ----
        pre_top_vel = body_vel_smooth[:top_frame]
        if len(pre_top_vel) > 5:
            quiet_threshold = np.percentile(body_vel_smooth, 30)
            quiet_frames = np.where(pre_top_vel <= quiet_threshold)[0]
            if len(quiet_frames) > 0:
                gaps = np.diff(quiet_frames)
                big_gaps = np.where(gaps > 3)[0]
                if len(big_gaps) > 0:
                    last_stretch_start = quiet_frames[big_gaps[-1] + 1]
                    address_frame = int(last_stretch_start)
                else:
                    address_frame = int(quiet_frames[-1])
            else:
                address_frame = max(0, top_frame - 20)
        else:
            address_frame = 0

        # ---- Step 4: Find TAKEAWAY — wrist y starts deviating from address ----
        address_wrist_y = wrist_y_smooth[address_frame]
        deviation_threshold = 0.01
        takeaway_frame = address_frame
        for i in range(address_frame, top_frame):
            if abs(wrist_y_smooth[i] - address_wrist_y) > deviation_threshold:
                takeaway_frame = i
                break

        # ---- Step 5: BACKSWING — wrists clearly moving up ----
        backswing_frame = takeaway_frame + max(1, (top_frame - takeaway_frame) // 3)

        # ---- Step 6: DOWNSWING — body velocity ramps up after top ----
        top_vel = body_vel_smooth[top_frame]
        impact_vel_val = body_vel_smooth[impact_frame]
        ds_threshold = top_vel + (impact_vel_val - top_vel) * 0.5
        downswing_frame = top_frame + 1
        for i in range(top_frame + 1, impact_frame):
            if body_vel_smooth[i] > ds_threshold:
                downswing_frame = i
                break

        # ---- Step 7: FOLLOW-THROUGH — body decelerating after impact ----
        post_impact_vel = body_vel_smooth[impact_frame:]
        impact_vel = body_vel_smooth[impact_frame]
        if len(post_impact_vel) > 1 and impact_vel > 0:
            half_vel_frames = np.where(post_impact_vel < impact_vel * 0.5)[0]
            if len(half_vel_frames) > 0:
                follow_through_frame = impact_frame + int(half_vel_frames[0])
            else:
                follow_through_frame = impact_frame + max(1, (n - impact_frame) // 4)
        else:
            follow_through_frame = impact_frame + 1

        # ---- Step 8: FINISH — body velocity returns to quiet level ----
        if follow_through_frame < n:
            post_ft_vel = body_vel_smooth[follow_through_frame:]
            quiet_threshold = np.percentile(body_vel_smooth, 30)
            quiet = np.where(post_ft_vel <= quiet_threshold)[0]
            if len(quiet) > 0:
                finish_frame = follow_through_frame + int(quiet[0])
            else:
                finish_frame = follow_through_frame + max(1, (n - follow_through_frame) // 3)
        else:
            finish_frame = n - 1

        phases = {
            SwingPhase.ADDRESS: address_frame,
            SwingPhase.TAKEAWAY: takeaway_frame,
            SwingPhase.BACKSWING: backswing_frame,
            SwingPhase.TOP: top_frame,
            SwingPhase.DOWNSWING: downswing_frame,
            SwingPhase.IMPACT: impact_frame,
            SwingPhase.FOLLOW_THROUGH: follow_through_frame,
            SwingPhase.FINISH: finish_frame,
        }

        return self._enforce_ordering(phases, n)

    def _find_top(self, wrist_y_smooth: np.ndarray, body_vel_smooth: np.ndarray, n: int) -> int:
        """
        Find the top of backswing: the wrist Y minimum that is followed
        by the strongest velocity burst (the downswing).

        This distinguishes the real top from:
        - Follow-through (hands also high, but velocity is *before* it)
        - Pre-swing waggles (may have velocity but no preceding wrist Y dip)
        """
        # Find candidate frames: local minima of wrist Y below the 40th percentile
        threshold = np.percentile(wrist_y_smooth, 40)
        candidates = []

        for i in range(3, n - 3):
            if (wrist_y_smooth[i] <= threshold
                    and wrist_y_smooth[i] <= wrist_y_smooth[i - 3]
                    and wrist_y_smooth[i] <= wrist_y_smooth[i + 3]):
                candidates.append(i)

        if not candidates:
            # Fallback: global minimum
            return int(np.argmin(wrist_y_smooth))

        # Deduplicate: keep only the deepest point in each cluster
        merged = []
        cluster = [candidates[0]]
        for i in range(1, len(candidates)):
            if candidates[i] - candidates[i - 1] <= 5:
                cluster.append(candidates[i])
            else:
                best = min(cluster, key=lambda f: wrist_y_smooth[f])
                merged.append(best)
                cluster = [candidates[i]]
        merged.append(min(cluster, key=lambda f: wrist_y_smooth[f]))

        # Score each candidate: max body velocity in the 5-20 frames after it
        best_frame = merged[0]
        best_score = -1.0
        for frame in merged:
            search_start = min(n, frame + 3)
            search_end = min(n, frame + 25)
            if search_start >= search_end:
                continue
            post_vel = np.max(body_vel_smooth[search_start:search_end])
            if post_vel > best_score:
                best_score = post_vel
                best_frame = frame

        return best_frame

    def _find_swing_window(self, body_vel_smooth: np.ndarray) -> tuple[int, int]:
        """
        Isolate the main swing activity from pre/post standing around.

        Finds the dominant velocity peak, then expands outward to where
        velocity drops below the quiet threshold.
        """
        n = len(body_vel_smooth)
        quiet_threshold = np.percentile(body_vel_smooth, 40)

        peak = int(np.argmax(body_vel_smooth))

        start = 0
        for i in range(peak, -1, -1):
            if body_vel_smooth[i] <= quiet_threshold:
                start = i
                break

        end = n
        for i in range(peak, n):
            if body_vel_smooth[i] <= quiet_threshold:
                end = i
                break

        start = max(0, start - 5)
        end = min(n, end + 5)

        return start, end

    def _extract_signals(self, keypoints_by_frame: list[list[PoseKeypoint]]):
        """
        Extract wrist y position and total body velocity signals.
        Returns (wrist_y, body_velocity) arrays, or (None, None) if data is insufficient.
        """
        n = len(keypoints_by_frame)
        wrist_y = np.full(n, np.nan)
        all_positions = []

        for i, keypoints in enumerate(keypoints_by_frame):
            if not keypoints:
                all_positions.append(None)
                continue

            kp_dict = {kp.name: kp for kp in keypoints}
            lw = kp_dict.get("left_wrist")
            rw = kp_dict.get("right_wrist")

            if lw and rw:
                wrist_y[i] = (lw.y + rw.y) / 2.0
            elif lw:
                wrist_y[i] = lw.y
            elif rw:
                wrist_y[i] = rw.y

            positions = np.array([[kp.x, kp.y] for kp in keypoints])
            all_positions.append(positions)

        valid = ~np.isnan(wrist_y)
        if valid.sum() < 10:
            return None, None

        wrist_y = np.interp(np.arange(n), np.where(valid)[0], wrist_y[valid])

        body_vel = np.zeros(n)
        for i in range(1, n):
            if all_positions[i] is not None and all_positions[i - 1] is not None:
                if all_positions[i].shape == all_positions[i - 1].shape:
                    diff = all_positions[i] - all_positions[i - 1]
                    body_vel[i] = np.sqrt((diff ** 2).sum(axis=1)).sum()

        return wrist_y, body_vel

    def _enforce_ordering(self, phases: dict[SwingPhase, int], n: int) -> dict[SwingPhase, int]:
        """Ensure each phase frame is strictly after the previous one."""
        ordered = [
            SwingPhase.ADDRESS, SwingPhase.TAKEAWAY, SwingPhase.BACKSWING,
            SwingPhase.TOP, SwingPhase.DOWNSWING, SwingPhase.IMPACT,
            SwingPhase.FOLLOW_THROUGH, SwingPhase.FINISH,
        ]
        result = {}
        prev = -1
        for phase in ordered:
            frame = max(phases[phase], prev + 1)
            frame = min(frame, n - 1)
            result[phase] = frame
            prev = frame
        return result
