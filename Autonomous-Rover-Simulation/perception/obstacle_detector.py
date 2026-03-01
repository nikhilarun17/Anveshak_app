"""
Obstacle detection using a forward cone of the LiDAR with velocity-based
stopping distance, persistence filter, and minimum cone-ray hit threshold.
"""
import math
from collections import deque

from utils.config import (
    MAX_DECELERATION,
    RESPONSE_TIME_S,
    MIN_STOPPING_DISTANCE,
    FORWARD_CONE_RAY_INDICES,
    PERSISTENCE_FRAMES,
    MIN_CONE_RAYS_HIT,
    LIDAR_MAX_RANGE,
    LIDAR_NO_HIT_THRESHOLD,
)


def compute_stopping_distance(v: float) -> float:
    """
    Stopping distance based on current speed, max deceleration, response time,
    and a minimum threshold so high-speed runs can still stop in time.

    d_stop = v * t_response + v² / (2 * a_max) + d_min
    """
    v = abs(v)
    reaction_dist = v * RESPONSE_TIME_S
    brake_dist = (v * v) / (2.0 * MAX_DECELERATION) if MAX_DECELERATION > 0 else 0.0
    return reaction_dist + brake_dist + MIN_STOPPING_DISTANCE


class ObstacleDetector:
    def __init__(self):
        self._persistence_buffer: deque[bool] = deque(maxlen=PERSISTENCE_FRAMES)
        self._no_hit_range = LIDAR_MAX_RANGE * LIDAR_NO_HIT_THRESHOLD

    def detect(self, lidar_points: list[tuple[float, float]], velocity: float = 0.0) -> list[tuple[float, float]]:
        """
        Detect obstacles in the forward cone within stopping distance.

        - Uses only cone rays (e.g. indices 0,1,2,3,4,33,34,35) for ~±35° forward.
        - Stopping distance is computed from current velocity (reaction + braking + min).
        - Percentile-style filter: at least MIN_CONE_RAYS_HIT must see an obstacle.
        - Persistence filter: obstacle must be present for PERSISTENCE_FRAMES consecutive frames.

        lidar_points: list of (x, y) in robot frame, same order as lidar rays (index = ray index).
        velocity: current linear velocity [m/s] for stopping distance.
        Returns: list of (x, y) in robot frame for cone points within stopping distance, or [] if not confirmed.
        """
        if len(lidar_points) <= max(FORWARD_CONE_RAY_INDICES, default=0):
            self._persistence_buffer.append(False)
            return []

        stopping_dist = compute_stopping_distance(velocity)
        hits_in_cone: list[tuple[float, float]] = []
        rays_with_hit = 0

        for idx in FORWARD_CONE_RAY_INDICES:
            if idx >= len(lidar_points):
                continue
            x, y = lidar_points[idx]
            r = math.hypot(x, y)
            # Only count as hit if in front, within stopping distance, and not a "no hit" (max range)
            if x > 0 and r <= stopping_dist and r < self._no_hit_range:
                rays_with_hit += 1
                hits_in_cone.append((x, y))

        obstacle_seen_this_frame = rays_with_hit >= MIN_CONE_RAYS_HIT
        self._persistence_buffer.append(obstacle_seen_this_frame)

        # Require obstacle to be seen for PERSISTENCE_FRAMES consecutive frames
        if len(self._persistence_buffer) >= PERSISTENCE_FRAMES and all(self._persistence_buffer):
            return hits_in_cone
        return []
