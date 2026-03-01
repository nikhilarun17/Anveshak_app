ROBOT_RADIUS = 0.5     # meters (50 cm)

# -----------------------------------------------------
# Obstacle detection & stopping distance
# -----------------------------------------------------
# Velocity-based stopping distance: d_stop = v * t_response + v²/(2*a_max) + d_min
MAX_DECELERATION = 40.0       # m/s² maximum deceleration
RESPONSE_TIME_S = 0.01       # s delay before deceleration is applied
MIN_STOPPING_DISTANCE = 0.1  # m minimum threshold (avoids late braking at high speed)

# Forward cone: only these lidar ray indices (angle_step=10° → 0,1,2,3,4 = 0°..40°, 33,34,35 = 330°,340°,350° ≈ ±35°)
FORWARD_CONE_RAY_INDICES = (0, 1, 2, 3, 4, 33, 34, 35)

# Noise rejection: require obstacle to be seen for this many consecutive frames
PERSISTENCE_FRAMES = 2
# Minimum number of cone rays that must see an obstacle within stopping distance (percentile-style filter)
MIN_CONE_RAYS_HIT = 2

# LiDAR max range (used to ignore "no hit" rays in cone)
LIDAR_MAX_RANGE = 4.0
# Treat ray as "no hit" if range >= this (avoid counting max-range as obstacle)
LIDAR_NO_HIT_THRESHOLD = 0.98  # fraction of max_range

# -----------------------------------------------------
# Obstacle avoidance (go-around behavior)
# -----------------------------------------------------
AVOID_SPEED = 3.0          # [m/s] forward speed while avoiding
AVOID_OMEGA = 1.5          # [rad/s] turn rate when deviating (left/right)
RESUME_CLEAR_FRAMES = 3    # frames with no obstacle before resuming path tracking

# -----------------------------------------------------
# Manual mode
# -----------------------------------------------------
# Min velocity used for obstacle horizon when in manual (so we keep blocking until path is clear)
MANUAL_OBSTACLE_HORIZON_VEL = 8.0  # [m/s] → ~0.9 m stopping distance (bigger wall buffer)
MANUAL_FWD_SPEED = 3.0             # [m/s] set by one "up" press
MANUAL_REV_SPEED = 3.0             # [m/s] set by one "down" press
MANUAL_TURN_RATE = 1.5             # [rad/s] set by one "left"/"right" press