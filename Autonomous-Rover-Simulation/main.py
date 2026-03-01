from robot import Robot
from sensors.lidar import LidarScan
from control.navigator import Navigator
from utils.config import (
    AVOID_SPEED,
    AVOID_OMEGA,
    RESUME_CLEAR_FRAMES,
    MANUAL_OBSTACLE_HORIZON_VEL,
    MANUAL_FWD_SPEED,
    MANUAL_REV_SPEED,
    MANUAL_TURN_RATE,
)
import matplotlib.pyplot as plt
import math
import csv

# -----------------------------------------------------
# Runtime modes & visualization toggles
# -----------------------------------------------------
MODE = "MANUAL"        # MANUAL | AUTO
SHOW_LIDAR = True
SHOW_ODOM = True

# -----------------------------------------------------
# Path following globals
# -----------------------------------------------------
path_points = []
with open("path.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = float(row["x"])
        y = float(row["y"])
        path_points.append((x, y))

current_idx = 0          # current waypoint index along path
LOOKAHEAD_DIST = 2.5     # [m] pure-pursuit lookahead distance (increased to prevent spinning)
BASE_SPEED = 10.0        # [m/s] nominal forward speed
GOAL_TOL = 0.2           # [m] distance to final goal to stop

# Obstacle avoidance state: resume path only after path is clear for N frames
obstacle_clear_frames = 0

# -----------------------------------------------------
# Control commands (shared state)
# -----------------------------------------------------
v = 0.0   # linear velocity [m/s]
w = 0.0   # angular velocity [rad/s]


def on_key(event):
    """Keyboard control & visualization toggles."""
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM

    # --- visualization toggles ---
    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        print(f"Odometry visualization: {'ON' if SHOW_ODOM else 'OFF'}")
        return

    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        print(f"LiDAR visualization: {'ON' if SHOW_LIDAR else 'OFF'}")
        return

    # --- mode switching ---
    if event.key == 'm':
        MODE = "MANUAL"
        v = 0.0
        w = 0.0
        print("Switched to MANUAL mode")
        return

    if event.key == 'a':
        MODE = "AUTO"
        print("Switched to AUTO mode")
        return

    # --- manual control: one press = direct speed (no accumulation, responsive) ---
    if MODE != "MANUAL":
        return

    if event.key == 'up':
        v = MANUAL_FWD_SPEED
    elif event.key == 'down':
        v = -MANUAL_REV_SPEED
    elif event.key == 'left':
        w = MANUAL_TURN_RATE
    elif event.key == 'right':
        w = -MANUAL_TURN_RATE
    elif event.key == ' ':
        v = 0.0
        w = 0.0

    # clamp commands
    v = max(min(v, 6.0), -6.0)
    w = max(min(w, 3.0), -3.0)


if __name__ == "__main__":
    lidar = LidarScan(max_range=4.0)
    robot = Robot()
    navigator = Navigator()

    plt.close('all')
    fig = plt.figure(num=2)
    fig.canvas.manager.set_window_title("Autonomy Debug View")
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)

    dt = 0.01 

    # -------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------
    while plt.fignum_exists(fig.number):
        # ground truth pose
        real_x, real_y, real_theta = robot.get_ground_truth()
        # odometry estimate
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()
        # LiDAR scan 
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan((real_x, real_y, real_theta))

        if MODE == "AUTO":
            # Use odometry pose for control
            ideal_x, ideal_y, ideal_theta = robot.get_odometry()

            # --- goal check ---
            goal_x, goal_y = path_points[-1]
            dxg = goal_x - ideal_x
            dyg = goal_y - ideal_y
            dist_to_goal = math.hypot(dxg, dyg)

            if dist_to_goal < GOAL_TOL:
                v = 0.0
                w = 0.0
            else:
                # --- obstacle detection (forward cone, stopping distance, persistence) ---
                obstacles = robot.detector.detect(lidar_points, velocity=v)

                if obstacles:
                    # Deviate: choose left or right from cone hits, or stop if blocked
                    obstacle_clear_frames = 0
                    decision = navigator.decide(obstacles)
                    if decision == "STOP":
                        v = 0.0
                        w = 0.0
                    elif decision == "LEFT":
                        v = AVOID_SPEED
                        w = AVOID_OMEGA
                    elif decision == "RIGHT":
                        v = AVOID_SPEED
                        w = -AVOID_OMEGA
                    else:
                        v = AVOID_SPEED
                        w = 0.0
                else:
                    # No obstacle in cone
                    obstacle_clear_frames += 1
                    if obstacle_clear_frames >= RESUME_CLEAR_FRAMES:
                        # Path clear: resume pure-pursuit
                        N = len(path_points)
                        lookahead_x, lookahead_y = path_points[-1]

                        for i in range(current_idx, N):
                            px, py = path_points[i]
                            dx = px - ideal_x
                            dy = py - ideal_y
                            dist = math.hypot(dx, dy)
                            if dist >= LOOKAHEAD_DIST:
                                lookahead_x, lookahead_y = px, py
                                current_idx = i
                                break

                        dx = lookahead_x - ideal_x
                        dy = lookahead_y - ideal_y
                        cos_th = math.cos(ideal_theta)
                        sin_th = math.sin(ideal_theta)
                        x_r = cos_th * dx + sin_th * dy
                        y_r = -sin_th * dx + cos_th * dy
                        Ld = math.hypot(x_r, y_r)

                        if Ld < 1e-6:
                            v = 0.0
                            w = 0.0
                        else:
                            kappa = 2.0 * y_r / (Ld * Ld)
                            speed_goal = min(1.0, dist_to_goal / 2.0)
                            speed_curv = 2.0 / (1.0 + 2.0 * abs(kappa))
                            v = BASE_SPEED * speed_goal * speed_curv
                            w = kappa * v
                            v = max(min(v, 10.0), -10.0)
                            w = max(min(w, 3.0), -3.0)
                    else:
                        # Cleared recently: short straight creep before full path resume
                        v = AVOID_SPEED
                        w = 0.0
        elif MODE == "MANUAL":
            # Block forward when obstacle ahead; allow turning (w) and reversing (v < 0).
            # Always use a fixed horizon velocity so the stopping zone is constant regardless
            # of current speed — this keeps the robot blocked right up to the wall.
            horizon_vel = MANUAL_OBSTACLE_HORIZON_VEL if v >= 0 else 0.0
            obstacles = robot.detector.detect(lidar_points, velocity=horizon_vel)
            if obstacles and v > 0:
                v = 0.0  # stop forward; turning (w) and reversing (v < 0) still allowed
        robot.step(
            lidar_points,
            lidar_rays,
            lidar_hits,
            v,
            w,
            dt,
            show_lidar=SHOW_LIDAR,
            show_odom=SHOW_ODOM
        )

        plt.pause(dt)