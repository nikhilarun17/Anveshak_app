# Autonomous Rover Task

This repository contains a lightweight 2D rover simulator with LiDAR sensing, odometry, and visualization.
The task is designed to evaluate **learning ability, debugging skills, and basic autonomy reasoning**, not prior ROS or simulation experience.

---

## Prerequisites

- Python 3.8+
- Works on Windows, Linux, and macOS

---

## Setup
All required Python libraries are listed in `requirements.txt`.

### Install Dependencies

####  Windows (PowerShell)
```powershell
pip3 install numpy
pip3 install matplotlib
```

####  Linux/macOS (bash)
```bash
pip3 install numpy
pip3 install matplotlib
```

## Run the simulator

From the project root:

####  Windows (PowerShell)
```powershell
python3 main.py
```

####  Linux/macOS (bash)
```bash
python3 main.py
```

-> A single window titled “Autonomy Debug View” will open.

## Controls

### Mode Control

| Key | Action |
|----|--------|
| **M** | Manual control |
| **A** | Autonomous (scripted) control |

---

### Manual Driving

| Key | Action |
|----|--------|
| **↑** | Move forward |
| **↓** | Move backward |
| **←** | Rotate left |
| **→** | Rotate right |
| **Space** | Stop |

---

### Visualization Toggles

| Key | Action |
|----|--------|
| **L** | Toggle LiDAR visualization |
| **O** | Toggle odometry visualization |

---

## LiDAR Data

The LiDAR simulates a **2D planar scanner** with the following characteristics:

- **360° field of view**
- **1 ray per 10 degrees**
- **36 beams total**
- **Maximum range:** `4.0 m`
- Variable Name `lidar_ranges`

This format is similar to real LiDAR drivers.

## Code Modification Rules
Autonomous control logic must be written only inside main.py, in the marked block.
Only v and w may be assigned.

```python
# write your autonomous code here!!!!!!!!!!!!!
```
---
Allowed Operations
- Inside this block, you may Read sensor data
```python
real_x, real_y, real_theta # ground truth pose
ideal_x, ideal_y, ideal_theta # odometry estimate
lidar_ranges # lidar ranges
```
---
Required Outputs
You must only assign values to:
```python
v  # linear velocity
w  # angular velocity
```
---

## Odometry Debugging Exception
You are allowed to modify files outside `main.py` **only for the purpose of fixing odometry bugs**.

---

Notes
- The simulator shows both ground truth and odometry estimate 
- LiDAR data represents real obstacle geometry
- Visualization can be toggled to reduce clutter
- There is no single “correct” solution.Clear logic, clean reasoning, and stable behavior matter more than performance.
- Do not modify any other files, For debugging the odometry you have to go through full code base to find the bug
- What ever changes you make to code base do properly document it with the explanation for the approach and the algorithm
- Video reference for path following -> [video](https://drive.google.com/file/d/1OyGZUBgm1-nCKmARCFtj8WqxE0Yr8Ymf/view?usp=drive_link)
