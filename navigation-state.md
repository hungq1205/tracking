# Semantic Indoor Navigation Platform for Blind and Low-Vision Users (NVBlox Architecture)

## Vision

Build a persistent navigation platform that allows blind and low-vision users to move independently across public venues.

The system combines:

* Persistent semantic venue mapping
* Real-time localization
* Dynamic obstacle avoidance
* Multi-floor navigation
* Voice guidance
* GPU-accelerated volumetric mapping

Target environments:

* Malls
* Supermarkets
* Cinemas
* Hospitals
* Airports
* Universities
* Stations
* Public buildings

---

# System Architecture

```text
OFFLINE MODE (scan_server/)                 ONLINE MODE (server/ — NavigationAgent)
──────────────────────────────             ──────────────────────────────────────

Operator scans venue                       User: "Navigate to kitchen"

        RGB-D                                       RGB-D Stream
          ↓                                               ↓
       RGB-D SLAM                                  Real-Time Localization
          ↓                                               ↓
 Pose Graph Optimization                         Pose Tracking + Map Matching
          ↓                                               ↓
      NVBlox Mapping                           Local NVBlox Reconstruction
(TSDF + Semantic Occupancy)                            ↓
          ↓                                    ESDF Local Navigation Layer
 Incremental ESDF Build                                ↓
          ↓                                    Dynamic Obstacle Layer
 Semantic Labeling                                   ↓
          ↓                                   Planner + Guidance
 Export Map Bundle                                  ↓
(TSDF + ESDF + Labels)                    TTS + Arrival Detection


Static map served by MapService
```

---

# 1. Data Collection Layer

## Goal

Create reusable venue-scale spatial maps.

## Inputs

* RGB camera
* Depth (sensor or estimated)
* IMU
* GPS (optional outdoor)
* Operator annotations

## Collection Output

```text
RGB
Depth
Trajectory
Metadata
```

Output:

```text
Raw Sensor Logs
```

---

# 2. Mapping Layer (NVBlox)

## Goal

Build persistent volumetric maps for navigation.

## Pipeline

```text
RGB-D
 ↓
SLAM
 ↓
Pose Graph
 ↓
NVBlox TSDF Integration
 ↓
Semantic Occupancy Fusion
 ↓
Incremental ESDF
 ↓
Venue Map Export
```

---

## Internal Map Layers

### TSDF Layer (Geometry)

Stores:

```text
distance_to_surface
surface_weight
```

Purpose:

* 3D reconstruction
* mesh generation
* dense geometry

---

### Semantic Occupancy Layer

Stores:

```text
occupancy
semantic_label
confidence
traversable
```

Examples:

```text
occupied
walkable
checkout
elevator
stairs
door
aisle
```

Purpose:

* destination understanding
* scene semantics
* accessibility reasoning

---

### ESDF Layer (Navigation)

Stores:

```text
distance_to_obstacle
```

Purpose:

* safe route generation
* obstacle clearance
* local replanning

---

## Multi-Floor Representation

```text
Building
├── Floor 1
│   ├── TSDF
│   ├── ESDF
│   └── Labels
├── Floor 2
└── Floor N
```

Connections:

```text
stairs
elevator
escalator
```

Output:

```text
Persistent Venue Map Bundle
```

---

# 3. Semantic Labeling Layer

## Goal

Convert geometry into destinations.

Semantic hierarchy:

```text
Building
 └── Floor
      └── Zone
            └── Label
```

Example:

```text
ground_floor
 ├── entrance
 ├── reception
 └── elevator_A

first_floor
 ├── kitchen
 └── meeting_room
```

Stored object:

```json
{
  "name": "kitchen",
  "zone": "first_floor",
  "center": [8.1,3.0,2.2],
  "radius": 1.2
}
```

Output:

```text
Semantic Navigation Graph
```

---

# 4. Global Navigation Database

## Goal

Store reusable navigation maps.

Structure:

```text
Venue
 ↓
Floors
 ↓
TSDF
 ↓
ESDF
 ↓
Semantic Labels
 ↓
Route Graph
```

Each edge stores:

* distance
* accessibility
* elevator availability
* traversal estimate

Output:

```text
Navigation Database
```

---

# 5. User Localization Layer

## Goal

Determine user pose inside stored maps.

Pipeline:

```text
RGB-D
 ↓
Visual SLAM
 ↓
Pose Estimation
 ↓
TSDF Map Matching
 ↓
Localization
```

Output:

```text
Current Floor
Current Position
Current Heading
```

---

# 6. Dynamic Environment Layer

## Goal

React to temporary obstacles.

Runtime:

```text
RGB-D
 ↓
Local NVBlox Update
 ↓
Local ESDF Recompute
 ↓
Obstacle Detection
 ↓
Planner Update
```

Examples:

```text
crowd
cleaning cart
temporary barrier
boxes
```

Rules:

* Never overwrite static venue map
* Maintain temporary local layer only

Output:

```text
Temporary Traversability Map
```

---

# 7. Route Planning Layer

## Global Planning

```text
Venue Graph
 ↓
Destination Route
```

## Local Planning

```text
ESDF
 ↓
Safe Corridor
 ↓
Obstacle Avoidance
```

Output:

```text
Navigation Actions
```

Example:

```text
Walk 10 meters
Turn right
Elevator ahead
Destination on left
```

---

# 8. Guidance Layer

Voice examples:

```text
Navigating to kitchen.

Obstacle ahead.

Move slightly left.

Destination reached.
```

Interfaces:

* Earbuds
* Bone conduction
* Haptic feedback

---

# 9. Continuous Learning Layer

Collect:

* successful routes
* localization failures
* semantic corrections
* new venue labels

Update:

```text
TSDF
ESDF
Semantic Graph
```

without rebuilding entire venues.

---

# Implementation Mapping

| Layer              | Runtime            |
| ------------------ | ------------------ |
| RGB-D Mapping      | NVBlox             |
| Geometry           | Sparse TSDF        |
| Navigation         | ESDF               |
| Semantics          | Semantic Occupancy |
| Localization       | SLAM               |
| Obstacle Avoidance | Local ESDF         |
| Guidance           | NavigationAgent    |

---

# Core Principle

Build once offline.

Localize continuously.

Update only local reality.

Keep the persistent venue map immutable.
