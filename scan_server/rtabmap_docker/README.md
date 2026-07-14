# RTAB-Map pose service

Runs RTAB-Map (RGB-D odometry + loop closure) in a container, exposing a
plain ZeroMQ REQ/REP socket so `scan_server/rtabmap_client.py` can feed it
synchronized RGB frames + depth maps and get back live 6-DoF poses. No ROS
anywhere — RTAB-Map's corelib (`rtabmap::Odometry`, `rtabmap::Rtabmap`) has no
ROS dependency; `rtabmap_ros`/`rtabmap_ros2` are separate wrapper packages on
top of it. `src/rtabmap_server.cc` talks a custom binary protocol over ZeroMQ
(see that file's header comment for the exact wire format) — same shape as
the ORB-SLAM3 service this replaces, minus every IMU field.

This replaces `scan_server/orbslam3_docker/` (removed): ORB-SLAM3's
mono-inertial mode depended on a Kalibr camera-IMU calibration that proved
unreliable in practice. RTAB-Map's RGB-D odometry needs **only camera
intrinsics** — no camera-IMU extrinsic, no IMU noise model, **no calibration
step at all**. Depth for the "D" in RGB-D comes from this project's own
DA3-ONNX estimator (computed in Python, sent per-frame over the wire), not a
real depth sensor.

Built on:
- [`introlab3it/rtabmap:noble`](https://hub.docker.com/r/introlab3it/rtabmap) —
  official prebuilt image (Ubuntu 24.04) with RTAB-Map's corelib already
  compiled, confirmed (via `docker run --rm introlab3it/rtabmap:noble find / -iname 'Rtabmap.h'`)
  to ship dev headers at `/usr/local/include/rtabmap-0.23/` and a CMake
  config at `/usr/local/lib/rtabmap-0.23/RTABMapConfig.cmake` — so this
  Dockerfile only adds ZeroMQ and compiles our small server against the
  existing `librtabmap_core.so`, same "don't rebuild the whole library"
  approach the ORB-SLAM3 service used.
- `src/rtabmap_server.cc` — our own executable, `find_package(RTABMap)`,
  using `zmq.hpp` (`cppzmq-dev` apt package on this base image, header-only)
  for the socket.

## Prerequisites (host machine)

- `docker` only — **no NVIDIA GPU/Container Toolkit required** (unlike the
  ORB-SLAM3 service, which needed GPU/OpenGL passthrough purely because its
  base image's Pangolin viewer expected it even when run headless). RTAB-Map's
  corelib has no GUI/OpenGL dependency at all.

## Fallback: building corelib from source

If a future base-image tag doesn't ship dev headers/cmake config (verify
with the `find` command above before building), build RTAB-Map's corelib
only from source instead of relying on the prebuilt image:

```bash
git clone --branch <pinned-tag> https://github.com/introlab/rtabmap.git
cd rtabmap/build
cmake .. -DBUILD_APP=OFF -DBUILD_TOOLS=OFF   # corelib only, no Qt/GUI needed
make -j$(nproc) && make install
```

then adapt this directory's Dockerfile to a multi-stage build: compile in a
build stage, copy the installed libs/headers into a slim runtime stage. This
is a normal, documented open-source CMake build (unlike ORB-SLAM3's
research-fork base image), so it's lower-risk than the "build from scratch"
path already avoided for ORB-SLAM3.

## Build

```bash
docker build -t tracking-rtabmap -f scan_server/rtabmap_docker/Dockerfile scan_server/rtabmap_docker
```

## Run

Via `docker-compose` (see the `rtabmap` service in the repo root
`docker-compose.yml`) — publishes port 5556. **No calibration file to
mount** — the entire reason for this migration.

Or standalone:

```bash
docker run --rm -p 5556:5556 tracking-rtabmap
```

Then point `scan_server.py` at it: `RTABMAP_ADDR=tcp://localhost:5556`.

## Interface

Single ZeroMQ REP socket, one request per call (`rtabmap_client.py` uses a
REQ socket — strict request/reply, matching this synchronous per-frame style):

- `TRACK` — RGB frame + depth map + per-frame camera intrinsics in → pose out
  (camera-to-world, loop-closure-corrected via `Rtabmap::getMapCorrection()`),
  or a "lost" status if odometry fails to converge on this frame. The reply
  also carries a `loop_closure` flag (1 if THIS frame's processing closed a
  loop) — `rtabmap_client.TrackedFrame.loop_closure` — telling the caller that
  previously-pulled `GET_CLOUD` nodes' poses may have just shifted.
- `GET_CLOUD` — reconstructs the surface RTAB-Map's own way: for every node
  with id greater than a given cursor, its stored SensorData is re-projected
  (`util3d::cloudRGBFromSensorData`), voxelized, and transformed by RTAB-Map's
  **current** graph-corrected pose (`getLocalOptimizedPoses`) — not
  `scan_session.py`'s own DA3-depth back-projection. Since poses used are
  always the latest, a `since_node_id=0` pull after a loop closure gives every
  already-reconstructed node's cloud its corrected position — see
  `scan_session.py`'s `_rtabmap_full_resync()`.
- `RESET` — fully reinitializes **both** the `Odometry` and `Rtabmap` objects
  (not just one), so a fresh scan never loop-closes against a previous,
  unrelated scan's map in the same container's memory
- `PING` — liveness check, used once at client construction

See `src/rtabmap_server.cc`'s header comment for the exact byte layout.
No `n_imu`/IMU sample fields exist anywhere in this protocol.

## Known limitations (v1)

- Single active SLAM session per container — matches scan_server's Gradio UI,
  which scans one location interactively at a time. Concurrent scans aren't
  supported.
- ~~Loop closure corrects RTAB-Map's own internal pose graph retroactively,
  but back-projected points are never re-projected after a later loop closure
  shifts earlier poses~~ — **fixed for RTAB-Map pose mode** via `GET_CLOUD` +
  the `loop_closure` flag (see above): `scan_session.py` discards and
  re-pulls every node's cloud, at its current corrected pose, whenever a loop
  closure is reported. This limitation still applies to the other 3 pose
  sources (VO/IMU+VO/DA3 poses), whose clouds come from `scan_session.py`'s
  own DA3-depth back-projection baked in at each batch's pose at the time —
  `pose_graph.py`'s VO path has the identical limitation it always did.
- `Odom/Strategy`/`Reg/Strategy` are set to reasonable defaults
  (Frame-to-Map odometry, feature-based registration) but not tuned against
  real DA3-ONNX depth noise characteristics yet — expect to revisit
  `Vis/MinInliers`, `Odom/ResetCountdown`, etc. once real recordings are
  available for comparison against the old ORB-SLAM3 runs.
- **Found via real recordings**: `RGBD/ProximityBySpace` (on by default)
  accepts a loop closure against ANY spatially-nearby node, which fires on
  almost every frame during a slow, close-range, single-room scan — the
  camera trajectory naturally stays within a couple meters of itself the
  whole time. Two mitigations are in place: `RGBD/ProximityAngle` is
  tightened from RTAB-Map's default 45° to 20° (`make_parameters()`) so a
  proximity match requires a more similar viewing angle before being
  accepted at all; and even when RTAB-Map does accept one, the server only
  sets the TRACK reply's `loop_closure` flag (see wire protocol above) if
  the resulting map correction moved by more than
  `SIGNIFICANT_CORRECTION_TRANSLATION_M`/`_ANGLE_RAD` (2 cm / ~1°) since the
  last one that did — so the client's expensive full `GET_CLOUD` resync
  only fires for corrections large enough to actually matter, not every
  trivial one RTAB-Map's own graph-consistency bookkeeping accepts
  internally. Verified against a real 79-frame bedroom recording: dropped
  from a full resync on nearly every frame to a single resync for the whole
  recording. Both thresholds/the angle value may still need further
  real-world tuning for larger/multi-room scans or faster motion.
