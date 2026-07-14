# RTAB-Map RGB-D pose service — plain ZeroMQ, no ROS.
#
# rtabmap's corelib (Odometry + Rtabmap classes) has no ROS dependency at all
# — rtabmap_ros/rtabmap_ros2 are separate wrapper packages on top of it. This
# replaces ORB-SLAM3 (scan_server/orbslam3_docker/, removed) as the pose
# source: RTAB-Map's RGB-D visual odometry + loop closure needs only camera
# intrinsics, no camera-IMU extrinsic and no IMU noise model, sidestepping
# the unreliable Kalibr calibration ORB-SLAM3 depended on. Depth for the "D"
# in RGB-D comes from this project's own DA3-ONNX estimator (computed in
# Python, sent over the wire per frame) — not a real depth sensor.
#
# Base image: introlab3it/rtabmap:noble — official prebuilt image with
# RTAB-Map's corelib already compiled. If this image doesn't ship dev
# headers/cmake config (verify with:
#   docker run --rm introlab3it/rtabmap:noble find / -iname 'Rtabmap.h' -o -iname 'RTABMapConfig.cmake'
# ), see README.md's "Fallback: building corelib from source" section instead
# of this Dockerfile.

FROM introlab3it/rtabmap:noble

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        cmake \
        build-essential \
        libzmq3-dev \
        cppzmq-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# cppzmq-dev — header-only C++ ZeroMQ binding (libzmq3-dev above is only the
# C library), available as a plain apt package on this base image's Ubuntu
# Noble; simpler and more reliable than vendoring zmq.hpp via wget from
# GitHub at build time (the orbslam3_docker/Dockerfile this replaces did that).

# Fail fast with a clear message if the base image doesn't expose RTAB-Map's
# CMake config where src/CMakeLists.txt expects it.
RUN (find / -iname 'RTABMapConfig.cmake' 2>/dev/null | grep -q . || \
     find / -iname 'librtabmap_core*' 2>/dev/null | grep -q .) || \
    (echo "[build] FATAL: no RTABMapConfig.cmake or librtabmap_core.* found on this image." >&2 && \
     echo "[build] See README.md's fallback: build corelib from source instead." >&2 && \
     exit 1)

# ── rtabmap_server (our standalone ZeroMQ pose service) ───────────────────────

COPY src /rtabmap_server_src
RUN mkdir /rtabmap_server_src/build && cd /rtabmap_server_src/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5556

ENTRYPOINT ["/entrypoint.sh"]
