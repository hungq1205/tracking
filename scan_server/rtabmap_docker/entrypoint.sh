#!/bin/bash
set -e

BIND_ADDR="${RTABMAP_BIND_ADDR:-tcp://*:5556}"

# Unlike orbslam3_docker's entrypoint.sh, there is no required calibration
# file to check for here — RTAB-Map RGB-D odometry takes per-frame camera
# intrinsics over the wire (see src/rtabmap_server.cc's wire protocol), not a
# mounted settings YAML. This is the concrete "no per-device calibration"
# payoff of replacing ORB-SLAM3 with RTAB-Map.

echo "[rtabmap] bind=$BIND_ADDR"
exec /rtabmap_server_src/build/rtabmap_server "$BIND_ADDR"
