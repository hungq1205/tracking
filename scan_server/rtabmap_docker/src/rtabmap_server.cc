// Standalone RTAB-Map RGB-D pose + surface-reconstruction service.
//
// No ROS anywhere — rtabmap's corelib (rtabmap::Odometry + rtabmap::Rtabmap)
// has no ROS dependency; rtabmap_ros/rtabmap_ros2 are separate wrapper
// packages on top of it. Talks to scan_server/rtabmap_client.py over a
// single ZeroMQ REP socket, one request/reply per call — same "no ROS, no
// multipart messages" shape scan_server/orbslam3_docker/src/orbslam3_server.cc
// used for ORB-SLAM3 (which this replaces).
//
// Unlike ORB-SLAM3's mono-inertial protocol, there is NO IMU payload at all —
// RTAB-Map's RGB-D visual odometry only needs camera intrinsics + a depth
// map (here, DA3-ONNX estimated depth from the Python side, not a real
// sensor), sidestepping ORB-SLAM3's unreliable camera-IMU calibration
// entirely. Runs BOTH rtabmap::Odometry (incremental pose) AND rtabmap::Rtabmap
// (the actual loop-closure/graph-optimization object) — bare Odometry alone
// has no loop closure, which is the whole reason to prefer this over
// ORB-SLAM3's drift-only-corrected-by-local-BA behavior.
//
// GET_CLOUD reconstructs each node's own point cloud from ITS stored
// SensorData (util3d::cloudRGBFromSensorData) transformed by RTAB-Map's own
// CURRENT graph-corrected pose (getLocalOptimizedPoses) — i.e. the surface
// is rebuilt "the RTAB-Map way" (its own per-node cloud generation +
// loop-closure-corrected poses), not scan_server's own DA3-depth
// back-projection. Unlike that Python-side back-projection, re-fetching
// after a loop closure gives EVERY already-reconstructed node's cloud its
// latest corrected position — closing the "stale historical points" gap
// documented in this directory's README.
//
// Wire protocol (little-endian, no padding — matches Python's struct "<"):
//
//   Request (single ZMQ frame):
//     byte 0        : command — 0x01 TRACK, 0x02 RESET, 0x03 PING, 0x04 GET_CLOUD
//     TRACK only, immediately following:
//       int64  timestamp_ns
//       int32  width
//       int32  height
//       double fx, fy, cx, cy            per-frame intrinsics (DA3's
//                                          estimated K can vary slightly
//                                          frame-to-frame, so this is sent
//                                          per-request, not a mounted file)
//       uint8[width*height*3]             raw RGB8, row-major
//       float32[width*height]             depth map, metres, row-major,
//                                          0.0 = invalid/no depth
//     GET_CLOUD only, immediately following:
//       int32  since_node_id              only nodes with id > this are returned
//       float  voxel_size                 0 = no voxel downsample
//       float  max_depth                  0 = no limit (matches Python's
//                                          _back_project_frame convention;
//                                          min depth fixed at 0.1 m server-side)
//
//   Reply (single ZMQ frame):
//     byte 0 : status — 0x00 OK, 0x01 LOST, 0x02 RESET_OK, 0x03 PONG
//     TRACK OK only:
//       double[7]  tx,ty,tz,qx,qy,qz,qw  (camera-to-world, the
//                  loop-closure-corrected pose — see corrected_pose() below)
//       uint8      loop_closure_flag      1 if THIS call closed a loop AND
//                  the resulting map correction shifted by more than a small
//                  threshold since the last time this flag was set (see
//                  SIGNIFICANT_CORRECTION_*  below) — the client uses this to
//                  trigger a full GET_CLOUD(since=0) resync instead of its
//                  normal incremental pull. RTAB-Map's own RGBD/ProximityBySpace
//                  (on by default) accepts a loop closure against ANY
//                  spatially-nearby node, which fires almost every frame when
//                  slowly scanning a small room — internally harmless (that's
//                  how RTAB-Map keeps its own graph consistent), but a full
//                  client-side resync (reprocessing every historical node) on
//                  every such trivial correction is not: this flag only goes
//                  out when the correction is big enough to actually matter.
//       int32      new_node_id            id of the node THIS frame became in
//                  RTAB-Map's memory, or -1 if this frame did NOT become a
//                  new node (e.g. insufficient displacement since the last
//                  one — RTAB-Map doesn't turn every processed frame into a
//                  node). Lets the client correlate its own per-frame DA3
//                  depth-consistency check (scan_session.py/feature_tracker.py
//                  — same mechanism used for IMU+VO/VO mode) with the SPECIFIC
//                  reconstructed node a bad frame produced, so that node's
//                  geometry can be vetoed when later pulled via GET_CLOUD —
//                  RTAB-Map itself has no way to know a frame's depth was
//                  wrong (it just trusts what TRACK sends), so this has to be
//                  filtered client-side; without a node_id link, the client
//                  would have no way to know which reconstructed node came
//                  from which frame.
//     GET_CLOUD OK only:
//       int32 node_count
//       repeated node_count times, ascending node id:
//         int32   node_id
//         double[7] pose                  world frame, same convention as TRACK
//         int32   point_count
//         float32[point_count*3]  xyz
//         uint8[point_count*3]    rgb

#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Geometry>

#include <opencv2/imgproc/imgproc.hpp>
#include <zmq.hpp>

#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_filtering.h>
#include <rtabmap/core/util3d_transforms.h>

namespace {

constexpr uint8_t CMD_TRACK = 0x01;
constexpr uint8_t CMD_RESET = 0x02;
constexpr uint8_t CMD_PING = 0x03;
constexpr uint8_t CMD_GET_CLOUD = 0x04;
constexpr uint8_t STATUS_OK = 0x00;
constexpr uint8_t STATUS_LOST = 0x01;
constexpr uint8_t STATUS_RESET_OK = 0x02;
constexpr uint8_t STATUS_PONG = 0x03;

// Minimum depth floor for cloud reconstruction — matches scan_session.py's
// _back_project_frame's own hardcoded 0.1 m mask, kept consistent so a
// GET_CLOUD-reconstructed surface isn't systematically noisier/sparser near
// the camera than the Python back-projection it replaces for RTAB-Map mode.
constexpr float MIN_DEPTH_M = 0.1f;

// A loop closure whose resulting map correction moved by less than these
// thresholds since the last SIGNIFICANT one is not reported to the client
// (loop_closure_flag stays 0) — see the wire-protocol comment above for why.
// Deliberately small enough that a real drift correction is never silently
// dropped (2 cm / ~1 degree is well below what matters for a room-scale
// occupancy grid), just large enough to filter out the "loop closure against
// an almost-identical, spatially-adjacent node" case RGBD/ProximityBySpace
// produces constantly during slow, close-range scanning.
constexpr float SIGNIFICANT_CORRECTION_TRANSLATION_M = 0.02f;
constexpr float SIGNIFICANT_CORRECTION_ANGLE_RAD = 0.0175f;  // ~1 degree

#pragma pack(push, 1)
struct TrackRequestHeader {
    int64_t timestamp_ns;
    int32_t width;
    int32_t height;
    double fx, fy, cx, cy;
};
struct GetCloudRequestHeader {
    int32_t since_node_id;
    float voxel_size;
    float max_depth;
};
#pragma pack(pop)

void send_status(zmq::socket_t& sock, uint8_t status) {
    zmq::message_t reply(1);
    std::memcpy(reply.data(), &status, 1);
    sock.send(reply, zmq::send_flags::none);
}

// RTAB-Map's Rtabmap::process() optimizes the pose graph internally; the
// standard way to get the corrected *global* pose for the current frame is
// to compose the map->odom correction with the raw odometry pose (the same
// technique rtabmap_ros' CoreWrapper uses to publish map->base_link).
// Confirmed against RTAB-Map 0.23's Rtabmap.h: `Transform getMapCorrection() const`
// exists exactly as used below.
rtabmap::Transform corrected_pose(const rtabmap::Rtabmap& rtabmap,
                                   const rtabmap::Transform& odomPose) {
    return rtabmap.getMapCorrection() * odomPose;
}

// Confirmed against RTAB-Map 0.23's Transform.h: getTranslation(x,y,z)
// exists directly, but there is no 4-arg getQuaternion() overload — only
// getQuaternionf() returning an Eigen::Quaternionf.
void pose_to_wire(const rtabmap::Transform& pose, double out[7]) {
    float x, y, z;
    pose.getTranslation(x, y, z);
    const Eigen::Quaternionf q = pose.getQuaternionf();
    out[0] = x; out[1] = y; out[2] = z;
    out[3] = q.x(); out[4] = q.y(); out[5] = q.z(); out[6] = q.w();
}

// (params, id) live at file scope so RESET can fully reconstruct both
// objects from scratch — a stale Rtabmap "Memory" could otherwise loop-close
// against nodes from a previous, unrelated scan of this same container.
rtabmap::ParametersMap make_parameters() {
    rtabmap::ParametersMap params;
    // Frame-to-map RGB-D visual odometry (no IMU fusion parameters touched
    // anywhere in this map — RTAB-Map must never expect/require IMU input).
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kOdomStrategy(), "0"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRegStrategy(), "0"));
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRGBDEnabled(), "true"));
    // GET_CLOUD reconstructs each node's surface from its own stored
    // SensorData after the fact (possibly much later, after a loop closure) —
    // Mem/BinDataKept (default true) already keeps compressed image+depth
    // blobs in the session's in-memory DB, which is all cloudRGBFromSensorData
    // needs once uncompressed; set explicitly here so a future RTAB-Map
    // default change can't silently break surface reconstruction.
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kMemBinDataKept(), "true"));
    // RGBD/ProximityAngle (default 45 degrees) governs how different two
    // nodes' viewing angles may be before RGBD/ProximityBySpace (on by
    // default) still considers them a valid one-to-one proximity match.
    // Tightened here — this project's scanning pattern is slow, close-range,
    // and often within one small room, where the DEFAULT threshold accepts
    // a loop closure against nearly every recent, barely-moved node (see
    // rtabmap_client.py's loop_closure handling and this file's TRACK reply
    // comment for the client-side half of this fix — this narrows how often
    // RTAB-Map reports a closure at all; that gates whether the client
    // bothers reacting to one it did report). Not zero/disabled — proximity
    // detection is still useful for genuine "returned to this spot" drift
    // correction, just less trigger-happy about accepting a near-duplicate
    // viewpoint as one. Real-world tuning against more/larger recordings
    // (multi-room scans, faster motion) may still adjust this further — see
    // the README's "Known limitations" section.
    params.insert(rtabmap::ParametersPair(rtabmap::Parameters::kRGBDProximityAngle(), "20"));
    return params;
}

// Reconstructs one node's world-frame RGB point cloud the "RTAB-Map way":
// its own stored SensorData -> cloudRGBFromSensorData (RTAB-Map's standard
// per-node cloud generation, decimation/depth-range filtered) -> voxelize
// (RTAB-Map's own filter, same algorithm scan_session.py's Open3D voxel
// downsample approximates) -> transformed into world frame by the CURRENT
// graph-corrected pose (not whatever pose was in effect when this node was
// first added) so a later loop closure is reflected the next time this is
// called — the whole point of sourcing reconstruction from RTAB-Map itself.
pcl::PointCloud<pcl::PointXYZRGB>::Ptr reconstruct_node_cloud(
    const rtabmap::Memory* mem, int node_id, const rtabmap::Transform& pose,
    float voxel_size, float max_depth) {
    rtabmap::SensorData data = mem->getNodeData(node_id, true, false, false, false);
    data.uncompressData();
    if (data.imageRaw().empty() || data.depthOrRightRaw().empty()) {
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    }
    // cloudRGBFromSensorData returns an ORGANIZED cloud for a single camera
    // (invalid-depth pixels present as non-finite points, not removed) — its
    // validIndices output is REQUIRED downstream: util3d::voxelize() on an
    // organized cloud with the no-indices overload (which passes an empty
    // IndicesPtr internally) explicitly refuses to filter and returns an
    // empty cloud (see voxelizeImpl's "not dense (organized) cloud with
    // empty indices" guard in RTAB-Map's util3d_filtering.cpp) — it has no
    // way to know which points are valid without them.
    std::vector<int> validIndices;
    auto cloud = rtabmap::util3d::cloudRGBFromSensorData(
        data, /*decimation=*/1, max_depth, MIN_DEPTH_M, &validIndices);
    if (cloud->empty()) {
        return cloud;
    }
    if (voxel_size > 0.0f) {
        pcl::IndicesPtr indices(new std::vector<int>(validIndices));
        cloud = rtabmap::util3d::voxelize(cloud, indices, voxel_size);
    }
    return rtabmap::util3d::transformPointCloud(cloud, pose);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string bind_addr = argc > 1 ? argv[1] : "tcp://*:5556";

    rtabmap::ParametersMap params = make_parameters();
    rtabmap::Odometry* odom = rtabmap::Odometry::create(params);
    rtabmap::Rtabmap slam;
    slam.init(params, "");  // empty path — in-memory/throwaway session DB,
                             // one continuous SLAM session per container
                             // (mirrors ORB-SLAM3's "single active session").

    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, zmq::socket_type::rep);
    sock.bind(bind_addr);
    std::cout << "[rtabmap_server] listening on " << bind_addr << std::endl;

    int next_id = 1;
    // Last map correction (getMapCorrection()) actually REPORTED to the
    // client as significant — see SIGNIFICANT_CORRECTION_* above. Compared
    // against the CURRENT correction each TRACK call; only updated when a
    // correction crosses the threshold, so a series of individually-tiny
    // corrections still accumulates and eventually gets reported once their
    // sum crosses it (comparing against a constantly-refreshed baseline
    // would let that drift hide forever).
    rtabmap::Transform last_reported_correction = rtabmap::Transform::getIdentity();

    while (true) {
        zmq::message_t req;
        auto received = sock.recv(req, zmq::recv_flags::none);
        if (!received || req.size() < 1) {
            send_status(sock, STATUS_LOST);
            continue;
        }

        const uint8_t* buf = static_cast<const uint8_t*>(req.data());
        const uint8_t cmd = buf[0];

        if (cmd == CMD_RESET) {
            // Full reinit of BOTH objects, not just a state clear — see the
            // header comment on why a stale Rtabmap memory is a correctness
            // hazard across scans, not just cosmetic drift.
            delete odom;
            odom = rtabmap::Odometry::create(params);
            slam.close();
            slam.init(params, "");
            next_id = 1;
            last_reported_correction = rtabmap::Transform::getIdentity();
            send_status(sock, STATUS_RESET_OK);
            continue;
        }

        if (cmd == CMD_PING) {
            send_status(sock, STATUS_PONG);
            continue;
        }

        if (cmd == CMD_GET_CLOUD) {
            if (req.size() != 1 + sizeof(GetCloudRequestHeader)) {
                send_status(sock, STATUS_LOST);
                continue;
            }
            GetCloudRequestHeader ghdr;
            std::memcpy(&ghdr, buf + 1, sizeof(GetCloudRequestHeader));

            const std::map<int, rtabmap::Transform> poses = slam.getLocalOptimizedPoses();
            const rtabmap::Memory* mem = slam.getMemory();

            std::vector<uint8_t> out;
            out.push_back(STATUS_OK);
            int32_t node_count = 0;
            for (const auto& kv : poses) {
                if (kv.first > ghdr.since_node_id) {
                    node_count++;
                }
            }
            const size_t count_offset = out.size();
            out.resize(out.size() + sizeof(int32_t));
            std::memcpy(out.data() + count_offset, &node_count, sizeof(int32_t));

            for (const auto& kv : poses) {
                const int node_id = kv.first;
                if (node_id <= ghdr.since_node_id || kv.second.isNull()) {
                    continue;
                }
                auto cloud = mem != nullptr
                    ? reconstruct_node_cloud(mem, node_id, kv.second, ghdr.voxel_size, ghdr.max_depth)
                    : pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());

                double wire_pose[7];
                pose_to_wire(kv.second, wire_pose);
                const int32_t point_count = static_cast<int32_t>(cloud->size());

                const size_t hdr_off = out.size();
                out.resize(out.size() + sizeof(int32_t) + sizeof(wire_pose) + sizeof(int32_t));
                uint8_t* hp = out.data() + hdr_off;
                std::memcpy(hp, &node_id, sizeof(int32_t));
                std::memcpy(hp + sizeof(int32_t), wire_pose, sizeof(wire_pose));
                std::memcpy(hp + sizeof(int32_t) + sizeof(wire_pose), &point_count, sizeof(int32_t));

                if (point_count > 0) {
                    // Both regions must be resized BEFORE taking either
                    // pointer — std::vector::resize() may reallocate the
                    // backing buffer, which would silently invalidate a
                    // pointer taken before a later resize() call (this was a
                    // real bug here: taking `xyz` before the second resize()
                    // for the rgb region caused heap corruption/SIGSEGV on
                    // any request with enough points to trigger a
                    // reallocation).
                    const size_t xyz_off = out.size();
                    const size_t xyz_bytes = static_cast<size_t>(point_count) * 3 * sizeof(float);
                    const size_t rgb_off = xyz_off + xyz_bytes;
                    const size_t rgb_bytes = static_cast<size_t>(point_count) * 3;
                    out.resize(rgb_off + rgb_bytes);

                    float* xyz = reinterpret_cast<float*>(out.data() + xyz_off);
                    uint8_t* rgb = out.data() + rgb_off;

                    for (int32_t i = 0; i < point_count; ++i) {
                        const auto& p = cloud->points[i];
                        xyz[i * 3 + 0] = p.x;
                        xyz[i * 3 + 1] = p.y;
                        xyz[i * 3 + 2] = p.z;
                        rgb[i * 3 + 0] = p.r;
                        rgb[i * 3 + 1] = p.g;
                        rgb[i * 3 + 2] = p.b;
                    }
                }
            }

            zmq::message_t reply(out.size());
            std::memcpy(reply.data(), out.data(), out.size());
            sock.send(reply, zmq::send_flags::none);
            continue;
        }

        if (cmd != CMD_TRACK || req.size() < 1 + sizeof(TrackRequestHeader)) {
            send_status(sock, STATUS_LOST);
            continue;
        }

        TrackRequestHeader hdr;
        std::memcpy(&hdr, buf + 1, sizeof(TrackRequestHeader));

        const size_t n_pixels = static_cast<size_t>(hdr.width) * hdr.height;
        const size_t img_bytes = n_pixels * 3;
        const size_t depth_bytes = n_pixels * sizeof(float);
        const size_t expected = 1 + sizeof(TrackRequestHeader) + img_bytes + depth_bytes;
        if (req.size() != expected || hdr.width <= 0 || hdr.height <= 0) {
            std::cerr << "[rtabmap_server] malformed request (got " << req.size()
                      << " bytes, expected " << expected << ")" << std::endl;
            send_status(sock, STATUS_LOST);
            continue;
        }

        const uint8_t* img_ptr = buf + 1 + sizeof(TrackRequestHeader);
        const uint8_t* depth_ptr = img_ptr + img_bytes;

        cv::Mat rgb(hdr.height, hdr.width, CV_8UC3, const_cast<uint8_t*>(img_ptr));
        cv::Mat depth(hdr.height, hdr.width, CV_32FC1, const_cast<uint8_t*>(depth_ptr));

        // localTransform's default is CameraModel::opticalRotation() (bridges
        // a robot's base_link/REP-103 convention (X-forward,Z-up) to camera
        // optical convention (X-right,Y-down,Z-forward)) — deliberately
        // overridden to Identity here: this project has no separate
        // base/IMU frame at all (that's the whole point of dropping ORB-SLAM3's
        // IMU dependency), and its existing pipeline (occupancy_map.py,
        // feature_tracker.py, DA3) already treats "pose" as literally the
        // camera's own extrinsic in optical/Y-down convention. Identity makes
        // the pose this server returns equal RTAB-Map's internal camera pose
        // directly, matching that convention with no extra rotation to undo.
        rtabmap::CameraModel model(
            hdr.fx, hdr.fy, hdr.cx, hdr.cy,
            rtabmap::Transform::getIdentity(), 0,
            cv::Size(hdr.width, hdr.height));

        const double t_s = hdr.timestamp_ns * 1e-9;
        rtabmap::SensorData data(rgb, depth, model, next_id++, t_s);

        rtabmap::OdometryInfo odomInfo;
        rtabmap::Transform odomPose = odom->process(data, &odomInfo);
        if (odomPose.isNull()) {
            send_status(sock, STATUS_LOST);
            continue;
        }

        slam.process(data, odomPose);
        // getLastLocationId() returns the id of whichever node RTAB-Map most
        // recently ADDED to its memory graph — NOT necessarily this call's
        // frame, since process() may decide not to add a node at all this
        // call (e.g. too little displacement since the last one). Comparing
        // it against the id we assigned THIS frame's SensorData (data.id())
        // is therefore the correct, edge-case-safe test: equal means this
        // exact frame became a node; anything else (including a smaller,
        // stale id from a previous call) means it didn't. Public API,
        // confirmed present in RTAB-Map 0.23's Rtabmap.h.
        const int32_t new_node_id =
            (slam.getLastLocationId() == data.id()) ? static_cast<int32_t>(data.id()) : -1;
        // getLoopClosureId() reflects the loop closure hypothesis accepted by
        // THIS process() call only (0 if none — reset to (0,0) at the top of
        // every Rtabmap::process() call and re-zeroed if later rejected by
        // geometric verification, confirmed against RTAB-Map 0.23's
        // Rtabmap.cpp; not a stale/sticky flag). An accepted closure means
        // graph poses MAY have shifted — but RGBD/ProximityBySpace accepts
        // one against any spatially-nearby node, which fires almost every
        // frame during slow, close-range scanning even though the resulting
        // correction is negligible. Only report it to the client (triggering
        // an expensive full resync there) if the correction actually moved
        // by more than SIGNIFICANT_CORRECTION_* since the last one that did.
        uint8_t loop_closure_flag = 0;
        if (slam.getLoopClosureId() > 0) {
            const rtabmap::Transform current_correction = slam.getMapCorrection();
            const rtabmap::Transform delta = last_reported_correction.inverse() * current_correction;
            const float translation_delta = delta.getNorm();
            const float angle_delta = delta.getAngle(rtabmap::Transform::getIdentity());
            if (translation_delta > SIGNIFICANT_CORRECTION_TRANSLATION_M ||
                angle_delta > SIGNIFICANT_CORRECTION_ANGLE_RAD) {
                loop_closure_flag = 1;
                last_reported_correction = current_correction;
            }
        }
        const rtabmap::Transform pose = corrected_pose(slam, odomPose);
        if (pose.isNull()) {
            send_status(sock, STATUS_LOST);
            continue;
        }

        double out_pose[7];
        pose_to_wire(pose, out_pose);

        zmq::message_t reply(1 + sizeof(out_pose) + 1 + sizeof(int32_t));
        uint8_t* rbuf = static_cast<uint8_t*>(reply.data());
        rbuf[0] = STATUS_OK;
        std::memcpy(rbuf + 1, out_pose, sizeof(out_pose));
        rbuf[1 + sizeof(out_pose)] = loop_closure_flag;
        std::memcpy(rbuf + 1 + sizeof(out_pose) + 1, &new_node_id, sizeof(int32_t));
        sock.send(reply, zmq::send_flags::none);
    }

    return 0;
}
