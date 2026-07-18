"""
Gradio GUI for frame_extractor — upload a video, get back only the frames
whose ORB features weren't already covered by a previously-accepted frame
(see extractor.py's docstring). RTAB-Map is optional (pose/node_id metadata
only, no longer gates acceptance). RAM++ tagging -> GroundingDINO-tiny
detection (tagging.py) is also optional — when enabled, every returned frame
additionally shows its RAM++ tags and GroundingDINO boxes.

Run:
    conda activate hrtf   # same env scan_server/ is developed against
    python frame_extractor/app.py
"""
import os

import gradio as gr

from extractor import extract_new_frames
from tagging import FrameTagger

# FrameTagger loads a 3GB RAM++ checkpoint + GroundingDINO tiny (~1-2 min,
# see tagging.py) — built once on first use and cached here instead of
# reloading it on every "Extract new frames" click.
_TAGGER_CACHE = {}


def _get_tagger(ram_checkpoint, gdino_model_id, device):
    key = (ram_checkpoint, gdino_model_id, device)
    if key not in _TAGGER_CACHE:
        _TAGGER_CACHE.clear()  # only one tagger resident on the GPU at a time
        _TAGGER_CACHE[key] = FrameTagger(
            ram_checkpoint=ram_checkpoint, gdino_model_id=gdino_model_id, device=device,
        )
    return _TAGGER_CACHE[key]


def run(video_file, use_rtabmap, rtabmap_addr, sample_fps, min_new_fraction, min_new_count,
        orb_features, min_raw_matches, ransac_threshold_px, min_rotation_deg, min_sharpness,
        da3_batch_size, da3_model, da3_onnx_path,
        use_tagging, ram_checkpoint, gdino_model_id, tag_batch_size,
        device, progress=gr.Progress()):
    if video_file is None:
        return [], "Upload a video first.", []

    def _progress_cb(done, total):
        progress(done / max(total, 1), desc=f"Frame {done}/{total}")

    frame_tagger = None
    if use_tagging:
        try:
            progress(0, desc="Loading RAM++ / GroundingDINO tiny (first run only)...")
            frame_tagger = _get_tagger(ram_checkpoint.strip(), gdino_model_id.strip(), device)
        except Exception as e:
            return [], f"Failed to load tagging models: {e}", []

    try:
        new_frames = extract_new_frames(
            video_path=video_file,
            sample_fps=sample_fps,
            min_new_fraction=min_new_fraction,
            min_new_count=int(min_new_count),
            orb_features=int(orb_features),
            min_raw_matches=int(min_raw_matches),
            ransac_threshold_px=ransac_threshold_px,
            min_rotation_deg=min_rotation_deg,
            min_sharpness=min_sharpness,
            rtabmap_addr=(rtabmap_addr.strip() or None) if use_rtabmap else None,
            da3_batch_size=int(da3_batch_size),
            da3_model=da3_model,
            da3_onnx_path=da3_onnx_path.strip() or None,
            device=device,
            frame_tagger=frame_tagger,
            tag_batch_size=int(tag_batch_size),
            progress_cb=_progress_cb,
        )
    except Exception as e:
        return [], f"Failed: {e}", []

    # gr.Gallery captions render as a single truncated line under the
    # thumbnail — fine for a short summary, unusable for tags/prompt/
    # detections (see the real caption pasted in chat: one unreadable
    # run-on line). So the gallery only gets the short summary; the full
    # per-frame breakdown (tags, GroundingDINO prompt, detections) goes into
    # `details` (one Markdown string per frame, same order) and is shown in
    # a separate panel when that frame is clicked (see gallery.select below).
    gallery = []
    details = []
    for nf in new_frames:
        summary = (
            f"frame {nf.frame_idx}  ·  t={nf.timestamp_s:.2f}s  ·  "
            f"{nf.new_feature_count} new ({nf.new_feature_fraction:.0%})"
        )
        if nf.sharpness:
            summary += f"  ·  sharpness {nf.sharpness:.0f}"
        gallery.append((nf.image_rgb, summary))

        lines = [
            f"### Frame {nf.frame_idx} · t={nf.timestamp_s:.2f}s",
            f"- **New features:** {nf.new_feature_count} ({nf.new_feature_fraction:.0%})",
            f"- **Vs. best match:** {nf.best_match_inliers} inliers, "
            f"{nf.pose_rotation_deg:.1f}° rot, {nf.pose_translation_px:.1f}px",
        ]
        if nf.sharpness:
            lines.append(f"- **Sharpness:** {nf.sharpness:.0f}")
        if nf.node_id != -1:
            lines.append(f"- **RTAB-Map node:** {nf.node_id}")
        if nf.tags:
            lines.append(f"- **RAM++ tags:** {', '.join(nf.tags)}")
        if nf.tag_prompt:
            lines.append(f"- **GroundingDINO prompt:** `{nf.tag_prompt}`")
        if nf.tag_detections:
            dets = ", ".join(f"{d.label} ({d.score:.2f})" for d in nf.tag_detections)
            lines.append(f"- **GroundingDINO detections:** {dets}")
        details.append("\n".join(lines))

    status = f"{len(new_frames)} new frame(s) found."
    return gallery, status, details


with gr.Blocks(title="Frame Extractor — ORB new-feature detection") as demo:
    gr.Markdown(
        "# Frame Extractor\n"
        "Mirrors how RTAB-Map's own visual odometry actually judges novelty: ORB detect → "
        "descriptor match against each previously-accepted frame → Essential Matrix + RANSAC "
        "geometric verification → only RANSAC inliers count as 'already seen'. Raw keypoint "
        "positions are NOT compared directly (ORB detections aren't stable frame to frame). "
        "A person walking through a static shot won't count as new; panning to an unseen part "
        "of the room will. Checked against every accepted frame so far, so panning back to an "
        "earlier view won't be re-accepted.\n\n"
        "Optionally, every accepted frame is also tagged with RAM++ (open-set image tagging) "
        "and those tags are fed straight into GroundingDINO tiny as its detection prompt — so "
        "each frame gets boxed for whatever RAM++ actually found in it, no fixed vocabulary "
        "needed up front."
    )
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input video")
            sample_fps = gr.Slider(minimum=0.5, maximum=15.0, value=5.0, step=0.5, label="Sampling FPS")
            min_new_fraction = gr.Slider(
                minimum=0.05, maximum=1.0, value=0.9, step=0.05,
                label="Min fraction of new (non-inlier) ORB features",
                info="A frame is only marked new once this much of it is unexplained by any prior "
                     "accepted frame. Raise toward 1.0 for stricter novelty; lower if real new views "
                     "are getting skipped.",
            )
            min_new_count = gr.Slider(
                minimum=0, maximum=1000, value=500, step=50,
                label="Min raw count of new (non-inlier) ORB features",
                info="Guards against accepting low-texture frames on noisy fraction alone.",
            )
            orb_features = gr.Slider(minimum=200, maximum=4000, value=1500, step=100, label="ORB features per frame")
            min_raw_matches = gr.Slider(
                minimum=4, maximum=100, value=20, step=1,
                label="Min descriptor matches before attempting RANSAC",
                info="Below this, a reference frame is skipped outright — clearly not the same view.",
            )
            ransac_threshold_px = gr.Slider(
                minimum=0.5, maximum=10.0, value=3.0, step=0.5,
                label="RANSAC inlier threshold (px)",
                info="Looser = more forgiving of motion blur, but lets more false matches through.",
            )
            min_rotation_deg = gr.Slider(
                minimum=0.0, maximum=180.0, value=30.0, step=5.0,
                label="Min pose rotation (deg)",
                info="A frame is only accepted if its recovered pose vs. the best-matching accepted "
                     "frame rotated at least this much — guards against jitter/noise looking like a "
                     "new view when the camera barely moved. Only applies once a matching reference "
                     "exists. (Translation isn't gated on — monocular translation has no metric scale "
                     "— but pixel displacement is still shown per frame below.)",
            )
            min_sharpness = gr.Slider(
                minimum=0.0, maximum=1000.0, value=100.0, step=10.0,
                label="Min sharpness (variance of Laplacian, 0=disabled)",
                info="A frame that otherwise passes every other check is still skipped if it's too "
                     "blurry — not accepted as a reference, not returned — so a temporarily out-of-"
                     "focus pass over new territory doesn't get 'used up' on a bad frame; a later, "
                     "sharper pass over the same area can still be accepted. Scales with resolution/"
                     "content, so tune per video (0 disables the check entirely).",
            )
            use_rtabmap = gr.Checkbox(
                label="Also fetch RTAB-Map pose/node_id (metadata only, doesn't affect selection)",
                value=False,
            )
            rtabmap_addr = gr.Textbox(
                label="RTAB-Map address",
                value=os.getenv("RTABMAP_ADDR", "tcp://localhost:5556"),
            )
            da3_batch_size = gr.Slider(
                minimum=1, maximum=64, value=32, step=1,
                label="DA3 batch size (RTAB-Map only)",
                info="Frames processed per DA3 multi-view depth inference call.",
            )
            da3_model = gr.Radio(["torch", "onnx"], value="onnx", label="DA3 depth backend (RTAB-Map only)")
            da3_onnx_path = gr.Textbox(label="DA3 ONNX path (only if backend=onnx)", value="DA3METRIC-LARGE.onnx")
            use_tagging = gr.Checkbox(
                label="Tag + detect each accepted frame (RAM++ -> GroundingDINO tiny)",
                value=False,
                info="First use loads a ~3GB RAM++ checkpoint + GroundingDINO tiny (~1-2 min); "
                     "cached in memory after that, not reloaded per run.",
            )
            ram_checkpoint = gr.Textbox(
                label="RAM++ checkpoint path",
                value=os.getenv(
                    "RAM_PLUS_CHECKPOINT",
                    os.path.join(os.path.dirname(__file__), "weights", "ram_plus_swin_large_14m.pth"),
                ),
            )
            gdino_model_id = gr.Textbox(
                label="GroundingDINO tagging model id",
                value="IDEA-Research/grounding-dino-tiny",
            )
            tag_batch_size = gr.Slider(
                minimum=1, maximum=8, value=1, step=1,
                label="Tagging batch size",
                info="RAM++/GroundingDINO frames processed per batched forward pass — independent "
                     "of the DA3 batch size above (different VRAM/speed profile, no chronological "
                     "dependency).",
            )
            device = gr.Radio(["cuda", "cpu"], value="cuda", label="Device")
            run_btn = gr.Button("Extract new frames", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="New frames (click one for full details)", columns=4, height=600)
            details_box = gr.Markdown(label="Frame details", value="Click a frame above for its tags/prompt/detections.")

    details_state = gr.State([])  # one Markdown string per gallery frame, same order

    run_btn.click(
        run,
        inputs=[video_input, use_rtabmap, rtabmap_addr, sample_fps, min_new_fraction, min_new_count,
                orb_features, min_raw_matches, ransac_threshold_px, min_rotation_deg, min_sharpness,
                da3_batch_size, da3_model, da3_onnx_path,
                use_tagging, ram_checkpoint, gdino_model_id, tag_batch_size,
                device],
        outputs=[gallery, status, details_state],
    )

    def _show_details(details, evt: gr.SelectData):
        if evt.index is None or evt.index >= len(details):
            return "No details for this frame."
        return details[evt.index]

    gallery.select(_show_details, inputs=[details_state], outputs=[details_box])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("FRAME_EXTRACTOR_PORT", "7863")))
