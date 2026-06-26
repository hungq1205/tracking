# flake8: noqa: E501
"""
CSS and HTML content for the Scan Server Gradio UI.
Adapted from depth_anything_3/app/css_and_html.py (Apache 2.0).
"""

import gradio as gr

SCAN_CSS = """
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

@media (prefers-color-scheme: dark) {
    html, body { background: #1e293b; color: #ffffff; }
    .gradio-container { background: #1e293b; color: #ffffff; }
    .link-btn {
        background: rgba(255,255,255,0.2); color: white;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    .link-btn:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .tech-bg {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        position: relative; overflow: hidden;
    }
    .tech-bg::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(circle at 20% 80%, rgba(59,130,246,0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139,92,246,0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(18,194,233,0.10) 0%, transparent 50%);
        animation: techPulse 8s ease-in-out infinite;
    }
    .gradio-container .panel,
    .gradio-container .block,
    .gradio-container .form {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 10px;
    }
    .gradio-container * { color: #ffffff; }
    .gradio-container label { color: #e0e0e0; }
    .gradio-container .markdown { color: #e0e0e0; }
    .description-container {
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(59,130,246,0.2);
    }
    .description-main { color: #e0e0e0; }
    .description-tip { color: #cbd5e1; }
}

@media (prefers-color-scheme: light) {
    html, body { background: #ffffff; color: #1e293b; }
    .gradio-container { background: #ffffff; color: #1e293b; }
    .tech-bg {
        background: linear-gradient(135deg, #ffffff, #f1f5f9);
        position: relative; overflow: hidden;
    }
    .link-btn {
        background: rgba(59,130,246,0.15);
        color: var(--body-text-color);
        border: 1px solid rgba(59,130,246,0.3);
    }
    .link-btn:hover {
        background: rgba(59,130,246,0.25);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59,130,246,0.2);
    }
    .tech-bg::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(circle at 20% 80%, rgba(59,130,246,0.10) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139,92,246,0.10) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(18,194,233,0.08) 0%, transparent 50%);
        animation: techPulse 8s ease-in-out infinite;
    }
    .gradio-container .panel,
    .gradio-container .block,
    .gradio-container .form {
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .gradio-container * { color: #1e293b; }
    .gradio-container label { color: #334155; }
    .gradio-container .markdown { color: #334155; }
    .description-container {
        background: linear-gradient(135deg, rgba(59,130,246,0.05) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid rgba(59,130,246,0.3);
    }
    .description-main { color: #1e293b; }
    .description-tip { color: #475569; }
}

@keyframes techPulse {
    0%, 100% { opacity: 0.5; }
    50%       { opacity: 0.8; }
}

.link-btn {
    display: inline-flex; align-items: center; gap: 8px;
    text-decoration: none; padding: 12px 24px;
    border-radius: 50px; font-weight: 500;
    transition: all 0.3s ease;
}

.custom-log * {
    font-style: italic;
    font-size: 18px !important;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    background-clip: text;
    font-weight: bold !important;
    color: transparent !important;
    text-align: center !important;
    animation: techGradient 3s ease infinite;
}

@keyframes techGradient {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.navigation-row {
    display: flex !important;
    align-items: flex-end !important;
    gap: 8px !important;
}
"""

SCAN_HEADER_HTML = """
<div class="tech-bg" style="text-align: center; margin-bottom: 5px; padding: 40px 20px; border-radius: 15px; position: relative; overflow: hidden;">
    <div style="position: relative; z-index: 2;">
        <h1 style="margin: 0; font-size: 3.5em; font-weight: 700;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: techGradient 3s ease infinite;
            letter-spacing: 2px;">
            Scan Server
        </h1>
        <p style="margin: 15px 0 0 0; font-size: 1.8em; font-weight: 300;" class="header-subtitle">
            Real-time 3D Map Builder
        </p>
        <p style="margin: 8px 0 0 0; font-size: 1em; opacity: 0.65;" class="header-subtitle">
            Upload a video or connect a gRPC client to build a 3D point-cloud map
        </p>
    </div>
</div>

<style>
    @keyframes techGradient {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @media (prefers-color-scheme: dark) {
        .header-subtitle { color: #cbd5e1; }
        .tech-bg { background: linear-gradient(135deg, #0f172a, #1e293b) !important; }
    }
    @media (prefers-color-scheme: light) {
        .header-subtitle { color: #475569; }
        .tech-bg { background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(139,92,246,0.1) 100%) !important; }
    }
</style>
"""

SCAN_DESCRIPTION_HTML = """
<div class="description-container" style="padding: 20px; border-radius: 15px; margin: 0 0 16px 0;">
    <p class="description-main" style="text-align: center; font-size: 1.2em; margin: 0; line-height: 1.7;">
        <strong>Upload a video</strong> &rarr; set a <em>Location Label</em> and optional <em>Zone</em> &rarr; click <strong>Scan</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Or stream frames from a <strong>gRPC client</strong> using <em>protobuf</em> and watch the map build live.
    </p>
    <p class="description-tip" style="text-align: center; font-style: italic; margin: 10px 0 0 0; font-size: 0.95em;">
        <i class="fas fa-lightbulb" style="color: #f59e0b; margin-right: 6px;"></i>
        <strong>Tip:</strong> Use a steady walking pace and overlapping views for best point-cloud quality.
    </p>
</div>
"""


def get_scan_theme() -> gr.themes.Base:
    """Adaptive blue-purple tech theme matching the DA3 Gradio app."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#eff6ff", c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd",
            c400="#60a5fa", c500="#3b82f6", c600="#2563eb", c700="#1d4ed8",
            c800="#1e40af", c900="#1e3a8a", c950="#172554",
        ),
        secondary_hue=gr.themes.Color(
            c50="#f5f3ff", c100="#ede9fe", c200="#ddd6fe", c300="#c4b5fd",
            c400="#a78bfa", c500="#8b5cf6", c600="#7c3aed", c700="#6d28d9",
            c800="#5b21b6", c900="#4c1d95", c950="#2e1065",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1",
            c400="#94a3b8", c500="#64748b", c600="#475569", c700="#334155",
            c800="#1e293b", c900="#0f172a", c950="#020617",
        ),
    )
