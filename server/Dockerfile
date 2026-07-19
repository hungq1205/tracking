FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 libegl1 \
        espeak-ng libsndfile1 \
        ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

RUN pip install \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu130

RUN pip install \
        numpy==2.3.5 \
        scipy==1.17.1 \
        pandas==3.0.3 \
        opencv-contrib-python==5.0.0.93 \
        pillow==12.3.0 \
        open3d==0.19.0 \
        transformers \
        sentence-transformers==5.6.0 \
        kokoro==0.9.4 \
        misaki==0.9.4 \
        onnxruntime-gpu==1.27.0 \
        grpcio==1.82.0 \
        protobuf==7.35.1 \
        pyzmq==27.1.0 \
        gradio==6.19.0 \
        plotly==6.8.0 \
        google-genai==2.9.0 \
        xformers==0.0.35 \
        einops==0.8.2 \
        huggingface_hub \
        imageio==2.37.3 \
        omegaconf==2.3.1 \
        evo==1.36.5 \
        e3nn==0.6.0 \
        moviepy==1.0.3 \
        plyfile==1.1.3 \
        pillow_heif==1.4.0 \
        safetensors==0.8.0 \
        trimesh==4.12.2 \
        fastapi==0.136.3 \
        uvicorn==0.50.2 \
        requests==2.34.2 \
        typer==0.26.8 \
        pycolmap==4.1.0

WORKDIR /app

ARG REPO_URL=https://github.com/hungq1205/tracking
ARG REPO_REF=vi-slam
RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" /tmp/tracking-src

RUN mkdir -p /app/scan_server \
    && cp -r /tmp/tracking-src/scan_server/Depth-Anything-3 /app/scan_server/Depth-Anything-3 \
    && pip install --no-deps -e /app/scan_server/Depth-Anything-3

RUN cp /tmp/tracking-src/scan_server/da3_wrapper.py \
       /tmp/tracking-src/scan_server/scan_session.py \
       /tmp/tracking-src/scan_server/stream_session.py \
       /tmp/tracking-src/scan_server/rtabmap_client.py \
       /tmp/tracking-src/scan_server/feature_tracker.py \
       /tmp/tracking-src/scan_server/map_exporter.py \
       /tmp/tracking-src/scan_server/occupancy_map.py \
       /tmp/tracking-src/scan_server/orb_novelty_gate.py \
       /tmp/tracking-src/scan_server/pose_graph.py \
       /tmp/tracking-src/scan_server/semantic_mapper.py \
       /tmp/tracking-src/scan_server/timing_utils.py \
       /tmp/tracking-src/scan_server/zone_labeler.py \
       /tmp/tracking-src/scan_server/gemma_vlm.py \
       /app/scan_server/

# ── server/ itself ──────────────────────────────────────────────────────
RUN cp -r /tmp/tracking-src/server /app/server \
    && rm -rf /tmp/tracking-src

COPY DA3METRIC-LARGE.onnx /app/DA3METRIC-LARGE.onnx

EXPOSE 50051 7860

CMD ["python", "server/grpc_server.py"]
