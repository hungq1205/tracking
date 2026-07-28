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

# Model weights live outside /app so /app can stay a clean git working
# directory that entrypoint.sh clones/updates fresh on every container
# start (no application code is copied into the image at build time at
# all any more) — baked in here for a self-contained deployment (no host
# volume mount needed at runtime just for the weights).
ENV DA3_ONNX_PATH=/opt/models/DA3METRIC-LARGE.onnx
COPY DA3METRIC-LARGE.onnx ${DA3_ONNX_PATH}

# Pre-cache GroundingDINO-base (server/tools/detector.py's GroundingDINODetector
# default model_id — the TrackingService/PerceptionService.AnalyzeFrame(DETECT)
# detector) into the image's HF cache, same "baked in at build time, not
# fetched at runtime" precedent as DA3METRIC-LARGE.onnx above — just via
# from_pretrained's own cache instead of a single-file COPY, since a HF Hub
# model has no equivalent standalone artifact. CPU-only here (no GPU during
# a normal `docker build`) — this only downloads/caches the weights, the
# same processor/model construction detector.py itself does at runtime,
# minus .to(device)/.half(), which need a real CUDA device to be meaningful.
RUN python3.11 -c "\
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; \
model_id = 'IDEA-Research/grounding-dino-base'; \
AutoProcessor.from_pretrained(model_id, use_fast=True); \
AutoModelForZeroShotObjectDetection.from_pretrained(model_id, low_cpu_mem_usage=True); \
print(f'[build] Pre-cached {model_id}')"

ENV REPO_URL=https://github.com/hungq1205/tracking \
    REPO_REF=vi-slam

WORKDIR /app

COPY server/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 50051 7860

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "server/grpc_server.py"]
