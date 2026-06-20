# # Step 1 — compile the wheel (runs builder fully before anything else)
# docker build \
#   -f Dockerfile.builder \
#   -t streaming-vlm-builder \
#   .

# Step 2 — build the runtime image (pulls wheel from the named image above)
# docker build .

# Pull the pre-compiled wheel from the separately built builder image
FROM streaming-vlm-builder AS builder

# Runtime image — no nvcc, no devel headers
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /home/hungq/projects/streaming-vlm

# Shallow clone + strip eval toolkit and git history (not needed at inference time)
RUN git clone --depth 1 --single-branch https://github.com/mit-han-lab/streaming-vlm . && \
    rm -rf .git eval/

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install packaging ninja wheel setuptools

# PyTorch in its own layer — cache survives requirements changes
RUN python3.11 -m pip install torch==2.7.1 torchvision torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

RUN python3.11 -m pip install qwen_vl_utils==0.0.11

COPY infer_requirements.txt infer_requirements.txt

# PaddlePaddle must be installed before paddleocr resolves its backend dependency
RUN python3.11 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu128/

RUN python3.11 -m pip install -r infer_requirements.txt

# Packages not covered by infer_requirements.txt (duplicates removed)
RUN python3.11 -m pip install \
    transformers==4.52.3 accelerate peft pillow-heif gpustat timm sentencepiece \
    liger_kernel numpy==1.24.4 bitsandbytes

# Install the pre-compiled wheel from the builder stage — no nvcc needed here
COPY --from=builder /dist/*.whl /tmp/
RUN python3.11 -m pip install /tmp/*.whl && rm -f /tmp/*.whl

COPY grab-pov.mp4 .
