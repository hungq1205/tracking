# # Step 1 — compile the wheel (runs builder fully before anything else)
# docker build \
#   -f Dockerfile.builder \
#   -t streaming-vlm-builder \
#   .

# Step 2 — build the runtime image (pulls wheel from the named image above)
# docker build .

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    build-essential \
    git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

# Required build deps
RUN python3.11 -m pip install \
    packaging \
    ninja \
    wheel \
    setuptools

# Install torch first
RUN python3.11 -m pip install \
    torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=8

# Install Block Sparse Attention
RUN git clone --depth 1 \
    https://github.com/mit-han-lab/Block-Sparse-Attention.git \
    /tmp/Block-Sparse-Attention && \
    cd /tmp/Block-Sparse-Attention && \
    python3.11 setup.py install && \
    rm -rf /tmp/Block-Sparse-Attention
