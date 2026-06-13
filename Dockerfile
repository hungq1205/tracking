# Use NVIDIA CUDA as base image to support GPU acceleration and 4-bit quantization
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
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

# Set python3.11 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /home/hungq/projects/streaming-vlm

# Pull the repository and copy local files
RUN git clone https://github.com/mit-han-lab/streaming-vlm .

# Install dependencies following the specified sequence
RUN python3.11 -m pip install --upgrade pip

RUN python3.11 -m pip install \
    packaging \
    ninja \
    wheel \
    setuptools

# Install PyTorch ecosystem and Qwen utilities
RUN python3.11 -m pip install torch==2.7.1 torchvision torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
RUN python3.11 -m pip install qwen_vl_utils==0.0.11

COPY infer_requirements.txt infer_requirements.txt

RUN if [ -f infer_requirements.txt ]; then python3.11 -m pip install -r infer_requirements.txt; fi

# Install core libraries (including bitsandbytes for 4-bit quantization support)
RUN python3.11 -m pip install transformers==4.52.3 accelerate deepspeed peft opencv-python decord datasets \
    tensorboard gradio pillow-heif gpustat timm sentencepiece openai \
    liger_kernel numpy==1.24.4 yt-dlp tqdm huggingface_hub ffmpeg wandb bitsandbytes

# Ensure grab-pov.mp4 is in the same directory as this Dockerfile
COPY grab-pov.mp4 .