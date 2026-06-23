#!/bin/bash
set -e

apt-get update && apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update && apt-get install -y \
    git wget curl ffmpeg libgl1 libglib2.0-0 \
    build-essential python3.11 python3.11-dev python3-pip \
    libportaudio2 portaudio19-dev
rm -rf /var/lib/apt/lists/*

update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

python3.11 -m pip install --upgrade pip
python3.11 -m pip install torch==2.7.1 torchvision torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python3.11 -m pip install qwen_vl_utils==0.0.11
python3.11 -m pip install -r /tmp/infer_requirements.txt
python3.11 -m pip install --no-build-isolation openai-whisper
python3.11 -m pip install sounddevice soundfile resampy
python3.11 -m pip install \
    transformers==4.52.3 accelerate decord \
    gradio pillow-heif gpustat timm sentencepiece \
    numpy==1.24.4 tqdm huggingface_hub bitsandbytes \
    mediapipe==0.10.35 spacy==3.8.14

apt-get update && apt-get install -y cuda-nvcc-12-8 && rm -rf /var/lib/apt/lists/*
python3.11 -m pip install flash-attn==2.7.4.post1 --no-build-isolation

cp -r /tmp/block_sparse_attn.egg \
    /usr/local/lib/python3.11/dist-packages/block_sparse_attn-0.0.2-py3.11-linux-x86_64.egg
echo '/usr/local/lib/python3.11/dist-packages/block_sparse_attn-0.0.2-py3.11-linux-x86_64.egg' \
    >> /usr/local/lib/python3.11/dist-packages/easy-install.pth

git clone https://github.com/mit-han-lab/streaming-vlm /opt/streaming-vlm
python3.11 -m pip install -e /opt/streaming-vlm/streaming_vlm/livecc_utils/

REPO=/home/hungq/projects/tracking
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch origin
    git -C "$REPO" reset --hard origin/main
else
    git clone https://github.com/hungq1205/tracking "$REPO"
fi

exec python "$REPO/server/grpc_server.py"
