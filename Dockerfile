FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CC=gcc
ENV VLM_MODEL_PATH=/models/qwen/3B
ENV VLM_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct

COPY infer_requirements.txt /tmp/infer_requirements.txt
COPY block_sparse_attn.egg/ /tmp/block_sparse_attn.egg/
COPY setup.sh /setup.sh

CMD ["bash", "/setup.sh"]
