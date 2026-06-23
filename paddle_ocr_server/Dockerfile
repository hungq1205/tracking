FROM paddlepaddle/paddle:3.3.1-gpu-cuda12.4-cudnn9.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python3 -c "import glob; paths=glob.glob('/usr/local/lib/python3.10/dist-packages/nvidia/*/lib'); open('/etc/ld.so.conf.d/nvidia-pip.conf','w').write('\n'.join(paths))" && \
    ldconfig

COPY server.py .

EXPOSE 8100

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]