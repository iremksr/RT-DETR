FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace/rtdetr_pytorch

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/rtdetr_pytorch \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY rtdetr_pytorch/requirements.txt /tmp/requirements.txt

# PyTorch ve torchvision CUDA imajında zaten kurulu.
# Diğer RT-DETR bağımlılıklarını yükle.
RUN grep -v -E '^(torch|torchvision)==' \
    /tmp/requirements.txt > /tmp/requirements-extra.txt \
    && pip install -r /tmp/requirements-extra.txt \
    && pip install "numpy<2"

COPY rtdetr_pytorch/ /workspace/rtdetr_pytorch/

CMD ["bash"]