FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1 \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    python3-dev \
    wget \
    unzip \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install --upgrade pip && \
    pip install \
        streamlit \
        matplotlib \
        scikit-learn \
        opencv-python \
        pycocotools \
        albumentations \
        pytorch-lightning \
        torchmetrics \
        tqdm \
        pandas \
        pyyaml \
        seaborn \
        -U tensorboard tensorboardX

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install 'git+https://github.com/facebookresearch/detectron2.git@main#egg=detectron2' && \
    pip install 'git+https://github.com/openai/CLIP.git'

COPY . /app

EXPOSE 8501

CMD ["python", "run.py"]
