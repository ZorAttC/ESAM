FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git wget curl build-essential cmake \
    python3.8 python3.8-dev python3-pip \
    libopenblas-dev libomp-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \  
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 和 pip 链接
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 安装 Python 基本工具
RUN pip install --upgrade pip setuptools wheel

# 安装 PyTorch + torchvision（针对 CUDA 11.6）
RUN pip install torch==1.13.1 torchvision --index-url https://download.pytorch.org/whl/cu116

# 安装 mmcv/mmengine/mmdet/mmdet3d
RUN pip install mmengine==0.10.3 \
    && pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.1/index.html \
    && pip install mmdet==3.2.0 \
    && pip install mmdet3d==1.4.0

# 克隆并安装 MinkowskiEngine（源码编译）
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /MinkowskiEngine && \
    cd /MinkowskiEngine && \
    python setup.py install --blas=openblas 

# 切换到非 root 用户
USER $USERNAME

# 克隆 segmentator 到 data 目录
WORKDIR /home/$USERNAME/workspace/data
RUN git clone https://github.com/Karbo123/segmentator.git

# 克隆 PointOps 并编译安装（假设需要）
WORKDIR /home/$USERNAME/workspace


# 安装其他依赖
RUN pip install --user \
    h5py==3.9.0 \
    icecream==2.1.3 \
    imageio==2.34.1 \
    matplotlib==3.5.2 \
    natsort==8.4.0 \
    numba==0.57.0 \
    numpy==1.24.1 \
    open3d==0.17.0 \
    opencv-python==4.7.0.72 \
    pandas==2.0.1 \
    pillow==10.2.0 \
    plyfile==1.0.2 \
    requests==2.31.0 \
    scikit-learn==1.2.2 \
    scipy==1.10.1 \
    setuptools==69.2.0 \
    tqdm==4.65.0 \
    trimesh==3.22.2

# 默认进入 bash
CMD ["bash"]