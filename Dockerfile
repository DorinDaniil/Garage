FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Arguments to build Docker Image using CUDA
# ARG USE_CUDA=0
ARG USE_CUDA=1
ARG TORCH_ARCH=

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA True
ENV CUDA_HOME /usr/local/cuda-11.6/
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    git \
    wget \
    vim \
    sudo \
    tar \
    unzip \
    openssh-server \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    swig \
    libopenblas-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python3-venv \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# symlink for python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# add user and his password
ENV USER=augmenter_docker
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}

# create some directories for mounting volumes
RUN mkdir augmenter_pipeline && chown -R ${UID}:${GID} /home/${USER}
COPY . /home/${USER}/augmenter_pipeline

USER root

# Ensure proper permissions for the augmenter_pipeline directory
RUN chown -R ${UID}:${GID} /home/${USER}/augmenter_pipeline
RUN chmod -R u+w /home/${USER}/augmenter_pipeline

USER ${UID}:${GID}

ENV PATH="/home/${USER}/.local/bin:$PATH"

# Configure Git to trust the repository's directory
RUN git config --global --add safe.directory /home/${USER}/augmenter_pipeline

# create venv
ENV VIRTUAL_ENV="/home/${USER}/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Install PyTorch with the correct CUDA version
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html

# install models for segmentation
RUN python -m pip install --no-cache-dir -e /home/${USER}/augmenter_pipeline/GenerativeAugmentations/models/GroundedSegmentAnything/segment_anything
# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN python -m pip install --no-cache-dir wheel
WORKDIR /home/${USER}/augmenter_pipeline/GenerativeAugmentations/models/GroundedSegmentAnything/GroundingDINO
RUN python /home/${USER}/augmenter_pipeline/GenerativeAugmentations/models/GroundedSegmentAnything/GroundingDINO/setup.py build
RUN python /home/${USER}/augmenter_pipeline/GenerativeAugmentations/models/GroundedSegmentAnything/GroundingDINO/setup.py install

WORKDIR /home/${USER}/augmenter_pipeline
# download weights
RUN git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/ /home/${USER}/augmenter_pipeline/checkpoints/ppt-v2-1
RUN wget -O /home/${USER}/augmenter_pipeline/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN wget -O /home/${USER}/augmenter_pipeline/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


# upgrade pip
ARG PIP_VERSION=23.3.1
ARG SETUPTOOLS_VERSION=68.2.2
RUN pip install pip==${PIP_VERSION} setuptools==${SETUPTOOLS_VERSION}

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# CMD ["python", "app.py"]
