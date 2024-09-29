FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV TZ=America/New_York

RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# install the required python packages
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt \
    && pip install 'git+https://github.com/facebookresearch/detectron2.git' \
    && ln -s /usr/bin/python3 /usr/bin/python

# install some extra packages
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    rsync \
    && rm -rf /var/lib/apt/lists/*


