FROM ubuntu:24.04

RUN apt update && apt install -y \
    x11-apps \
    mesa-utils \
    vulkan-tools \
    cmake \
    clang \
    build-essential \
    libasound2-dev \
    libpulse-dev \
    libopenal-dev \
    zlib1g-dev \
    libedit-dev \
    libvulkan-dev \
    libudev-dev \
    git \
    libevdev-dev \
    libjack-dev \
    libsndio-dev \
    libx11-dev \
    libxext-dev

WORKDIR /workspace

RUN apt-get install -y sudo \
    && echo ubuntu ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

CMD ["bash"]

