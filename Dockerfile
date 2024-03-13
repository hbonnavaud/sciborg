FROM ubuntu:22.04
# I prefer 22.04 than latest because pytorch is compatible with cuda 11.8 or 12.1 which are not
# (at the moment) compatible with the latest ubuntu version.
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update
RUN apt-get install -y \
   libglfw3 \
   curl \
   git \
   libgl1-mesa-dev \
   libgl1-mesa-glx \
   libglew-dev \
   libosmesa6-dev \
   python3-pip \
   python3-numpy \
   python3-scipy \
   net-tools \
   unzip \
   vim \
   wget \
   xpra \
   xserver-xorg-dev\
   patchelf\
   python3-dev

RUN echo 'lb_release -a'

# Install mujoco
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN tar -xf mujoco210-linux-x86_64.tar.gz
RUN mkdir ~/.mujoco
RUN mv mujoco210 ~/.mujoco/.
RUN rm mujoco210-linux-x86_64.tar.gz

# Install mujoco_py
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -U 'mujoco-py<2.2,>=2.1'
RUN python3 -m pip install "cython<3" numpy torch gym gymnasium matplotlib
