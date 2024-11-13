FROM nvidia/cuda:12.6.1-base-ubuntu24.04

#Run the frontend first so it doesn't throw an error later
RUN apt-get update \
  && export TZ="America/New_York" \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
  && ln -fs "/usr/share/zoneinfo/$TZ" /etc/localtime \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && apt-get clean

# General dependencies for development
RUN apt-get update \
  && apt-get install -y \
  build-essential \
  cmake \
  cppcheck \
  gdb \
  git \
  libeigen3-dev \
  g++ \
  libbluetooth-dev \
  libcwiid-dev \
  libgoogle-glog-dev \
  libspnav-dev \
  libusb-dev \
  libpcl-dev \
  lsb-release \
  mercurial \
  python3-dbg \
  python3-empy \
  python3-pip \
  python3-venv \
  software-properties-common \
  sudo \
  wget \
  curl \
  cmake-curses-gui \
  geany \
  tmux \
  dbus-x11 \
  iputils-ping \
  default-jre \
  iproute2 \
  vim \
  && apt-get clean

# remove ubuntu user
RUN userdel -f ubuntu

# add a user
ENV user_id=1000
ARG USER jason
RUN useradd -U --uid ${user_id} -ms /bin/bash $USER \
 && echo "$USER:$USER" | chpasswd \
 && adduser $USER sudo \
 && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER

# Set locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
  locale-gen
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=n_US:en

# Commands below run as the developer user
USER $USER

# Install ROS Jazzy
RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && sudo apt-get update \
  && sudo apt-get install -y \
  ros-dev-tools \
  python3-rosdep \
  python3-vcstool \
  ros-jazzy-desktop

RUN sudo rosdep init \
  && sudo apt-get clean

RUN rosdep update

#RUN python3 -m venv torch \
# && source torch/bin/activate \
# && pip3 install torch torchvision torchaudio

#RUN pip3 install torch torchvision torchaudio --break-system-packages

WORKDIR /home/$USER

COPY ./ml-depth-pro-ros ./ml-depth-pro-ros
RUN cd ml-depth-pro-ros \
 && sudo pip3 install . --break-system-packages

RUN mkdir -p ws/src
RUN sudo chown -R $USER:$USER ws/
