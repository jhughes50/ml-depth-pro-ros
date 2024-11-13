#!/bin/bash

set -eo pipefail

docker run -it --rm --gpus all \
    --network=host \
    -u $UID \
    --privileged \
    -e "TERM=xterm-256color" \
    -v "/dev:/dev" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "./ml-depth-pro-ros/ros:/home/`whoami`/ws/src" \
    -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
    -v "/etc/localtime:/etc/localtime:ro" \
    --security-opt seccomp=unconfined \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    --name depth-pro-$(hostname)-base \
    depth-pro-$(hostname):base \
    bash
