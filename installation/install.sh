#!/bin/bash
bash install_vizdoom_dependencies.sh
bash install_pytorch.sh
sudo apt install python3-pip
sudo apt install nvidia-cuda-dev
pip3 install -r requirements.txt
