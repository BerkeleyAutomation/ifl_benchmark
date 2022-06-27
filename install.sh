#!/bin/bash

# install Python dependencies in a virtualenv
virtualenv -p python3.6 venv
. venv/bin/activate
pip3 install numpy scipy gym dotmap matplotlib tqdm opencv-python pyyaml
pip3 install rl-games==1.1.4
pip3 install torch==1.10.1
pip3 install torchvision==0.11.2
