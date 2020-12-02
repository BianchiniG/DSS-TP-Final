#!/bin/bash

sudo apt-get update
sudo apt-get install -y unzip wget libgl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.zip
unzip cmake-3.19.1.zip
cd cmake-3.19.1
./bootstrap
sudo make
sudo make install
rm -f cmake-3.19.1.zip
rm -rf cmake-3.19.1
