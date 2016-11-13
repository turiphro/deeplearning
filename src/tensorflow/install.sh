#!/bin/bash

# Install TensorFlow
# Not on Ubuntu Linux 64bit with python 3.5 and CPU only? See https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl
sudo pip3 install --upgrade $TF_BINARY_URL
