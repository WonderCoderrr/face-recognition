#!/bin/bash

# Create a new conda environment
conda create -n face-recognition-env python=3.10 -y

# Activate the new environment
conda activate face-recognition-env

# Install libs
conda install opencv
conda install pyyaml

# Run the main Python script
python main.py

# Deactivate the env
conda deactivate
