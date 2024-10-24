#!/bin/bash

# create the conda environment
conda create -n videoinsta python=3.11
conda activate videoinsta

# install general dependencies
# note that this requires CUDA 11.3
# PLEASE ADOPT THIS LINE DEPENDING ON YOUR SYSTEM AND CUDA VERSION
# take a look at: https://pytorch.org/get-started/locally/
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 xformers
pip install transformers
pip install python-dotenv
pip install apex

# install llama dependencies
pip install sentencepiece
pip install accelerate
pip install protobuf

# install LaViLa dependencies
pip install timm
pip install decord
pip install einops
pip install pandas
pip install pytorchvideo
pip install ftfy
pip install spacy
pip install scikit-learn
# note that this may raise an error which can be ignored
pip install git+https://github.com/Maluuba/nlg-eval.git@master

# install UniVTG dependencies
pip install ffmpeg