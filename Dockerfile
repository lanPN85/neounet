FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

LABEL maintainer="lanpn <phan.ngoclan58@gmail.com>"

# System dependencies
RUN apt-get update && apt-get install -y\
    git python3-dev python3-pip
RUN pip3 install --upgrade pip

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install torch
ARG PYTORCH="torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html"
RUN pip install ${PYTORCH}

RUN apt-get update && apt-get install -y\
    ca-certificates apt-transport-https gnupg

RUN mkdir /workspace
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Cache folder for Torch models
RUN mkdir -p /cache/torch
ENV TORCH_HOME /cache/torch
