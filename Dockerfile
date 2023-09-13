# Use an official Python runtime as the base image
FROM python:3.9

# Use the official Nvidia CUDA image as the base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3-pip \
	python3-setuptools \
	build-essential \
	&& \
	apt-get clean && \
	python -m pip install --upgrade pip

RUN apt-get update && apt-get install -y git

WORKDIR /workspace
COPY ./   /workspace

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install timm

RUN git clone https://github.com/Project-MONAI/MONAI.git /app/MONAI && \
    cd /app/MONAI && \
    git checkout 07de215c

# Install the MONAI package
RUN pip install /app/MONAI

RUN pip install 'monai[itk]'



