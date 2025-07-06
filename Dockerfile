# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set up Python
RUN apt-get update && apt-get install -y python3.9 python3.9-venv python3-pip git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Entrypoint for FastAPI with Uvicorn and hot reload off
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 