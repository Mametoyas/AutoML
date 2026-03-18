FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.10 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support (for MLP / neural net models)
RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install XGBoost with GPU support
RUN pip install --no-cache-dir xgboost==2.0.3

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw results

# Set environment
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

ENTRYPOINT ["python", "main.py"]
CMD ["--use-synthetic", "--quick"]
