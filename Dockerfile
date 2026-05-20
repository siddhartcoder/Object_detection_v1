# ============================================================
# BDD100K Data Analysis — Docker Container
# ============================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (fixed for Debian trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    ffmpeg \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (kept for extendability)
COPY requirements.txt .

# Install Python dependencies
# RUN pip install --no-cache-dir \
#     numpy>=1.24.0 \
#     pandas>=2.0.0 \
#     opencv-python-headless>=4.8.0 \
#     Pillow>=10.0.0 \
#     matplotlib>=3.7.0 \
#     seaborn>=0.12.0 \
#     plotly>=5.15.0 \
#     dash>=2.11.0 \
#     dash-bootstrap-components>=1.4.0 \
#     scipy>=1.11.0 \
#     scikit-learn>=1.3.0 \
#     tqdm>=4.65.0 \
#     pyyaml>=6.0.0

RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    opencv-python-headless>=4.8.0 \
    Pillow>=10.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    plotly>=5.15.0 \
    dash>=2.11.0 \
    dash-bootstrap-components>=1.4.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0.0 \
    ultralytics>=8.0.0


# Copy application source code
COPY src/ ./src/
COPY main.py .
COPY README.md .

# Create directories (containers expect these paths)
RUN mkdir -p /app/data/labels /app/data/images /app/results

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default command
CMD ["python", "main.py", "--task", "analysis", \
     "--train_json", "/app/data/labels/bdd100k_labels_images_train.json", \
     "--val_json", "/app/data/labels/bdd100k_labels_images_val.json", \
     "--image_dir", "/app/data/images", \
     "--output_dir", "/app/results/analysis"]

# Expose dashboard port
EXPOSE 8080

# Metadata
LABEL maintainer="Siddhart Asthana"
LABEL description="BDD100K Object Detection EDA — Bosch Applied CV Assignment"
LABEL version="1.0"