# Multi-architecture Dockerfile for PC (x86_64) and Raspberry Pi (arm64/arm32)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive
ENV ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

WORKDIR /app

# Install system deps (works on both x86 and ARM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    openssl \
    libsndfile1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    espeak-ng \
    libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Generate self-signed SSL cert (10yr validity) for HTTPS browser mic
RUN openssl req -x509 -newkey rsa:4096 -keyout /etc/ssl/certs/piper-key.pem -out /etc/ssl/certs/piper-cert.pem \
    -days 3650 -nodes -subj "/CN=pipertalk"

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir streamlit requests pydub PyPDF2 pymupdf
RUN pip install --no-cache-dir imageio-ffmpeg ffmpeg-python fastapi uvicorn jinja2
RUN pip install --no-cache-dir pillow pytesseract pdf2image
RUN pip install --no-cache-dir "piper-tts[onnx,espeak]"

# Copy app
COPY . .

# Create directories for models and data
RUN mkdir -p /app/data /app/models

EXPOSE 8501
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["uvicorn", "piper_ui:app", "--host=0.0.0.0", "--port=5000", "--timeout-keep-alive=300"]
