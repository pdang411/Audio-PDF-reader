# Multi-architecture Dockerfile for PC (x86_64) and Raspberry Pi (arm64/arm32)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system deps (works on both x86 and ARM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libsndfile1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

# Install dependencies in stages for better caching
RUN pip install --no-cache-dir streamlit requests pydub PyPDF2 pymupdf
RUN pip install --no-cache-dir imageio-ffmpeg ffmpeg-python fastapi uvicorn jinja2
RUN pip install --no-cache-dir pillow pytesseract pdf2image
RUN pip install --no-cache-dir "piper-tts[onnx,espeak]"
RUN pip install --no-cache-dir "starlette==0.46.0"

# Copy app
COPY . .

# Create directories for models and data
RUN mkdir -p /app/data /app/models

EXPOSE 8501
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "-c", "import piper_ui; piper_ui.run_dual_server()"]
