FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    portaudio19-dev \
    libsndfile1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app
COPY . .

EXPOSE 8501
EXPOSE 5000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]