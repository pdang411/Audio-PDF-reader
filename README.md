# PIPER-Audiobook

PDF Audiobook Reader with Piper TTS - works on PC, Raspberry Pi 4, and any Linux system.

## Features

- **PDF Reader**: Upload or load PDFs from server
- **Text-to-Speech**: Uses Piper TTS for natural sounding voice
- **Page Navigation**: Navigate through PDF pages easily
- **Voice Commands**: Use microphone for hands-free control
- **OCR Support**: Read scanned/image PDFs
- **Speed Control**: Adjust reading speed 0.5x to 2.0x

## Quick Start (Docker)

### On PC or Raspberry Pi 4

```bash
# Clone or download the project
cd PIPER-Audiobook

# Build and run with Docker Compose
docker-compose up -d --build

# Access the apps:
# - PDF Reader UI: http://localhost:8501
# - Piper TTS UI:  http://localhost:5000/ui
```

### Raspberry Pi 4 Specific

Make sure you have Docker installed:
```bash
curl -sSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

## Manual Installation (No Docker)

### Ubuntu/Debian/Raspberry Pi OS

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3-pip ffmpeg tesseract-ocr

# Install Python packages
pip3 install -r requirements.txt

# Start Piper TTS server (terminal 1)
python3 piper_ui.py

# Start PDF Reader (terminal 2)
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

### Windows

1. Install Python 3.11+
2. Install ffmpeg (add to PATH)
3. Install Tesseract OCR (optional, for scanned PDFs)
4. Run: `pip install -r requirements.txt`
5. Start Piper: `python piper_ui.py`
6. Start Streamlit: `streamlit run app.py`

## Usage

1. Open http://localhost:8501
2. Upload a PDF or select from server
3. Navigate pages with sidebar controls
4. Click "Start Reading" to generate audio
5. Download audio as WAV file

### Voice Commands

Try saying:
- "next" - go to next page
- "previous" - go to previous page
- "page 5" - go to page 5
- "read page" - generate audio for current page
- "speed 1.5" - set reading speed

## Docker Architecture

Two containers run together:
- **piper-tts**: TTS server on port 5000
- **pdf-reader**: Streamlit app on port 8501

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| TTS_BASE_URL | http://localhost:5000 | Piper TTS server URL |
| MODEL | en_US-lessac-medium | Default voice model |
| DATA_DIR | /app/data | Voice models directory |

## Troubleshooting

### "Internal Server Error"
- Make sure Docker containers are running: `docker-compose ps`
- Check Piper TTS is healthy: `curl http://localhost:5000/health`
- View logs: `docker-compose logs -f`

### Slow on Raspberry Pi
- Use a lighter voice model
- Enable OCR only when needed
- Use smaller PDFs first

### OCR not working
- Install Tesseract: `sudo apt-get install tesseract-ocr`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

Piper TTS set up once your docker container is running go to http://localhost:5000/ui to download Voice model for Piper TTS

<img width="2514" height="1370" alt="Piper TTS set up 1" src="https://github.com/user-attachments/assets/eb1e39b6-5f9d-4be3-8e0d-4e16f7e763c4" />

<img width="2498" height="1376" alt="Piper TTS set up" src="https://github.com/user-attachments/assets/841eb869-897f-411b-a881-20a8be755dc7" />


## License

MIT
