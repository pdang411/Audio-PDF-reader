# Audio PDF Reader

PDF Audiobook Reader with Piper TTS — works on PC (x86_64) and Raspberry Pi (arm64/arm32).

## Features

- **PDF Reader**: Upload or load PDFs from server, page-by-page with image preview
- **Text-to-Speech**: High-quality neural TTS via Piper with customizable voice settings
- **Page Navigation**: Next/prev, first/last, direct page input, chapter/bookmark navigation
- **Voice Commands**: Hands-free control via browser microphone (Web Speech API)
- **OCR Support**: Tesseract OCR for scanned/image PDFs
- **Speed Control**: Adjust reading speed 0.5x–2.0x with pitch preservation
- **Sync Highlighting**: Word-by-word highlighting synchronized with audio playback
- **Audio Download**: Export pages as WAV files
- **Voice Pipeline**: Real-time LLM → TTS streaming via WebSocket (`/voice-pipeline`)
- **Agent System**: AI agent with web search, weather, file control, YouTube, and more

## Quick Start (Docker Compose)

```bash
docker compose up -d --build
```

Then access:
- **PDF Reader UI**: http://localhost:8501
- **Piper TTS Control Panel**: http://localhost:5000/ui
- **Voice Pipeline (AI chat)**: http://localhost:5000/voice-pipeline

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TTS_BASE_URL` | `http://piper-tts:5000` | Piper TTS server URL |
| `MODEL` | `en_US-lessac-medium` | Default voice model |
| `DATA_DIR` | `/app/data` | Voice models directory |
| `LLM_BASE_URL` | `http://192.168.1.16:11434` | Ollama LLM URL |
| `TTS_MAX_CHARS` | `10000` | Max chars per TTS chunk |

### Docker Bridge Network

Both containers communicate over a dedicated bridge network (`172.21.0.0/16`):
- `piper-tts` → `172.21.0.10` (port 5000)
- `pdf-reader` → `172.21.0.11` (port 8501)

## Manual Installation

```bash
pip install -r requirements.txt

# Terminal 1 — Start Piper TTS
python piper_ui.py

# Terminal 2 — Start PDF Reader
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## Voice Commands

Prefix commands with "pdf":
- `pdf next` / `pdf back` — page navigation
- `pdf page 5` — go to page 5
- `pdf go chapter 2` — jump to chapter
- `pdf read` — generate audio for current page
- `pdf speed 1.5` — set reading speed

## Project Structure

```
.
├── app.py              # Streamlit PDF Reader
├── piper_ui.py         # FastAPI Piper TTS server
├── docker-compose.yml  # Multi-container orchestration
├── Dockerfile          # Multi-arch container build
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Piper TTS control panel UI
├── data/               # Voice models (gitignored)
└── models/             # Extra models (gitignored)
```

### PIPER TTS SET UP VOICE MODEL

Piper TTS set up once your docker container is running go to http://localhost:5000/ui to download Voice model for Piper TTS

<img width="2514" height="1370" alt="Piper TTS set up 1" src="https://github.com/user-attachments/assets/eb1e39b6-5f9d-4be3-8e0d-4e16f7e763c4" />

<img width="2498" height="1376" alt="Piper TTS set up" src="https://github.com/user-attachments/assets/841eb869-897f-411b-a881-20a8be755dc7" />

## License

MIT
