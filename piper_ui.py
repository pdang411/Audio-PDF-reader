"""Piper TTS Web UI - Run directly"""

import io
import json
import logging
import os
import wave
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import VOICES_JSON, download_voice

from contextlib import asynccontextmanager

_LOGGER = logging.getLogger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global default_voice
    logging.basicConfig(level=logging.INFO)

    download_dir = Path(DATA_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)

    model_path = download_dir / f"{MODEL}.onnx"

    if not model_path.exists():
        _LOGGER.info(f"Downloading voice model: {MODEL}")
        try:
            download_voice(MODEL, download_dir)
        except Exception as e:
            _LOGGER.error(f"Failed to download voice model: {e}")
            _LOGGER.warning(
                "Server starting without default voice. Use /download to retry."
            )

    if model_path.exists():
        try:
            default_voice = PiperVoice.load(model_path, use_cuda=USE_CUDA)
            loaded_voices[MODEL] = default_voice
            _LOGGER.info(f"Loaded voice: {MODEL}")
        except Exception as e:
            _LOGGER.error(f"Failed to load voice model: {e}")
            default_voice = None
    else:
        _LOGGER.warning(f"Model file not found: {model_path}")
        default_voice = None

    yield


app = FastAPI(title="Piper TTS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

MODEL = os.getenv("MODEL", "en_GB-cori-medium")
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
USE_CUDA = os.getenv("USE_CUDA", "false").lower() == "true"

loaded_voices: Dict[str, PiperVoice] = {}
default_voice: Optional[PiperVoice] = None


class DownloadRequest(BaseModel):
    voice: str
    force_redownload: bool = False


class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speaker: Optional[str] = None
    speaker_id: Optional[int] = None
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w_scale: Optional[float] = None


@app.get("/")
async def root():
    return {"message": "Piper TTS API", "model": MODEL, "ui": "/ui"}


@app.get("/health")
async def health():
    return {
        "status": "ok" if default_voice is not None else "degraded",
        "message": "Ready"
        if default_voice is not None
        else "No voice model loaded - use /download to download a voice",
        "model": MODEL,
        "default_voice_ready": default_voice is not None,
        "loaded_voices": list(loaded_voices.keys()),
    }


@app.get("/ui", response_class=HTMLResponse)
async def webui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/voices")
async def list_voices() -> Dict[str, Any]:
    voices_dict: Dict[str, Any] = {}

    for onnx_path in Path(DATA_DIR).glob("*.onnx"):
        config_path = Path(f"{onnx_path}.onnx.json")
        if config_path.exists():
            model_id = onnx_path.stem
            with open(config_path, encoding="utf-8") as config_file:
                voices_dict[model_id] = json.load(config_file)

    return voices_dict


@app.get("/all-voices")
async def list_all_voices():
    try:
        with urlopen(VOICES_JSON, timeout=30) as response:
            return json.load(response)
    except Exception as e:
        _LOGGER.warning(f"Failed to fetch voices list: {e}")
        return {}


@app.post("/download")
async def download_voice_endpoint(req: DownloadRequest):
    global MODEL, default_voice
    download_dir = Path(DATA_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)

    voice_name = req.voice.strip('"').strip("'")

    try:
        download_voice(voice_name, download_dir, force_redownload=req.force_redownload)

        model_path = download_dir / f"{voice_name}.onnx"
        if model_path.exists():
            try:
                voice = PiperVoice.load(model_path, use_cuda=USE_CUDA)
                loaded_voices[voice_name] = voice
                MODEL = voice_name

                if default_voice is None:
                    default_voice = voice

                return {
                    "status": "success",
                    "voice": voice_name,
                    "message": "Voice downloaded and loaded successfully",
                }
            except Exception as e:
                return {
                    "status": "loaded",
                    "voice": voice_name,
                    "message": f"Voice downloaded but failed to load: {e}",
                }
        else:
            raise HTTPException(
                status_code=500, detail="Download failed - model file not created"
            )

    except Exception as e:
        _LOGGER.error(f"Voice download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/synthesize")
async def synthesize_get(
    text: str,
    voice: Optional[str] = None,
    speaker: Optional[str] = None,
    speaker_id: Optional[int] = None,
    length_scale: Optional[float] = None,
    noise_scale: Optional[float] = None,
    noise_w_scale: Optional[float] = None,
) -> Response:
    req = SynthesizeRequest(
        text=text,
        voice=voice,
        speaker=speaker,
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
    )
    return await synthesize(req)


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest) -> Response:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    model_id = (req.voice if req.voice else MODEL).strip('"').strip("'")

    voice = loaded_voices.get(model_id)

    if voice is None:
        model_path = Path(DATA_DIR) / f"{model_id}.onnx"
        if model_path.exists():
            try:
                voice = PiperVoice.load(model_path, use_cuda=USE_CUDA)
                loaded_voices[model_id] = voice
            except Exception as e:
                _LOGGER.error(f"Failed to load voice {model_id}: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to load voice model: {str(e)}"
                )

    if voice is None:
        voice = default_voice

    if voice is None:
        raise HTTPException(
            status_code=503,
            detail="No voice model available. Please download a voice model using the /ui interface or POST to /download",
        )

    speaker_id = req.speaker_id
    if voice.config.num_speakers > 1 and speaker_id is None:
        if req.speaker:
            speaker_id = voice.config.speaker_id_map.get(req.speaker)
        if speaker_id is None:
            speaker_id = 0

    if speaker_id is not None and speaker_id >= voice.config.num_speakers:
        speaker_id = 0

    syn_config = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=req.length_scale or voice.config.length_scale,
        noise_scale=req.noise_scale or voice.config.noise_scale,
        noise_w_scale=req.noise_w_scale or voice.config.noise_w_scale,
    )

    with io.BytesIO() as wav_io:
        wav_file = wave.open(wav_io, "wb")
        with wav_file:
            wav_params_set = False
            for i, audio_chunk in enumerate(voice.synthesize(req.text, syn_config)):
                if not wav_params_set:
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)
                    wav_params_set = True
                wav_file.writeframes(audio_chunk.audio_int16_bytes)

        return Response(content=wav_io.getvalue(), media_type="audio/wav")


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def audio_speech(req: SynthesizeRequest) -> Response:
    return await synthesize(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=300)
