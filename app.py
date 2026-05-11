import re
import struct
from io import BytesIO
import os
import html
import requests
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment
from pydub.utils import which

import PyPDF2
import fitz  # PyMuPDF

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pygame
except Exception:
    pygame = None

try:
    import pyaudio
except Exception:
    pyaudio = None

import tempfile
import datetime
import base64
import uuid
import io

TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:5000")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "10000"))
PDF_READ_CHUNK_SIZE = int(os.getenv("PDF_READ_CHUNK_SIZE", str(1024 * 1024)))


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 22050) -> bytes:
    data_size = len(pcm_data)
    header = bytearray(44)
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, 36 + data_size)
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    struct.pack_into("<I", header, 16, 16)
    struct.pack_into("<H", header, 20, 1)
    struct.pack_into("<H", header, 22, 1)
    struct.pack_into("<I", header, 24, sample_rate)
    struct.pack_into("<I", header, 28, sample_rate * 2)
    struct.pack_into("<H", header, 32, 2)
    struct.pack_into("<H", header, 34, 16)
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, data_size)
    return bytes(header) + pcm_data


class PiperHTTPClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def synthesize(
        self,
        text: str,
        voice: str = None,
        speaker_id: int = None,
        length_scale: float = None,
        noise_scale: float = None,
        noise_w_scale: float = None,
    ) -> bytes:
        url = f"{self.base_url}/synthesize"
        payload = {"text": text, "response_format": "pcm"}
        if voice:
            payload["voice"] = voice
        if speaker_id is not None:
            payload["speaker_id"] = speaker_id
        if length_scale is not None:
            payload["length_scale"] = length_scale
        if noise_scale is not None:
            payload["noise_scale"] = noise_scale
        if noise_w_scale is not None:
            payload["noise_w_scale"] = noise_w_scale
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        sr = int(resp.headers.get("X-Piper-Sample-Rate", 22050))
        return pcm_to_wav(resp.content, sr)


_piper_client = None


def get_piper_client():
    global _piper_client
    if _piper_client is None:
        try:
            _piper_client = PiperHTTPClient(TTS_BASE_URL, timeout=60)
        except Exception as e:
            print(f"Failed to initialize Piper HTTP client: {e}")
            _piper_client = False
    return _piper_client if _piper_client else None


def time_now():
    return datetime.datetime.now().strftime("%I:%M %p")


def chat_lm(prompt: str):
    return "(LLM disabled in this application)"


def speak(text: str):
    client = get_piper_client()
    if not client:
        print("Piper TTS not available")
        return

    try:
        audio_bytes = client.synthesize(text)
        if audio_bytes:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            with open(tmp.name, "wb") as f:
                f.write(audio_bytes)
            if pygame:
                try:
                    pygame.mixer.init()
                    pygame.mixer.music.load(tmp.name)
                    pygame.mixer.music.play()
                    print(text)
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    pygame.mixer.quit()
                    try:
                        os.remove(tmp.name)
                    except Exception:
                        pass
                    return
                except Exception as e:
                    print(f"pygame playback failed: {e}")
            print(f"Audio saved to: {tmp.name}")
    except Exception as e:
        print(f"Piper TTS failed: {e}")


def listen(timeout: float = 5.0) -> str:
    raise RuntimeError(
        "Audio recording requires browser microphone access. Use the record button in the UI."
    )


class LLMHTTPClient:
    def __init__(
        self, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL, timeout: int = 60
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, stream: bool = False) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": stream}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}")


_llm_client = None


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        try:
            _llm_client = LLMHTTPClient()
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")
            _llm_client = False
    return _llm_client if _llm_client else None


from functools import lru_cache


# --- OCR Support for scanned PDFs ---
def ocr_image_to_text(image_bytes: bytes) -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        if TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""


def ocr_pdf_page(pdf_bytes: bytes, page_index: int, zoom: float = 2.0) -> str:
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            p = d.load_page(page_index)
            mat = fitz.Matrix(zoom, zoom)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
        finally:
            d.close()
        return ocr_image_to_text(img_bytes)
    except Exception as e:
        print(f"PDF page OCR failed: {e}")
        return ""


def ocr_pdf_file(pdf_bytes: bytes, zoom: float = 2.0) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_count = len(doc)
            if page_count == 0:
                return ""
            results = []
            for i in range(page_count):
                page = doc.load_page(i)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                text = ocr_image_to_text(img_bytes)
                results.append(text)
        finally:
            doc.close()
        return "\n\n".join(results)
    except Exception as e:
        print(f"PDF file OCR failed: {e}")
        return ""


st.set_page_config(layout="wide")
try:
    st.set_option("server.maxUploadSize", 500)
except Exception:
    pass

st.title("PDF Page-by-Page Audiobook Reader")


def _set_ffmpeg():
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            AudioSegment.converter = exe
            return
    except Exception:
        pass

    env_ff = os.environ.get("FFMPEG_PATH")
    if env_ff and os.path.isfile(env_ff):
        AudioSegment.converter = env_ff
        return

    sys_ff = which("ffmpeg") or which("avconv")
    if sys_ff:
        AudioSegment.converter = sys_ff
        return

    candidates = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.join(os.getcwd(), ".venv", "Scripts", "ffmpeg.exe"),
        os.path.join(os.getcwd(), ".venv", "Library", "bin", "ffmpeg.exe"),
    ]
    for c in candidates:
        try:
            if os.path.isfile(c):
                AudioSegment.converter = c
                return
        except Exception:
            continue
    return


_set_ffmpeg()

_conv = getattr(AudioSegment, "converter", None)
if not (_conv and os.path.isfile(_conv)) and not (which("ffmpeg") or which("avconv")):
    st.sidebar.warning(
        "ffmpeg/avconv not found. Audio functions may fail. "
        "Install ffmpeg and add to PATH, or set FFMPEG_PATH to full ffmpeg.exe path."
    )

if pytesseract is None:
    st.sidebar.warning(
        "Tesseract OCR not installed. For scanned PDFs, install Tesseract and add to PATH. "
        "Download: https://github.com/UB-Mannheim/tesseract/wiki"
    )
elif TESSERACT_PATH and not os.path.isfile(TESSERACT_PATH):
    st.sidebar.warning(
        f"Tesseract OCR not found at: {TESSERACT_PATH}. "
        "Set TESSERACT_PATH environment variable to the tesseract.exe path."
    )


@st.cache_data(show_spinner=False)
def load_pdf_texts(pdf_bytes: bytes):
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        try:
            for i in range(d.page_count):
                p = d.load_page(i)
                texts.append(p.get_text("text") or "")
        finally:
            d.close()
        return texts
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return texts
        except Exception as e2:
            print(f"PyPDF2 also failed: {e2}")
            return [""]


@st.cache_data(show_spinner=False)
def get_pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return d.page_count
        finally:
            d.close()
    except Exception as e:
        print(f"PyMuPDF page count failed: {e}")
        try:
            return len(PyPDF2.PdfReader(BytesIO(pdf_bytes)).pages)
        except Exception:
            return 0


@st.cache_data(show_spinner=False)
def get_pdf_bookmarks(pdf_bytes: bytes) -> list:
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            toc = d.get_toc()
            bookmarks = []
            for item in toc:
                level, title, page_num = item[:3]
                bookmarks.append({"title": title.strip(), "page": page_num - 1})
            return bookmarks
        finally:
            d.close()
    except Exception as e:
        print(f"PDF bookmark extraction failed: {e}")
        return []


@st.cache_data(show_spinner=False)
def get_pdf_page_text(pdf_bytes: bytes, page_index: int, use_ocr: bool = False) -> str:
    text = ""
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            p = d.load_page(page_index)
            text = p.get_text("text") or ""
        finally:
            d.close()
    except Exception as e:
        print(f"PyMuPDF page text failed: {e}")
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            if 0 <= page_index < len(reader.pages):
                text = reader.pages[page_index].extract_text() or ""
        except Exception:
            pass

    if use_ocr and pytesseract is not None:
        ocr_text = ocr_pdf_page(pdf_bytes, page_index, zoom=2.0)
        if ocr_text.strip():
            text = ocr_text

    return text


@st.cache_data(show_spinner=False)
def cached_page_image(pdf_bytes: bytes, page_index: int, zoom: float = 1.6):
    try:
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            p = d.load_page(page_index)
            mat = fitz.Matrix(zoom, zoom)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            return pix.tobytes("png")
        finally:
            d.close()
    except Exception as e:
        print(f"PDF page image rendering failed: {e}")
        return None


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', text)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'(?m)^#{1,6}\s+', '', text)
    text = re.sub(r'(?m)^>\s+', '', text)
    text = re.sub(r'(?m)^[\*\-\+]\s+', '', text)
    text = re.sub(r'(?m)^\d+\.\s+', '', text)
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _split_text(text: str, max_len: int) -> list[str]:
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        cut = text.rfind(". ", 0, max_len)
        if cut == -1:
            cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = text.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len
        part = text[:cut].strip()
        if part:
            parts.append(part)
        text = text[cut:].lstrip()
    return parts


@st.cache_data(show_spinner=False, max_entries=20)
def _synthesize_cached(clean_text: str) -> bytes:
    if not clean_text or not clean_text.strip():
        return b""

    client = get_piper_client()
    if not client:
        raise RuntimeError("Piper TTS not available")

    if len(clean_text) <= TTS_MAX_CHARS:
        return client.synthesize(clean_text)

    chunks = _split_text(clean_text, TTS_MAX_CHARS)
    if not chunks:
        return b""

    pcm_parts = []
    sample_rate = 22050
    first = True
    for chunk in chunks:
        if not chunk.strip():
            continue
        wav_bytes = client.synthesize(chunk)
        if not wav_bytes or len(wav_bytes) <= 44:
            continue
        if first:
            sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
            first = False
        pcm_parts.append(wav_bytes[44:])

    if not pcm_parts:
        return b""

    pcm_data = b"".join(pcm_parts)
    return pcm_to_wav(pcm_data, sample_rate)


def convert_text_to_audio(text: str, rate: float = 1.0):
    if not text or not str(text).strip():
        return None

    clean_text = clean_text_for_tts(str(text))

    try:
        r = float(rate)
    except Exception:
        r = 1.0

    audio_bytes = _synthesize_cached(clean_text)
    if not audio_bytes:
        return None

    if abs(r - 1.0) < 1e-6:
        return BytesIO(audio_bytes)

    try:
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
        audio = audio._spawn(
            audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * r)}
        )
        audio = audio.set_frame_rate(44100)
        out = BytesIO()
        audio.export(out, format="wav")
        out.seek(0)
        return out
    except Exception:
        return BytesIO(audio_bytes)


def build_sync_player_html(
    audio_bytes: bytes, text: str, element_id: str | None = None
):
    if element_id is None:
        element_id = f"syncplayer_{uuid.uuid4().hex}"

    raw = text.replace("\r\n", "\n")
    lines = [ln for ln in raw.split("\n")]
    words = []
    for ln in lines:
        for w in ln.split():
            words.append(w)

    if not words:
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return f'<audio controls src="data:audio/wav;base64,{b64}"></audio>'

    total_chars = sum(len(w) for w in words)
    try:
        from pydub import AudioSegment as _AS

        seg = _AS.from_file(BytesIO(audio_bytes), format="wav")
        total_ms = len(seg)
    except Exception:
        total_ms = max(1000, int(total_chars * 30))

    per_word_ms = [int(len(w) / total_chars * total_ms) for w in words]
    diff = total_ms - sum(per_word_ms)
    i = 0
    while diff > 0:
        per_word_ms[i % len(per_word_ms)] += 1
        diff -= 1
        i += 1

    timestamps = []
    cum = 0
    for ms in per_word_ms:
        timestamps.append(round(cum / 1000.0, 3))
        cum += ms

    out_lines = []
    widx = 0
    for ln in lines:
        parts = []
        for w in ln.split():
            safe = html.escape(w)
            parts.append(f'<span class="word" data-idx="{widx}">{safe}</span>')
            widx += 1
        out_lines.append(" ".join(parts))
    body_html = "<br/>".join(out_lines)

    b64 = base64.b64encode(audio_bytes).decode("ascii")

    js = f'''<style>
.word {{ color: #222; }}
.word.active {{ background: #fffa8b; color: #000; }}
</style>
<div id="{element_id}_container">
    <audio id="{element_id}_audio" controls src="data:audio/wav;base64,{b64}"></audio>
  <div id="{element_id}_text" style="margin-top:12px; font-size:16px; line-height:1.5;">{body_html}</div>
</div>
<script>
(function(){{
  const audio = document.getElementById('{element_id}_audio');
  const words = Array.from(document.querySelectorAll('#{element_id}_text .word'));
  const timestamps = {timestamps};
  function highlight(idx){{
    words.forEach((w,i)=>{{w.classList.toggle('active', i===idx);}});
  }}
  audio.addEventListener('timeupdate', ()=>{{
    const t = audio.currentTime;
    let idx = 0;
    for(let i=0;i<timestamps.length;i++){{ if(t >= timestamps[i]) idx = i; else break; }}
    highlight(idx);
  }});
}})();
</script>'''

    return js


def parse_voice_command(cmd: str) -> dict:
    cmd = cmd.lower().strip()

    if not cmd.startswith("pdf"):
        return {"type": "unknown", "text": cmd}

    cmd = re.sub(r"^pdf\s+", "", cmd)

    if (m := re.search(r"(?:go\s+)?chapter\s*(\d+)", cmd)):
        return {"type": "chapter_num", "value": int(m.group(1))}

    if (m := re.search(r"page\s*(\d+)", cmd)):
        return {"type": "page", "value": int(m.group(1))}

    if (m := re.search(r"go\s+to\s+(\d+)", cmd)):
        return {"type": "page", "value": int(m.group(1))}

    if re.search(r"(?:next|forward)\s+chapter", cmd):
        return {"type": "next_chapter"}

    if re.search(r"(?:prev|previous|back)\s+chapter", cmd):
        return {"type": "prev_chapter"}

    if "chapter" in cmd or re.search(r"go\s+to\s+[a-z]", cmd):
        return {"type": "chapter_name", "text": cmd}

    if re.search(r"next|forward|turn", cmd):
        return {"type": "next"}

    if re.search(r"prev|previous|back", cmd):
        return {"type": "prev"}

    if (m := re.search(r"speed\s*([\d.]+)", cmd)):
        return {"type": "speed", "value": float(m.group(1))}

    if re.search(r"read|play|speak", cmd):
        return {"type": "read"}

    return {"type": "unknown", "text": cmd}


def exec_voice_command(cmd: str, pdf_bytes, total_pages, enable_ocr, rate, sync_highlight):
    parsed = parse_voice_command(cmd)
    t = parsed["type"]

    if t == "next":
        if st.session_state["page_index"] < total_pages - 1:
            st.session_state["page_index"] += 1
            st.rerun()

    elif t == "prev":
        if st.session_state["page_index"] > 0:
            st.session_state["page_index"] -= 1
            st.rerun()

    elif t == "page":
        p = parsed["value"] - 1
        if 0 <= p < total_pages:
            st.session_state["page_index"] = p
            st.rerun()

    elif t == "chapter_num":
        bookmarks = get_pdf_bookmarks(pdf_bytes)
        if bookmarks:
            ch = parsed["value"] - 1
            if 0 <= ch < len(bookmarks):
                st.session_state["page_index"] = bookmarks[ch]["page"]
                st.sidebar.success(f"Went to chapter {ch+1}: {bookmarks[ch]['title']}")
                st.rerun()
            else:
                st.sidebar.info(f"Chapter {parsed['value']} not found. Max: {len(bookmarks)}")
        else:
            st.sidebar.info("No chapters in this PDF")

    elif t == "next_chapter":
        bookmarks = get_pdf_bookmarks(pdf_bytes)
        if bookmarks:
            for bm in bookmarks:
                if bm["page"] > st.session_state["page_index"]:
                    st.session_state["page_index"] = bm["page"]
                    st.sidebar.success(f"Next chapter: {bm['title']}")
                    st.rerun()
                    break

    elif t == "prev_chapter":
        bookmarks = get_pdf_bookmarks(pdf_bytes)
        if bookmarks:
            for bm in reversed(bookmarks):
                if bm["page"] < st.session_state["page_index"]:
                    st.session_state["page_index"] = bm["page"]
                    st.sidebar.success(f"Previous chapter: {bm['title']}")
                    st.rerun()
                    break

    elif t == "chapter_name":
        bookmarks = get_pdf_bookmarks(pdf_bytes)
        if bookmarks:
            for bm in bookmarks:
                if bm["title"].lower() in parsed.get("text", ""):
                    st.session_state["page_index"] = bm["page"]
                    st.sidebar.success(f"Went to: {bm['title']}")
                    st.rerun()
                    break
            else:
                names = ", ".join(b['title'][:30] for b in bookmarks[:5])
                st.sidebar.info(f"Chapters: {names}")
        else:
            st.sidebar.info("No chapters in this PDF")

    elif t == "speed":
        st.session_state["reading_rate"] = parsed["value"]
        st.sidebar.success(f"Reading speed set to {parsed['value']}")

    elif t == "read":
        page_text = (get_pdf_page_text(pdf_bytes, st.session_state["page_index"], use_ocr=enable_ocr) or "")
        if page_text.strip():
            with st.spinner("Generating audio for current page..."):
                try:
                    buf = convert_text_to_audio(page_text, st.session_state.get("reading_rate", rate))
                    if not buf:
                        st.sidebar.warning("No text found on this page to generate audio.")
                    else:
                        st.session_state["last_audio_bytes"] = buf.getvalue()
                    st.sidebar.audio(st.session_state["last_audio_bytes"], format="audio/wav", autoplay=True)
                    try:
                        if sync_highlight:
                            components.html(build_sync_player_html(st.session_state["last_audio_bytes"], page_text), height=260)
                    except Exception:
                        pass
                except Exception as e:
                    st.sidebar.error(f"Audio generation failed: {e}")

    elif t == "unknown":
        st.sidebar.info(f"Sending to LLM: '{cmd}'")
        with st.spinner("LLM processing..."):
            try:
                llm = get_llm_client()
                if llm:
                    reply = llm.generate(cmd)
                    st.sidebar.success(f"LLM: {reply}")
                    with st.spinner("Generating audio..."):
                        wav = convert_text_to_audio(reply, rate=rate)
                        if wav:
                            st.sidebar.audio(wav, format="audio/wav", autoplay=True)
                            if sync_highlight:
                                try:
                                    components.html(build_sync_player_html(wav.getvalue(), reply), height=260)
                                except Exception:
                                    pass
                else:
                    st.sidebar.error("LLM not available")
            except Exception as e:
                st.sidebar.error(f"LLM error: {e}")


# --- UI: Sidebar uploader / speed controls ---
use_local_file = st.sidebar.checkbox("Load PDF from server (faster than upload)")
local_pdf_choice = None
if use_local_file:
    import glob

    pdf_files = glob.glob(os.path.join(os.getcwd(), "*.pdf"))
    if not pdf_files:
        st.sidebar.info(
            "No PDF files found in project root. Place your PDF in the project folder."
        )
    else:
        names = [os.path.basename(p) for p in pdf_files]
        sel = st.sidebar.selectbox("Choose PDF from server", names)
        local_pdf_choice = os.path.join(os.getcwd(), sel)
        st.sidebar.caption(f"Selected: {local_pdf_choice}")

uploaded_pdf = (
    None if use_local_file else st.sidebar.file_uploader("Upload PDF", type=["pdf"])
)

sidebar_rate = st.sidebar.slider(
    "Reading speed (1.0 = normal)", 0.5, 2.0, 1.0, 0.1, key="sidebar_rate"
)
st.session_state["reading_rate"] = float(
    st.session_state.get("sidebar_rate", sidebar_rate)
)
rate = st.session_state.get("reading_rate", float(sidebar_rate))
st.sidebar.write(f"Current speed: {rate}")
sync_highlight = st.sidebar.checkbox(
    "Enable sync highlighting (word-by-word)", value=True
)

enable_ocr = False
if uploaded_pdf or local_pdf_choice:

    if local_pdf_choice:
        path = local_pdf_choice
        try:
            total_size = os.path.getsize(path)
            read_bytes = BytesIO()
            chunk = PDF_READ_CHUNK_SIZE
            with open(path, "rb") as fh:
                p = st.sidebar.progress(0)
                read = 0
                while True:
                    data = fh.read(chunk)
                    if not data:
                        break
                    read_bytes.write(data)
                    read += len(data)
                    p.progress(min(100, int(read / total_size * 100)))
            pdf_bytes = read_bytes.getvalue()
        except Exception as e:
            pdf_bytes = b""
    else:
        try:
            total_size = getattr(uploaded_pdf, "size", None)
        except Exception:
            total_size = None
        read_bytes = BytesIO()
        chunk = PDF_READ_CHUNK_SIZE
        p = None
        if total_size:
            p = st.sidebar.progress(0)
        read = 0
        while True:
            data = uploaded_pdf.read(chunk)
            if not data:
                break
            read_bytes.write(data)
            read += len(data)
            if p and total_size:
                p.progress(min(100, int(read / total_size * 100)))
        pdf_bytes = read_bytes.getvalue()

    if not pdf_bytes:
        st.sidebar.error("No PDF data received. Please try uploading a valid PDF file.")
        st.info("Upload a PDF in the sidebar to begin.")
        st.stop()

    total_pages = get_pdf_page_count(pdf_bytes)

    st.sidebar.write(f"Pages: {total_pages}")

    show_diag = st.sidebar.checkbox("Show diagnostics")
    if show_diag:
        st.sidebar.write(f"pdf_bytes size: {len(pdf_bytes)} bytes")
        try:
            d_check = fitz.open(stream=pdf_bytes, filetype="pdf")
            raw_count = d_check.page_count
            d_check.close()
        except Exception:
            try:
                raw_count = len(PyPDF2.PdfReader(BytesIO(pdf_bytes)).pages)
            except Exception:
                raw_count = "(error)"
        st.sidebar.write(f"Reader page count: {raw_count}")
        per_page = []
        limit = min(total_pages, 20)
        for i in range(limit):
            try:
                t = get_pdf_page_text(pdf_bytes, i, use_ocr=enable_ocr)
                per_page.append(f"p{i + 1}: {len(t)} chars")
            except Exception:
                per_page.append(f"p{i + 1}: (error)")
        if total_pages > limit:
            per_page.append(f"... +{total_pages - limit} more pages")
        st.sidebar.write("Page lengths:")
        st.sidebar.write(per_page)
        st.sidebar.write(f"session page_index: {st.session_state.get('page_index')}")

    poppler_path = None

    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0
    if st.session_state["page_index"] >= total_pages:
        st.session_state["page_index"] = max(0, total_pages - 1)

    audio_container = st.empty()

    st.sidebar.subheader("Navigation")

    page_input = st.sidebar.number_input(
        "Page number",
        min_value=1,
        max_value=max(1, total_pages),
        value=st.session_state["page_index"] + 1,
        step=1,
        key="page_input",
    )
    if page_input - 1 != st.session_state["page_index"]:
        st.session_state["page_index"] = page_input - 1

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.sidebar.button("Prev", key="nav_prev"):
            if st.session_state["page_index"] > 0:
                st.session_state["page_index"] -= 1
    with col2:
        if st.sidebar.button("Next", key="nav_next"):
            if st.session_state["page_index"] < total_pages - 1:
                st.session_state["page_index"] += 1

    colf, coll = st.sidebar.columns([1, 1])
    with colf:
        if st.sidebar.button("First", key="nav_first"):
            st.session_state["page_index"] = 0
    with coll:
        if st.sidebar.button("Last", key="nav_last"):
            st.session_state["page_index"] = max(0, total_pages - 1)

    enable_ocr = st.sidebar.toggle(
        "Enable OCR for scanned PDFs",
        value=False,
        help="Turn on to extract text from scanned/image PDFs",
    )

    # --- Voice assistant (Web Speech API via browser) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Voice Assistant")
    st.sidebar.write("Use your microphone to give a voice command for the reader.")
    st.sidebar.caption(
        "Try: 'pdf next page', 'pdf back', 'pdf page 5', 'pdf go chapter 1', 'pdf read', 'pdf speed 1.5'"
    )

    st.sidebar.html(
        """
        <style>
            #voice-btn {
                padding: 8px 16px; font-size: 16px; cursor: pointer;
                border-radius: 4px; border: 1px solid #ccc; background: #4CAF50; color: #fff;
                width: 100%;
            }
            #voice-btn:hover { background: #45a049; }
            #voice-btn:disabled { opacity: 0.6; cursor: not-allowed; }
            #voice-status { margin-top: 5px; font-size: 13px; min-height: 20px; }
        </style>
        <button id="voice-btn">Start Voice Command</button>
        <div id="voice-status"></div>
        <script>
        (function() {
            var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            var btn = document.getElementById('voice-btn');
            var status = document.getElementById('voice-status');
            if (!SR) {
                status.textContent = 'Not supported. Try Chrome.';
                btn.disabled = true;
                return;
            }
            btn.addEventListener('click', function() {
                if (btn.disabled) return;
                btn.disabled = true;
                btn.textContent = 'Listening...';
                status.textContent = 'Speak now...';
                var r = new SR();
                r.lang = 'en-US';
                r.continuous = false;
                r.interimResults = false;
                var timeout = setTimeout(function() {
                    try { r.stop(); } catch(e) {}
                    status.textContent = 'Timed out. Try again.';
                    btn.disabled = false;
                    btn.textContent = 'Start Voice Command';
                }, 10000);
                r.onresult = function(e) {
                    clearTimeout(timeout);
                    var text = e.results[0][0].transcript;
                    status.textContent = 'Done!';
                    var url = new URL(window.location);
                    url.searchParams.set('voice', text);
                    window.location.search = url.search;
                };
                r.onerror = function(e) {
                    clearTimeout(timeout);
                    status.textContent = 'Error: ' + e.error;
                    btn.disabled = false;
                    btn.textContent = 'Start Voice Command';
                };
                r.onend = function() {
                    clearTimeout(timeout);
                    btn.disabled = false;
                    btn.textContent = 'Start Voice Command';
                };
                try { r.start(); } catch(err) {
                    status.textContent = 'Error: ' + err.message;
                    btn.disabled = false;
                    btn.textContent = 'Start Voice Command';
                    clearTimeout(timeout);
                }
            });
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )

    recognized = st.query_params.get("voice")

    if recognized and isinstance(recognized, str) and recognized.strip() and (uploaded_pdf or local_pdf_choice):
        cmd = recognized.strip().lower()
        st.sidebar.info(f"Voice: '{cmd}'")
        st.query_params.clear()
        exec_voice_command(cmd, pdf_bytes, total_pages, enable_ocr, rate, sync_highlight)

    st.sidebar.subheader("Select Page")

    page_index = st.session_state["page_index"]
    page_image = None
    try:
        page_image = cached_page_image(pdf_bytes, page_index, zoom=1.6)
    except Exception:
        page_image = None

    if page_image is not None:
        st.image(page_image, caption=f"Page {page_index + 1}")
    else:
        st.info("Page image unavailable (render failed). Showing text-only view.")

    page_text = get_pdf_page_text(pdf_bytes, page_index, use_ocr=enable_ocr) or ""
    escaped = html.escape(page_text)
    html_content = f"""
    <div style='width:1200px; height:1200px; overflow:auto; border:1px solid #ddd; padding:12px; background:#fff;'>
      <pre style='white-space:pre-wrap; font-family:inherit; font-size:16px; line-height:1.5; margin:0;'>
{escaped}
      </pre>
    </div>
    """
    components.html(html_content, height=1200)

    if "last_audio_bytes" in st.session_state and st.session_state["last_audio_bytes"]:
        try:
            st.audio(st.session_state["last_audio_bytes"], format="audio/wav")
        except Exception:
            pass

    audio_buffer = None
    if st.sidebar.button("Start Reading", key="start_reading"):
        with st.spinner("Generating audio for this page..."):
            try:
                audio_buffer = convert_text_to_audio(page_text, rate)
            except Exception as e:
                st.sidebar.error(f"Audio generation failed: {e}")
                audio_buffer = None

    if audio_buffer is not None:
        audio_container.audio(audio_buffer, format="audio/wav", autoplay=True)
        try:
            if sync_highlight:
                html_player = build_sync_player_html(audio_buffer.getvalue(), page_text)
                components.html(html_player, height=260)
        except Exception:
            pass
        st.sidebar.download_button(
            "Download This Page as WAV",
            data=audio_buffer,
            file_name=f"page_{page_index + 1}.wav",
            mime="audio/wav",
            key=f"download_play_{page_index}_{uuid.uuid4().hex}",
        )

    if st.sidebar.button("Generate (no play)", key="gen_no_play"):
        with st.spinner("Generating audio for this page (no play)..."):
            try:
                audio_buffer = convert_text_to_audio(page_text, rate)
            except Exception as e:
                st.sidebar.error(f"Audio generation failed: {e}")
                audio_buffer = None

    if audio_buffer is not None:
        st.sidebar.download_button(
            "Download This Page as WAV",
            data=audio_buffer,
            file_name=f"page_{page_index + 1}.wav",
            mime="audio/wav",
            key=f"download_noplay_{page_index}_{uuid.uuid4().hex}",
        )
else:
    st.info("Upload a PDF in the sidebar to begin.")
