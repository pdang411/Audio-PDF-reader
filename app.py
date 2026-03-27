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

# --- Speech / TTS / local LM imports (embedded per user request) ---
# Ollama integration removed to simplify local LM handling.

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

# Local LLM integrations removed: this app runs without access to external LLMs.
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://piper:5000")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.1.19:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "2000"))
PDF_READ_CHUNK_SIZE = int(os.getenv("PDF_READ_CHUNK_SIZE", str(1024 * 1024)))


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
        payload = {"text": text}
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
        return resp.content


_piper_client = None


def get_piper_client():
    """Lazily create Piper client only when needed."""
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
    """Browser microphone is used via Web Speech API in the UI."""
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


# Use caching for expensive operations
from functools import lru_cache


# --- OCR Support for scanned PDFs ---
def ocr_image_to_text(image_bytes: bytes) -> str:
    """Extract text from an image using Tesseract OCR."""
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
    """Render a PDF page to image and extract text via OCR."""
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
    """OCR all pages of a PDF file and return combined text."""
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


# --- Page layout & upload limits ---
st.set_page_config(layout="wide")
try:
    st.set_option("server.maxUploadSize", 500)  # allow up to 500 MB uploads
except Exception:
    pass

st.title("📚 PDF Page-by-Page Audiobook Reader")


# --- Simplified, robust ffmpeg detection for pydub (why: pydub needs AudioSegment.converter) ---
def _set_ffmpeg():
    """Set AudioSegment.converter with a small fallback chain."""
    # 1) Prefer imageio-ffmpeg bundled exe (if available)
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            AudioSegment.converter = exe
            return
    except Exception:
        pass

    # 2) Respect explicit env var
    env_ff = os.environ.get("FFMPEG_PATH")
    if env_ff and os.path.isfile(env_ff):
        AudioSegment.converter = env_ff
        return

    # 3) Try system ffmpeg/avconv
    sys_ff = which("ffmpeg") or which("avconv")
    if sys_ff:
        AudioSegment.converter = sys_ff
        return

    # 4) Try some common Windows locations (no hard fail)
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

    # If nothing found, leave AudioSegment.converter unset; we'll warn later.
    return


_set_ffmpeg()

# Sidebar ffmpeg warning if still not set
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


# --- Cache PDF text extraction (fast re-use) ---
@st.cache_data(show_spinner=False)
def load_pdf_texts(pdf_bytes: bytes):
    """Return list of page text extracted from PDF bytes.

    Prefer PyMuPDF (fitz) if available for speed; otherwise fall back to PyPDF2.
    """
    try:
        # fast path: use fitz if present
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
            return [""]  # Return empty list on failure


@st.cache_data(show_spinner=False)
def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Return number of pages in PDF bytes without extracting full text."""
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
    """Extract bookmarks/chapters from PDF. Returns list of (title, page_num)."""
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
    """Extract and return text for a single PDF page (cached). Uses OCR as fallback for scanned pages."""
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


# --- Cache single-page image rendering (lazy per page) using PyMuPDF ---
@st.cache_data(show_spinner=False)
def cached_page_image(pdf_bytes: bytes, page_index: int, zoom: float = 1.6):
    """Render a single PDF page to PNG bytes using PyMuPDF and cache the result."""
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


# --- Audio conversion (using Piper TTS) ---
def convert_text_to_audio(text: str, rate: float = 1.0):
    if not text or not str(text).strip():
        return None

    clean_text = str(text).strip()

    def _synth_chunk(chunk: str) -> bytes:
        if not chunk.strip():
            return b""

        client = get_piper_client()
        if not client:
            raise RuntimeError("Piper TTS not available")

        return client.synthesize(chunk)

    if len(clean_text) <= TTS_MAX_CHARS:
        audio_bytes = _synth_chunk(clean_text)
    else:
        from pydub import AudioSegment as _AS

        def _split_long_text(s: str, max_len: int):
            parts = []
            while s:
                if len(s) <= max_len:
                    parts.append(s)
                    break
                cut = s.rfind(". ", 0, max_len)
                if cut == -1:
                    cut = s.rfind("\n", 0, max_len)
                if cut == -1:
                    cut = s.rfind(" ", 0, max_len)
                if cut == -1:
                    cut = max_len
                part = s[:cut].strip()
                if part:
                    parts.append(part)
                s = s[cut:].lstrip()
            return parts

        chunks = _split_long_text(clean_text, TTS_MAX_CHARS)
        if not chunks:
            return None

        combined_seg = None
        for idx, chunk in enumerate(chunks):
            try:
                bytes_chunk = _synth_chunk(chunk)
            except Exception as exc:
                raise RuntimeError(
                    f"Piper TTS failed on chunk {idx + 1}/{len(chunks)}: {exc}"
                ) from exc

            if not bytes_chunk:
                continue
            seg = _AS.from_file(BytesIO(bytes_chunk), format="wav")
            combined_seg = seg if combined_seg is None else combined_seg + seg

        if combined_seg is None:
            return None

        out_joined = BytesIO()
        combined_seg.export(out_joined, format="wav")
        out_joined.seek(0)
        audio_bytes = out_joined.getvalue()

    wav_buffer = BytesIO(audio_bytes)

    # If rate is 1.0, return original TTS WAV bytes immediately.
    try:
        r = float(rate)
    except Exception:
        r = 1.0
    if abs(r - 1.0) < 1e-6:
        out = BytesIO(wav_buffer.getvalue())
        out.seek(0)
        return out

    # Use pydub for speed adjustment (preserve pitch)
    try:
        audio = AudioSegment.from_file(wav_buffer, format="wav")
        # adjust speed via frame_rate override (why: preserve pitch-ish)
        audio = audio._spawn(
            audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * r)}
        )
        audio = audio.set_frame_rate(44100)
        out_buffer = BytesIO()
        audio.export(out_buffer, format="wav")
        out_buffer.seek(0)
        return out_buffer
    except Exception:
        # As a last resort, return original wav bytes
        out = BytesIO(wav_buffer.getvalue())
        out.seek(0)
        return out


def build_sync_player_html(
    audio_bytes: bytes, text: str, element_id: str | None = None
):
    """Return HTML that embeds audio (base64) and highlights words as audio plays.

    This uses approximate timing proportional to each word's character length.
    """
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
        # try to get accurate duration using pydub
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


# --- UI: Sidebar uploader / speed controls ---
# Option: prefer loading a file already on the server (faster than upload)
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

# live reading speed slider (applies immediately)
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

# --- When PDF uploaded or selected from server: main app flow ---
enable_ocr = False
if uploaded_pdf or local_pdf_choice:
    # Debug info

    if local_pdf_choice:
        # read file from disk with a progress bar (avoid upload time)
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
        # Read uploaded file in chunks to avoid large memory spikes
        while True:
            data = uploaded_pdf.read(chunk)
            if not data:
                break
            read_bytes.write(data)
            read += len(data)
            if p and total_size:
                p.progress(min(100, int(read / total_size * 100)))
        pdf_bytes = read_bytes.getvalue()

    # Check if we have any PDF data
    if not pdf_bytes:
        st.sidebar.error("No PDF data received. Please try uploading a valid PDF file.")
        st.info("Upload a PDF in the sidebar to begin.")
        st.stop()

    # determine page count quickly (cached)
    total_pages = get_pdf_page_count(pdf_bytes)

    # debug: show total pages in sidebar
    st.sidebar.write(f"Pages: {total_pages}")

    # Diagnostic toggle: show detailed page info
    show_diag = st.sidebar.checkbox("Show diagnostics")
    if show_diag:
        st.sidebar.write(f"pdf_bytes size: {len(pdf_bytes)} bytes")
        # report raw reader page count via fitz or PyPDF2
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
        # show per-page text length summary (first 20 pages only to avoid slow work)
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

    # Use PyMuPDF for page rendering; no Poppler required
    poppler_path = None

    # initialize page index if missing and clamp to available pages
    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0
    # ensure page_index is within bounds (handles uploading a different PDF)
    if st.session_state["page_index"] >= total_pages:
        st.session_state["page_index"] = max(0, total_pages - 1)

    # main UI placeholders
    audio_container = st.empty()

    # Sidebar navigation (kept behavior)
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
        if st.sidebar.button("◀ Prev", key="nav_prev"):
            if st.session_state["page_index"] > 0:
                st.session_state["page_index"] -= 1
    with col2:
        if st.sidebar.button("Next ▶", key="nav_next"):
            if st.session_state["page_index"] < total_pages - 1:
                st.session_state["page_index"] += 1

    # First / Last quick navigation
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

    # --- Voice assistant integration (embedded) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Voice Assistant")
    st.sidebar.write("Use your microphone to give a voice command for the reader.")
    st.sidebar.caption(
        "Try commands: 'next', 'previous', 'read page', 'go to page 5', 'go to chapter 1', 'speed 1.5'"
    )

    import streamlit.components.v1 as components

    audio_recorder_html = """
    <style>
    .rec-btn {padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; margin-bottom: 10px;}
    .start-btn {background-color: #4CAF50; color: white;}
    .stop-btn {background-color: #f44336; color: white; display: none;}
    </style>
    <div style="padding: 10px;">
        <button id="startBtn" class="rec-btn start-btn">🎤 Start Voice Command</button>
        <button id="stopBtn" class="rec-btn stop-btn">⏹ Stop</button>
        <p id="status" style="margin: 10px 0; font-size: 14px;">Click to start speaking</p>
        <p id="result" style="margin: 10px 0; font-size: 14px; font-weight: bold; color: #2196F3;"></p>
    </div>
    <script>
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const status = document.getElementById('status');
    const result = document.getElementById('result');
    let recognition;

    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
        recognition = new SpeechRecognition();
    } else {
        status.textContent = "Speech API not supported in this browser";
    }

    if (recognition) {
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
            status.textContent = "Listening... Speak now";
        };

        recognition.onend = function() {
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            result.textContent = "Recognized: " + transcript;
            window.parent.postMessage({type: 'streamlit:setComponentValue', value: transcript}, '*');
            status.textContent = "Done!";
        };

        recognition.onerror = function(event) {
            status.textContent = "Error: " + event.error;
        };

        startBtn.onclick = function() {
            try {
                recognition.start();
            } catch(e) {
                status.textContent = "Error: " + e.message;
            }
        };

        stopBtn.onclick = function() {
            recognition.stop();
        };
    }
    </script>
    """
    recognized = components.html(audio_recorder_html, height=200)

    # Only process voice commands if PDF is loaded
    if (
        recognized
        and isinstance(recognized, str)
        and recognized.strip()
        and (uploaded_pdf or local_pdf_choice)
    ):
        cmd = recognized.strip().lower()

        # Debug: show what was recognized
        st.sidebar.info(f"Voice: '{cmd}'")

        # navigation - next page
        if (
            "next" in cmd
            or "forward" in cmd
            or "page next" in cmd
            or "turn page" in cmd
        ):
            if st.session_state["page_index"] < total_pages - 1:
                st.session_state["page_index"] += 1
                st.rerun()
        # navigation - previous/back page
        elif (
            "prev" in cmd
            or "previous" in cmd
            or "back" in cmd
            or "go back" in cmd
            or "page back" in cmd
        ):
            if st.session_state["page_index"] > 0:
                st.session_state["page_index"] -= 1
                st.rerun()
        # go to page X (e.g., "page 5" or "go to page 5")
        elif "page" in cmd:
            import re

            m = re.search(r"page\s*(\d+)", cmd)
            if m:
                p = int(m.group(1)) - 1
                if 0 <= p < total_pages:
                    st.session_state["page_index"] = p
                    st.rerun()
        # go to chapter
        elif "chapter" in cmd or "go to" in cmd:
            import re

            bookmarks = get_pdf_bookmarks(pdf_bytes)
            if bookmarks:
                # Try to find chapter by number (e.g., "chapter 1", "chapter 2")
                m = re.search(r"chapter\s*(\d+)", cmd)
                if m:
                    ch_num = int(m.group(1)) - 1
                    if 0 <= ch_num < len(bookmarks):
                        st.session_state["page_index"] = bookmarks[ch_num]["page"]
                        st.sidebar.success(
                            f"Went to chapter {ch_num + 1}: {bookmarks[ch_num]['title']}"
                        )
                        st.rerun()
                else:
                    # Try to find chapter by name
                    for i, bm in enumerate(bookmarks):
                        if bm["title"].lower() in cmd:
                            st.session_state["page_index"] = bm["page"]
                            st.sidebar.success(f"Went to chapter: {bm['title']}")
                            st.rerun()
                            break
                    else:
                        st.sidebar.info(
                            f"Available chapters: {', '.join(bm['title'][:30] for bm in bookmarks[:5])}"
                        )
            else:
                st.sidebar.info("No chapters found in this PDF")
        # set speed
        elif "speed" in cmd:
            import re

            m = re.search(r"speed\s*([0-9]*\.?[0-9]+)", cmd)
            if m:
                val = float(m.group(1))
                st.session_state["reading_rate"] = val
                st.sidebar.success(f"Reading speed set to {val}")
        # read current page
        elif "read" in cmd or "play" in cmd or "speak" in cmd:
            page_text = (
                get_pdf_page_text(
                    pdf_bytes,
                    st.session_state["page_index"],
                    use_ocr=enable_ocr,
                )
                or ""
            )
            if page_text.strip():
                with st.spinner("Generating audio for current page..."):
                    try:
                        buf = convert_text_to_audio(
                            page_text,
                            st.session_state.get("reading_rate", rate),
                        )
                        if not buf:
                            st.sidebar.warning(
                                "No text found on this page to generate audio."
                            )
                        else:
                            st.session_state["last_audio_bytes"] = buf.getvalue()
                        # show playback in sidebar
                        st.sidebar.audio(
                            st.session_state["last_audio_bytes"],
                            format="audio/wav",
                        )
                        # also render synchronized player in main area
                        try:
                            if sync_highlight:
                                html_player = build_sync_player_html(
                                    st.session_state["last_audio_bytes"],
                                    page_text,
                                )
                                components.html(html_player, height=260)
                        except Exception:
                            pass
                    except Exception as e:
                        st.sidebar.error(f"Audio generation failed: {e}")
        else:
            # Fallback: send to LLM (Ollama)
            st.sidebar.info(f"Sending to LLM: '{cmd}'")
            with st.spinner("LLM processing..."):
                try:
                    llm = get_llm_client()
                    if llm:
                        reply = llm.generate(cmd)
                        st.sidebar.success(f"LLM: {reply}")
                        # Convert LLM response to audio using Piper
                        with st.spinner("Generating audio..."):
                            wav = convert_text_to_audio(reply, rate=rate)
                            if wav:
                                st.sidebar.audio(wav, format="audio/wav")
                                if sync_highlight:
                                    try:
                                        player_html = build_sync_player_html(
                                            wav.getvalue(), reply
                                        )
                                        components.html(player_html, height=260)
                                    except Exception:
                                        pass
                    else:
                        st.sidebar.error("LLM not available")
                except Exception as e:
                    st.sidebar.error(f"LLM error: {e}")

    st.sidebar.subheader("Select Page")

    # show image for current page (lazy conversion)
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

    # show text in scrollable fixed-size container (kept original layout)
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

    # If a voice-command or button generated audio bytes, make them available in main view
    if "last_audio_bytes" in st.session_state and st.session_state["last_audio_bytes"]:
        try:
            st.audio(st.session_state["last_audio_bytes"], format="audio/wav")
        except Exception:
            # if bytes are not valid audio, ignore
            pass

    # Sidebar action buttons and audio generation (kept behavior)
    audio_buffer = None
    if st.sidebar.button("Start Reading", key="start_reading"):
        with st.spinner("Generating audio for this page..."):
            try:
                audio_buffer = convert_text_to_audio(page_text, rate)
            except Exception as e:
                st.sidebar.error(f"Audio generation failed: {e}")
                audio_buffer = None

    if audio_buffer is not None:
        audio_container.audio(audio_buffer, format="audio/wav")
        # synchronized player (word-by-word) in main area
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
