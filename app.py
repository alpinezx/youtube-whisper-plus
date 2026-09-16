"""
YouTube Whisper Plus — Main App
Local YouTube transcription via Faster-Whisper + yt-dlp + Gradio.
"""

import os
import json
import time
import warnings
import gradio as gr
from pathlib import Path
from datetime import datetime, timedelta

from download_video import download_audio, cleanup_download

# Suppress HuggingFace unauthenticated request warning
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
MODELS_DIR    = PROJECT_ROOT / "models"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
SETTINGS_FILE = PROJECT_ROOT / "settings.json"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"]
DEFAULT_MODEL  = "turbo"

LOCAL_AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm", ".mkv", ".mov"]


# ─────────────────────────────────────────────
#  Settings persistence
# ─────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "model":    DEFAULT_MODEL,
    "language": "Auto Detect",
    "browser":  "None (no cookies)",
}

def load_settings() -> dict:
    """Load saved settings, falling back to defaults for any missing keys."""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return {**DEFAULT_SETTINGS, **saved}
    except Exception:
        pass
    return dict(DEFAULT_SETTINGS)

def save_settings(model: str, language: str, browser: str) -> None:
    """Persist the current dropdown selections to disk."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({"model": model, "language": language, "browser": browser}, f, indent=2)
    except Exception as e:
        print(f"  ⚠  Could not save settings: {e}")

# Load once at startup
_settings = load_settings()

LANGUAGES = {
    "Auto Detect":  None,
    "English":      "en",
    "Spanish":      "es",
    "French":       "fr",
    "German":       "de",
    "Italian":      "it",
    "Japanese":     "ja",
    "Chinese":      "zh",
    "Portuguese":   "pt",
    "Russian":      "ru",
    "Korean":       "ko",
}

BROWSERS = {
    "None (no cookies)": None,
    "Firefox":           "firefox",
    "Chrome":            "chrome",
    "Edge":              "edge",
    "Brave":             "brave",
}


# ─────────────────────────────────────────────
#  Model Cache — load once, reuse forever
# ─────────────────────────────────────────────
_model_cache: dict = {}

def get_model(model_name: str):
    """Load a Whisper model, using float16 on GPU and int8 on CPU."""
    import torch
    from faster_whisper import WhisperModel

    if model_name not in _model_cache:
        print(f"  → Loading Whisper model: {model_name} ...")

        # FIX: use int8 on CPU (faster + no silent float16 fallback warning)
        on_gpu        = torch.cuda.is_available()
        compute_type  = "float16" if on_gpu else "int8"
        device        = "cuda"    if on_gpu else "cpu"

        _model_cache[model_name] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(MODELS_DIR),
        )
        print(f"  ✓ Model ready: {model_name} ({device} / {compute_type})")
    return _model_cache[model_name]


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def format_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

def format_elapsed(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m}m {s}s" if m else f"{s}s"

def _srt_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp: HH:MM:SS,mmm"""
    td  = timedelta(seconds=seconds)
    total_ms  = int(td.total_seconds() * 1000)
    h, rem    = divmod(total_ms, 3_600_000)
    m, rem    = divmod(rem, 60_000)
    s, ms     = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_srt(
    segments: list,
    title: str = "",
    model: str = "",
    language: str = "",
    word_count: int = 0,
) -> str:
    """
    Build an SRT string from a list of (start, end, text) tuples.
    Prepends a metadata comment block so the file is self-documenting
    when opened in a subtitle editor or plain text viewer.
    """
    lines = []
    if title:    lines.append(f"; Title    : {title}")
    if model:    lines.append(f"; Model    : {model}")
    if language: lines.append(f"; Language : {language}")
    if word_count: lines.append(f"; Words    : ~{word_count:,}")
    lines.append(f"; Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for i, (start, end, text) in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  Clear
# ─────────────────────────────────────────────
def clear_ui_url():
    """Reset YouTube tab fields."""
    # transcript, thumbnail, title, status, txt_file, srt_file, url_input
    return "", None, "", "Ready.", None, None, ""

def clear_ui_local():
    """Reset Local File tab fields."""
    # transcript, title, status, txt_file, srt_file, local_file
    return "", "", "Ready.", None, None, None


# ─────────────────────────────────────────────
#  Shared transcription core
# ─────────────────────────────────────────────
def _transcribe_audio(
    audio_path: str,
    model_name: str,
    language_code: str | None,
    title: str,
    url: str,
    thumbnail_path: str | None,
    status: list,
):
    """
    Shared generator: load model → transcribe → save → yield updates.

    Yields tuples of:
        (transcript_text, thumbnail_path, title, status_str, txt_path, srt_path)
    """
    # ── Load model ───────────────────────────
    status.append(f"\n⏳  Loading model '{model_name}'...")
    yield "", thumbnail_path, title, "\n".join(status), None, None

    try:
        model = get_model(model_name)
    except Exception as e:
        status[-1] = f"✗  Model load failed:\n    {e}"
        yield "", thumbnail_path, title, "\n".join(status), None, None
        return

    status[-1] = f"✓  Model ready: {model_name}"

    # ── Transcribe ───────────────────────────
    status.append("⏳  Transcribing...")
    yield "", thumbnail_path, title, "\n".join(status), None, None

    try:
        raw_segments, info = model.transcribe(
            audio_path,
            language=language_code,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        # Materialise the generator (needed for both plain text and SRT)
        segments      = [(s.start, s.end, s.text) for s in raw_segments]
        transcript    = "\n\n".join(t.strip() for _, _, t in segments)
        detected_lang = info.language if not language_code else None

    except Exception as e:
        status[-1] = f"✗  Transcription failed:\n    {e}"
        yield "", thumbnail_path, title, "\n".join(status), None, None
        return

    word_count  = len(transcript.split())
    lang_label  = detected_lang or language_code or "unknown"
    status[-1]  = "✓  Transcription complete"
    status.append(f"     Language : {lang_label}")
    status.append(f"     Words    : ~{word_count:,}")
    status.append(f"     Segments : {len(segments)}")

    # ── Save .txt + .srt + thumbnail ─────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(
        c for c in title if c.isalnum() or c in " _-"
    )[:60].strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{safe_title}_{timestamp}"

    txt_path = srt_path = saved_thumbnail = None

    # Copy thumbnail into outputs/ so cleanup_download can't delete it
    if thumbnail_path:
        try:
            import shutil
            thumb_ext     = Path(thumbnail_path).suffix
            saved_thumb_p = OUTPUTS_DIR / f"{base_name}{thumb_ext}"
            shutil.copy2(thumbnail_path, saved_thumb_p)
            saved_thumbnail = str(saved_thumb_p)
        except Exception:
            saved_thumbnail = thumbnail_path  # fall back to original if copy fails

    try:
        txt_path = OUTPUTS_DIR / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Title    : {title}\n")
            if url:
                f.write(f"URL      : {url}\n")
            f.write(f"Model    : {model_name}\n")
            f.write(f"Language : {lang_label}\n")
            f.write(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("─" * 60 + "\n\n")
            f.write(transcript)
        status.append(f"\n💾  Saved → outputs/{txt_path.name}")
    except Exception as e:
        status.append(f"⚠  Could not save .txt: {e}")
        txt_path = None

    try:
        srt_content = build_srt(
            segments,
            title=title,
            model=model_name,
            language=lang_label,
            word_count=word_count,
        )
        srt_path    = OUTPUTS_DIR / f"{base_name}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        status.append(f"💾  Saved → outputs/{srt_path.name}")
    except Exception as e:
        status.append(f"⚠  Could not save .srt: {e}")
        srt_path = None

    yield (
        transcript,
        saved_thumbnail or thumbnail_path,
        title,
        "\n".join(status),
        str(txt_path) if txt_path else None,
        str(srt_path) if srt_path else None,
    )


# ─────────────────────────────────────────────
#  Pipeline — YouTube URL
# ─────────────────────────────────────────────
def run_pipeline(
    url: str,
    model_name: str,
    language_label: str,
    browser_label: str,
):
    """Download → transcribe → save. Yields live status updates to Gradio."""
    url = url.strip()
    if not url:
        yield "", None, "", "⚠  Please enter a YouTube URL.", None, None
        return

    save_settings(model_name, language_label, browser_label)

    language_code = LANGUAGES.get(language_label)
    browser       = BROWSERS.get(browser_label)
    started_at    = time.time()
    status        = []

    # ── Download ─────────────────────────────
    status.append("⏳  Downloading audio...")
    if browser:
        status.append(f"     Cookies  : {browser_label}")
    yield "", None, "", "\n".join(status), None, None

    try:
        video = download_audio(url, DOWNLOADS_DIR, cookies_from_browser=browser)
    except Exception as e:
        status[-1] = f"✗  Download failed:\n    {e}"
        if "Sign in" in str(e) or "bot" in str(e).lower():
            status += [
                "",
                "  Tip: YouTube flagged this as a bot request.",
                "  Try selecting your browser in the Cookies dropdown",
                "  and clicking Transcribe again.",
            ]
        yield "", None, "", "\n".join(status), None, None
        return

    status[-1] = f"✓  {video.title}"
    if video.duration:
        status.append(f"     Duration : {format_duration(video.duration)}")
        status.append(f"     Uploader : {video.uploader}")

    yield "", video.thumbnail_path, video.title, "\n".join(status), None, None

    # ── Transcribe ───────────────────────────
    last = None
    try:
        async_gen = _transcribe_audio(
            video.audio_path, model_name, language_code,
            video.title, url, video.thumbnail_path, status,
        )
        for update in async_gen:
            last = update
            yield update
    except Exception:
        cleanup_download(DOWNLOADS_DIR)
        raise

    cleanup_download(DOWNLOADS_DIR)

    if last:
        elapsed = int(time.time() - started_at)
        status.append(f"⏱  Total time  : {format_elapsed(elapsed)}")
        transcript, thumb, title_out, _, txt_path, srt_path = last
        yield transcript, thumb, title_out, "\n".join(status), txt_path, srt_path


# ─────────────────────────────────────────────
#  Pipeline — Local file
# ─────────────────────────────────────────────
def run_local_pipeline(
    local_file,
    model_name: str,
    language_label: str,
):
    """Transcribe a locally uploaded audio/video file."""
    if local_file is None:
        yield "", "", "⚠  Please upload a file first.", None, None
        return

    audio_path    = local_file
    file_name     = Path(audio_path).name
    started_at    = time.time()
    status        = [f"✓  Local file: {file_name}"]
    language_code = LANGUAGES.get(language_label)

    yield "", file_name, "\n".join(status), None, None

    last = None
    for update in _transcribe_audio(
        audio_path, model_name, language_code,
        file_name, "", None, status,
    ):
        last = update
        # _transcribe_audio yields (transcript, thumb, title, status, txt, srt)
        # drop thumb (index 1) for the local pipeline
        transcript, _, title_out, status_str, txt_path, srt_path = update
        yield transcript, title_out, status_str, txt_path, srt_path

    if last:
        elapsed = int(time.time() - started_at)
        status.append(f"⏱  Total time  : {format_elapsed(elapsed)}")
        transcript, _, title_out, __, txt_path, srt_path = last
        yield transcript, title_out, "\n".join(status), txt_path, srt_path


# ─────────────────────────────────────────────
#  Outputs management
# ─────────────────────────────────────────────
def list_outputs() -> list[str]:
    """
    Return a sorted list of base names (without extension) for every .txt
    file in OUTPUTS_DIR, newest first.
    """
    if not OUTPUTS_DIR.exists():
        return []
    files = sorted(OUTPUTS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files]


def load_output(stem: str):
    """
    Load a saved transcript into the Saved Outputs tab viewer only.
    Never touches the main transcript box or download buttons.
    """
    if not stem:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    txt_path = OUTPUTS_DIR / f"{stem}.txt"
    srt_path = OUTPUTS_DIR / f"{stem}.srt"

    # Find paired thumbnail
    thumbnail_path = None
    for ext in (".jpg", ".jpeg", ".webp", ".png"):
        candidate = OUTPUTS_DIR / f"{stem}{ext}"
        if candidate.exists():
            thumbnail_path = str(candidate)
            break

    try:
        raw = txt_path.read_text(encoding="utf-8")
        if "─" * 10 in raw:
            body = raw.split("─" * 10, 1)[-1].lstrip("\n")
        else:
            body = raw
    except Exception as e:
        body = f"⚠  Could not read file: {e}"

    return (
        body,
        gr.update(value=thumbnail_path, visible=thumbnail_path is not None),
        str(txt_path) if txt_path.exists() else None,
        str(srt_path) if srt_path.exists() else None,
        f"Loaded: {stem}",
    )


def delete_output(stem: str):
    """
    Delete the .txt, .srt, and thumbnail for a given stem.
    Returns an updated file list for the radio component.
    """
    if not stem:
        return gr.update(), "No file selected."

    removed = []
    for ext in (".txt", ".srt", ".jpg", ".jpeg", ".webp", ".png"):
        path = OUTPUTS_DIR / f"{stem}{ext}"
        if path.exists():
            try:
                path.unlink()
                removed.append(ext)
            except OSError as e:
                return gr.update(), f"⚠  Could not delete {path.name}: {e}"

    remaining = list_outputs()
    msg = f"🗑  Deleted {stem} ({', '.join(removed)})" if removed else "File not found."
    return gr.update(choices=remaining, value=None), msg


# ─────────────────────────────────────────────
#  Dark CSS
# ─────────────────────────────────────────────
DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Outfit:wght@300;400;600&display=swap');

body, .gradio-container {
    background: #0d0f14 !important;
    font-family: 'Outfit', sans-serif !important;
}

h1.app-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    letter-spacing: 0.08em;
    color: #e2e8f0;
    margin: 0;
}

.subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #38bdf8;
    letter-spacing: 0.12em;
    margin-top: 4px;
}

.header-block {
    padding: 18px 24px 14px;
    border-bottom: 1px solid #1e2533;
    margin-bottom: 8px;
}

.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
    margin-top: 12px;
}

.gradio-container .gr-button-primary,
button.primary,
button[class*="primary"],
.svelte-1uw5tnk.primary,
footer button.primary,
div.svelte-jd79tz button.primary,
.gr-button.primary {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
    color: #0d0f14 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 14px !important;
    font-size: 1rem !important;
}

/* Override Gradio 6 accent colour (drives tab underline + active state) */
:root {
    --color-accent: #38bdf8 !important;
    --color-accent-soft: rgba(56, 189, 248, 0.15) !important;
}

/* Hide the processing spinner on image components when empty */
.wrap.default.full.svelte-1uj8rng[data-testid="status-tracker"],
div.svelte-1uj8rng[data-testid="status-tracker"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}
.generating,
.generating:not(.hide) {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 1px #38bdf8 !important;
}

@keyframes pulse {
    0%, 100% { border-color: #38bdf8; box-shadow: 0 0 0 1px #38bdf8; }
    50%       { border-color: #818cf8; box-shadow: 0 0 0 2px #818cf8; }
}
button.svelte-11gaq1.selected::after,
button.selected::after {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
    background-color: #38bdf8 !important;
}

button.svelte-11gaq1.selected,
button.selected {
    color: #38bdf8 !important;
}

.clear-btn {
    background: transparent !important;
    border: 1px solid #334155 !important;
    color: #64748b !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
    transition: border-color 0.2s, color 0.2s !important;
}

.clear-btn:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
}

.status-box textarea {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.80rem !important;
    background: #0a0c10 !important;
    color: #94a3b8 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 6px !important;
}

.transcript-box textarea {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    background: #0a0c10 !important;
    color: #cbd5e1 !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 6px !important;
    line-height: 1.7 !important;
}

.footer-note {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.70rem;
    color: #334155;
    margin-top: 20px;
    padding: 10px 16px;
    background: #0a0c10;
    border: 1px solid #1e2533;
    border-radius: 6px;
    line-height: 1.8;
}

.tab-label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.80rem !important;
    letter-spacing: 0.10em !important;
}

.section-header-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 4px !important;
    margin-top: 12px !important;
}

.section-header-row .section-label {
    margin: 0 !important;
}

.copy-btn {
    background: transparent !important;
    border: 1px solid #334155 !important;
    color: #38bdf8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.80rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 6px !important;
    white-space: nowrap !important;
    width: auto !important;
    min-width: unset !important;
    padding: 4px 14px !important;
    flex-shrink: 0 !important;
}
    background: transparent !important;
    border: 1px solid #334155 !important;
    color: #64748b !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.80rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 6px !important;
    transition: border-color 0.2s, color 0.2s !important;
}

.delete-btn:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
}

.load-btn {
    background: transparent !important;
    border: 1px solid #334155 !important;
    color: #38bdf8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.80rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 6px !important;
    transition: border-color 0.2s, color 0.2s !important;
}

.load-btn:hover {
    border-color: #38bdf8 !important;
    color: #e2e8f0 !important;
}

.outputs-status {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #64748b !important;
}
"""


# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────
with gr.Blocks(title="YouTube Whisper Plus") as demo:

    gr.HTML("""
        <div class="header-block">
            <h1 class="app-title">YOUTUBE WHISPER PLUS</h1>
            <div class="subtitle">// FASTER-WHISPER · YT-DLP · LOCAL · PRIVATE</div>
        </div>
    """)

    # ── Model / language controls (shared across tabs) ───
    gr.HTML('<div class="section-label">// Model Settings</div>')
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=WHISPER_MODELS,
            value=_settings["model"],
            label="Whisper Model",
            scale=2,
        )
        language_dropdown = gr.Dropdown(
            choices=list(LANGUAGES.keys()),
            value=_settings["language"],
            label="Language",
            scale=2,
        )
        browser_dropdown = gr.Dropdown(
            choices=list(BROWSERS.keys()),
            value=_settings["browser"],
            label="Browser Cookies (YouTube only)",
            scale=2,
        )

    # ── Source tabs ───────────────────────────
    gr.HTML('<div class="section-label">// Source</div>')
    with gr.Tabs():

        # ── Tab 1: YouTube URL ────────────────
        with gr.Tab("▶  YouTube URL", elem_classes=["tab-label"]):
            with gr.Row():
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    scale=5,
                )
            with gr.Row():
                run_btn   = gr.Button("▶  TRANSCRIBE", variant="primary", scale=5)
                clear_btn = gr.Button("✕  Clear", variant="secondary", scale=1, elem_classes=["clear-btn"])

            gr.HTML('<div class="section-label">// Info & Status</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    thumbnail = gr.Image(label="Thumbnail", interactive=False, height=180)
                with gr.Column(scale=3):
                    video_title = gr.Textbox(label="Title / Filename", interactive=False)
                    status_box  = gr.Textbox(
                        label="Status", lines=8, interactive=False,
                        elem_classes=["status-box"], value="Ready.",
                    )

            with gr.Row(elem_classes=["section-header-row"]):
                gr.HTML('<div class="section-label" style="margin:0;align-self:center;">// Transcript</div>')
                copy_btn = gr.Button("📋  Copy to Clipboard", scale=0, elem_classes=["copy-btn"], min_width=0)
            transcript_output = gr.Textbox(
                label="Transcript", lines=16, interactive=False,
                elem_classes=["transcript-box"], placeholder="Transcript will appear here...",
            )
            with gr.Row():
                txt_file = gr.File(label="💾  Download Plain Text (.txt)", interactive=False, scale=2)
                srt_file = gr.File(label="💾  Download Subtitles (.srt)",  interactive=False, scale=2)

        # ── Tab 2: Local file ─────────────────
        with gr.Tab("📂  Local File", elem_classes=["tab-label"]):
            local_file_input = gr.File(
                label="Upload audio or video file",
                file_types=[".mp3", ".wav", ".m4a", ".ogg", ".flac",
                            ".mp4", ".webm", ".mkv", ".mov"],
                file_count="single",
            )
            with gr.Row():
                local_run_btn   = gr.Button("▶  TRANSCRIBE FILE", variant="primary", scale=5)
                local_clear_btn = gr.Button("✕  Clear", variant="secondary", scale=1, elem_classes=["clear-btn"])

            gr.HTML('<div class="section-label">// Info & Status</div>')
            with gr.Row():
                local_title      = gr.Textbox(label="Title / Filename", interactive=False)
                local_status_box = gr.Textbox(
                    label="Status", lines=8, interactive=False,
                    elem_classes=["status-box"], value="Ready.",
                )

            with gr.Row(elem_classes=["section-header-row"]):
                gr.HTML('<div class="section-label" style="margin:0;align-self:center;">// Transcript</div>')
                local_copy_btn = gr.Button("📋  Copy to Clipboard", scale=0, elem_classes=["copy-btn"], min_width=0)
            local_transcript = gr.Textbox(
                label="Transcript", lines=16, interactive=False,
                elem_classes=["transcript-box"], placeholder="Transcript will appear here...",
            )
            with gr.Row():
                local_txt_file = gr.File(label="💾  Download Plain Text (.txt)", interactive=False, scale=2)
                local_srt_file = gr.File(label="💾  Download Subtitles (.srt)",  interactive=False, scale=2)

        # ── Tab 3: Saved outputs ──────────────
        with gr.Tab("🗂  Saved Outputs", elem_classes=["tab-label"]) as outputs_tab:
            gr.HTML('<div class="section-label">// Saved Transcriptions</div>')
            _initial = list_outputs()
            outputs_empty = gr.HTML(
                value="" if _initial else '<div class="footer-note">No saved transcriptions yet. Transcribe something and it will appear here.</div>',
                visible=not bool(_initial),
            )
            outputs_radio = gr.Radio(
                choices=_initial, value=None,
                label="Select a transcription",
                interactive=True, visible=bool(_initial),
            )
            with gr.Row():
                load_btn   = gr.Button("↩  Load", scale=3, elem_classes=["load-btn"])
                delete_btn = gr.Button("🗑  Delete", scale=1, elem_classes=["delete-btn"])
            outputs_status = gr.Textbox(
                label="", interactive=False, lines=1,
                elem_classes=["outputs-status"], show_label=False,
            )

            with gr.Row(elem_classes=["section-header-row"]):
                gr.HTML('<div class="section-label" style="margin:0;align-self:center;">// Transcript</div>')
                hist_copy_btn = gr.Button("📋  Copy to Clipboard", scale=0, elem_classes=["copy-btn"], min_width=0)
            hist_thumbnail = gr.Image(label="Thumbnail", interactive=False, height=180, visible=False)
            hist_transcript = gr.Textbox(
                label="", lines=16, interactive=False,
                elem_classes=["transcript-box"],
                placeholder="Select a transcription above and click Load...",
                show_label=False,
            )
            with gr.Row():
                hist_txt_file = gr.File(label="💾  Download Plain Text (.txt)", interactive=False, scale=2)
                hist_srt_file = gr.File(label="💾  Download Subtitles (.srt)",  interactive=False, scale=2)

    gr.HTML("""
        <div class="footer-note">
            WORKFLOW &nbsp;→&nbsp;
            Paste a YouTube URL <em>or</em> upload a local file &nbsp;·&nbsp;
            Pick model + language &nbsp;·&nbsp;
            Click TRANSCRIBE &nbsp;·&nbsp;
            Copy or download transcript / SRT
            &nbsp;&nbsp;|&nbsp;&nbsp;
            If YouTube blocks the download, select your browser in the Cookies dropdown and try again.
        </div>
    """)

    # ── Wire up — YouTube ─────────────────────
    _yt_outputs = [transcript_output, thumbnail, video_title, status_box, txt_file, srt_file]

    run_btn.click(
        fn=run_pipeline,
        inputs=[url_input, model_dropdown, language_dropdown, browser_dropdown],
        outputs=_yt_outputs,
    )

    clear_btn.click(
        fn=clear_ui_url,
        inputs=[],
        outputs=_yt_outputs + [url_input],
    )

    copy_btn.click(
        fn=None,
        inputs=[transcript_output],
        outputs=[],
        js="""(text) => {
            navigator.clipboard.writeText(text);
            const btns = document.querySelectorAll('.copy-btn');
            btns.forEach(btn => {
                if (btn.innerText.includes('Copy')) {
                    const original = btn.innerText;
                    btn.innerText = '✓  Copied!';
                    btn.style.borderColor = '#38bdf8';
                    btn.style.color = '#38bdf8';
                    btn.style.background = 'rgba(56, 189, 248, 0.12)';
                    setTimeout(() => {
                        btn.innerText = original;
                        btn.style.borderColor = '';
                        btn.style.color = '';
                        btn.style.background = '';
                    }, 2000);
                }
            });
        }""",
    )

    # ── Wire up — Local file ──────────────────
    _local_outputs = [local_transcript, local_title, local_status_box, local_txt_file, local_srt_file]

    local_run_btn.click(
        fn=run_local_pipeline,
        inputs=[local_file_input, model_dropdown, language_dropdown],
        outputs=_local_outputs,
    )

    local_clear_btn.click(
        fn=clear_ui_local,
        inputs=[],
        outputs=_local_outputs + [local_file_input],
    )

    local_copy_btn.click(
        fn=None,
        inputs=[local_transcript],
        outputs=[],
        js="""(text) => {
            navigator.clipboard.writeText(text);
            const btns = document.querySelectorAll('.copy-btn');
            btns.forEach(btn => {
                if (btn.innerText.includes('Copy')) {
                    const original = btn.innerText;
                    btn.innerText = '✓  Copied!';
                    btn.style.borderColor = '#38bdf8';
                    btn.style.color = '#38bdf8';
                    btn.style.background = 'rgba(56, 189, 248, 0.12)';
                    setTimeout(() => {
                        btn.innerText = original;
                        btn.style.borderColor = '';
                        btn.style.color = '';
                        btn.style.background = '';
                    }, 2000);
                }
            });
        }""",
    )

    # ── Persist settings on every dropdown change ──
    def _save_on_change(model, language, browser):
        save_settings(model, language, browser)

    for dropdown in [model_dropdown, language_dropdown, browser_dropdown]:
        dropdown.change(
            fn=_save_on_change,
            inputs=[model_dropdown, language_dropdown, browser_dropdown],
            outputs=[],
        )

    # ── Outputs tab — refresh list when tab is selected ──
    def _refresh_outputs():
        items = list_outputs()
        has   = bool(items)
        return (
            gr.update(choices=items, value=None, visible=has),
            gr.update(visible=not has),
        )

    outputs_tab.select(
        fn=_refresh_outputs,
        inputs=[],
        outputs=[outputs_radio, outputs_empty],
    )

    # ── Outputs tab — load (into tab-local viewer only) ──
    load_btn.click(
        fn=load_output,
        inputs=[outputs_radio],
        outputs=[hist_transcript, hist_thumbnail, hist_txt_file, hist_srt_file, outputs_status],
    )

    hist_copy_btn.click(
        fn=None,
        inputs=[hist_transcript],
        outputs=[],
        js="""(text) => {
            navigator.clipboard.writeText(text);
            const btns = document.querySelectorAll('.copy-btn');
            btns.forEach(btn => {
                if (btn.innerText.includes('Copy')) {
                    const original = btn.innerText;
                    btn.innerText = '✓  Copied!';
                    btn.style.borderColor = '#38bdf8';
                    btn.style.color = '#38bdf8';
                    btn.style.background = 'rgba(56, 189, 248, 0.12)';
                    setTimeout(() => {
                        btn.innerText = original;
                        btn.style.borderColor = '';
                        btn.style.color = '';
                        btn.style.background = '';
                    }, 2000);
                }
            });
        }""",
    )

    # ── Outputs tab — delete ──────────────────
    def _delete_and_refresh(stem):
        radio_update, msg = delete_output(stem)
        items = list_outputs()
        has   = bool(items)
        return (
            gr.update(choices=items, value=None, visible=has),
            gr.update(visible=not has),
            msg,
            # Clear the viewer below
            "",           # hist_transcript
            None,         # hist_thumbnail
            None,         # hist_txt_file
            None,         # hist_srt_file
        )

    delete_btn.click(
        fn=_delete_and_refresh,
        inputs=[outputs_radio],
        outputs=[outputs_radio, outputs_empty, outputs_status,
                 hist_transcript, hist_thumbnail, hist_txt_file, hist_srt_file],
    )


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=DARK_CSS,
    )
