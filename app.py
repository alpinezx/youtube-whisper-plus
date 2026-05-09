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
from datetime import datetime

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
            # Merge with defaults so new keys are always present
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
    from faster_whisper import WhisperModel
    if model_name not in _model_cache:
        print(f"  → Loading Whisper model: {model_name} ...")
        _model_cache[model_name] = WhisperModel(
            model_name,
            device="auto",
            compute_type="float16",   # float16 on GPU, graceful fallback on CPU
            download_root=str(MODELS_DIR),
        )
        print(f"  ✓ Model ready: {model_name}")
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


# ─────────────────────────────────────────────
#  Clear
# ─────────────────────────────────────────────
def clear_ui():
    """Reset all UI fields back to defaults."""
    return "", None, "", "Ready.", None, ""


# ─────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────
def run_pipeline(
    url: str,
    model_name: str,
    language_label: str,
    browser_label: str,
):
    """
    Download → transcribe → save. Yields live status updates to Gradio.
    """
    url = url.strip()
    if not url:
        yield "", None, "", "⚠  Please enter a YouTube URL.", None
        return

    # Persist the user's current selections immediately
    save_settings(model_name, language_label, browser_label)

    language_code = LANGUAGES.get(language_label)
    browser       = BROWSERS.get(browser_label)
    started_at    = time.time()
    status        = []

    # ── Download ─────────────────────────────
    status.append("⏳  Downloading audio...")
    if browser:
        status.append(f"     Cookies  : {browser_label}")
    yield "", None, "", "\n".join(status), None

    try:
        video = download_audio(url, DOWNLOADS_DIR, cookies_from_browser=browser)
    except Exception as e:
        status[-1] = f"✗  Download failed:\n    {e}"
        if "Sign in" in str(e) or "bot" in str(e).lower():
            status.append("")
            status.append("  Tip: YouTube flagged this as a bot request.")
            status.append("  Try selecting your browser in the Cookies dropdown")
            status.append("  and clicking Transcribe again.")
        yield "", None, "", "\n".join(status), None
        return

    status[-1] = f"✓  {video.title}"
    if video.duration:
        status.append(f"     Duration : {format_duration(video.duration)}")
        status.append(f"     Uploader : {video.uploader}")

    yield "", video.thumbnail_path, video.title, "\n".join(status), None

    # ── Load model ───────────────────────────
    status.append(f"\n⏳  Loading model '{model_name}'...")
    yield "", video.thumbnail_path, video.title, "\n".join(status), None

    try:
        model = get_model(model_name)
    except Exception as e:
        status[-1] = f"✗  Model load failed:\n    {e}"
        yield "", video.thumbnail_path, video.title, "\n".join(status), None
        return

    status[-1] = f"✓  Model ready: {model_name}"

    # ── Transcribe ───────────────────────────
    status.append("⏳  Transcribing...")
    yield "", video.thumbnail_path, video.title, "\n".join(status), None

    try:
        segments, info = model.transcribe(
            video.audio_path,
            language=language_code,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        transcript    = " ".join(seg.text.strip() for seg in segments)
        detected_lang = info.language if not language_code else language_label

    except Exception as e:
        status[-1] = f"✗  Transcription failed:\n    {e}"
        yield "", video.thumbnail_path, video.title, "\n".join(status), None
        return

    status[-1] = "✓  Transcription complete"
    status.append(f"     Language : {detected_lang}")
    status.append(f"     Words    : ~{len(transcript.split()):,}")

    # ── Save ─────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(
        c for c in video.title if c.isalnum() or c in " _-"
    )[:60].strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = OUTPUTS_DIR / f"{safe_title}_{timestamp}.txt"

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Title    : {video.title}\n")
            f.write(f"URL      : {url}\n")
            f.write(f"Model    : {model_name}\n")
            f.write(f"Language : {detected_lang}\n")
            f.write(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("─" * 60 + "\n\n")
            f.write(transcript)
        status.append(f"\n💾  Saved → outputs/{out_path.name}")
    except Exception as e:
        status.append(f"⚠  Could not save: {e}")

    # ── Elapsed time ─────────────────────────
    elapsed = int(time.time() - started_at)
    status.append(f"⏱  Total time  : {format_elapsed(elapsed)}")

    # ── Cleanup audio ─────────────────────────
    try:
        audio = DOWNLOADS_DIR / "audio.mp3"
        if audio.exists():
            audio.unlink()
    except Exception:
        pass

    yield transcript, video.thumbnail_path, video.title, "\n".join(status), str(out_path)


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

.gradio-container .gr-button-primary {
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

    # ── Inputs row ────────────────────────────
    gr.HTML('<div class="section-label">// Input</div>')
    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=4,
        )
        model_dropdown = gr.Dropdown(
            choices=WHISPER_MODELS,
            value=_settings["model"],
            label="Whisper Model",
            scale=1,
        )
        language_dropdown = gr.Dropdown(
            choices=list(LANGUAGES.keys()),
            value=_settings["language"],
            label="Language",
            scale=1,
        )
        browser_dropdown = gr.Dropdown(
            choices=list(BROWSERS.keys()),
            value=_settings["browser"],
            label="Browser Cookies",
            scale=1,
        )

    with gr.Row():
        run_btn   = gr.Button("▶  TRANSCRIBE", variant="primary", scale=5)
        clear_btn = gr.Button("✕  Clear", variant="secondary", scale=1, elem_classes=["clear-btn"])

    # ── Video info + status ───────────────────
    gr.HTML('<div class="section-label">// Video Info</div>')
    with gr.Row():
        with gr.Column(scale=1):
            thumbnail = gr.Image(
                label="Thumbnail",
                interactive=False,
                height=180,
            )
        with gr.Column(scale=3):
            video_title = gr.Textbox(
                label="Title",
                interactive=False,
            )
            status_box = gr.Textbox(
                label="Status",
                lines=8,
                interactive=False,
                elem_classes=["status-box"],
                value="Ready.",
            )

    # ── Transcript ────────────────────────────
    gr.HTML('<div class="section-label">// Transcript</div>')
    transcript_output = gr.Textbox(
        label="",
        lines=16,
        interactive=False,
        elem_classes=["transcript-box"],
        placeholder="Transcript will appear here...",
    )

    output_file = gr.File(
        label="💾  Download Transcript (.txt)",
        interactive=False,
    )

    gr.HTML("""
        <div class="footer-note">
            WORKFLOW &nbsp;→&nbsp;
            Paste YouTube URL &nbsp;·&nbsp;
            Pick model + language &nbsp;·&nbsp;
            Click TRANSCRIBE &nbsp;·&nbsp;
            Copy or download transcript
            &nbsp;&nbsp;|&nbsp;&nbsp;
            If YouTube blocks the download, select your browser in the Cookies dropdown and try again.
        </div>
    """)

    # ── Wire up ───────────────────────────────
    run_btn.click(
        fn=run_pipeline,
        inputs=[url_input, model_dropdown, language_dropdown, browser_dropdown],
        outputs=[transcript_output, thumbnail, video_title, status_box, output_file],
    )

    clear_btn.click(
        fn=clear_ui,
        inputs=[],
        outputs=[transcript_output, thumbnail, video_title, status_box, output_file, url_input],
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


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        css=DARK_CSS,
    )
