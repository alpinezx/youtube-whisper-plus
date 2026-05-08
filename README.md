# 🎙 YouTube Whisper Plus

A modern, local YouTube transcription app powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).  
GPU-accelerated, private, and self-contained. No data leaves your machine.

> Forked from [danilotpnta/Youtube-Whisper](https://github.com/danilotpnta/Youtube-Whisper) — modernised and extended.

---

## 🚀 Quick Start

### You only need two things installed manually

| Requirement | How to get it |
|---|---|
| **UV** | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` |
| **FFmpeg** | Drop `ffmpeg.exe` + `ffprobe.exe` into `./ffmpeg/` — or already on system PATH |

> **NVIDIA GPU?** No extra setup needed — the launcher detects it automatically via `nvidia-smi` and installs the right PyTorch CUDA build. Compatible with CUDA 12.1 through 13.x drivers.

Everything else — **Python, PyTorch, Gradio, faster-whisper, yt-dlp, Whisper models** — is handled automatically on first run.

### Run
```
double-click launch.bat
```

That's it — one launcher for everyone. It auto-detects everything:

1. Installs Python 3.11/3.12 automatically via UV if needed
2. Detects your NVIDIA GPU via `nvidia-smi`
   - GPU found → installs PyTorch with CUDA support (fast ⚡)
   - No GPU → installs CPU-only PyTorch (slower but works)
3. Creates a virtual environment
4. Installs all Python dependencies
5. Opens the app in your browser at `http://127.0.0.1:7860`

Whisper models download automatically on first transcription.

---

## 📁 Project Structure

```
youtube-whisper-plus/
│
├── app.py                ← Gradio UI + transcription pipeline
├── download_video.py     ← yt-dlp audio + thumbnail downloader
├── launch.bat            ← Windows launcher (GPU/CUDA)
├── requirements.txt      ← Python dependencies
│
├── .venv/                ← auto-created by UV, always disposable
├── ffmpeg/               ← drop ffmpeg.exe + ffprobe.exe here (optional)
├── models/               ← Whisper models auto-downloaded here
├── outputs/              ← Finished transcripts saved here
└── downloads/            ← Temp audio files (auto-cleaned after transcription)
```

---

## 🎙 Whisper Models

| Model | Size | Speed | Quality | Notes |
|---|---|---|---|---|
| tiny | ~75 MB | ⚡⚡⚡⚡ | ★☆☆☆ | Quick drafts |
| base | ~145 MB | ⚡⚡⚡⚡ | ★★☆☆ | Short clips |
| small | ~465 MB | ⚡⚡⚡ | ★★★☆ | Good balance |
| **medium** | **~1.5 GB** | **⚡⚡** | **★★★★** | **Default** |
| large-v2 | ~3 GB | ⚡ | ★★★★ | High accuracy |
| large-v3 | ~3 GB | ⚡ | ★★★★ | Best accuracy |
| turbo | ~1.6 GB | ⚡⚡⚡ | ★★★★ | Fast + accurate |

Models download automatically on first use and cache in `./models/`.

---

## ✨ Improvements over the original repo

| | Original | This fork |
|---|---|---|
| Whisper backend | `openai-whisper` (slow) | `faster-whisper` (4–5× faster) |
| Model loading | Reloads every click | Cached — loads once per session |
| Environment | Conda | UV (lightweight, fast) |
| FFmpeg | Manual system install | Bundled or system, auto-detected |
| Launcher | Manual conda steps | Double-click `.bat` |
| Transcript saving | ❌ | ✅ Auto-saved to `./outputs/` |
| Temp file cleanup | ❌ | ✅ Auto-cleaned |
| Language support | 6 fixed options | 10 + auto-detect |
| Whisper Turbo model | ❌ | ✅ |
| VAD silence filtering | ❌ | ✅ |
| Dark UI | ❌ | ✅ |

---

## 🛠 Tips

- **`.venv` is always disposable** — delete it and re-run the launcher to reset cleanly
- **FFmpeg on PATH already?** — launcher detects it automatically, no action needed
- **Bundling FFmpeg** — grab `ffmpeg.exe` + `ffprobe.exe` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and drop into `./ffmpeg/`
- **Slow on CPU?** — use `tiny`, `base`, or `small` models. `medium`+ really wants a GPU

---

## 📄 License

MIT
