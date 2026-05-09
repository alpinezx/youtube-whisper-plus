# 🎙 YouTube Whisper Plus

A modern, local transcription app powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).  
GPU-accelerated, private, and self-contained. No data leaves your machine.

> **Windows only** — a Linux/Docker port is planned for a future release.

> Forked from [danilotpnta/Youtube-Whisper](https://github.com/danilotpnta/Youtube-Whisper) — modernised and extended.

---

## 🌐 Supported Sites

Despite the name, this app works with **hundreds of sites** — not just YouTube. yt-dlp handles the downloading and supports virtually any platform with publicly accessible video or audio, including:

- YouTube
- X / Twitter
- Instagram
- TikTok
- Vimeo
- SoundCloud
- Reddit
- BBC
- ...and [hundreds more](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

If a site requires a login to view content, select your browser in the **Browser Cookies** dropdown and yt-dlp will borrow its session to access it.

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
├── outputs/              ← Finished transcripts saved here (.txt + .srt)
└── downloads/            ← Temp audio files (auto-cleaned after transcription)
```

---

## 🎙 Whisper Models

| Model | Size | Speed | Quality | Notes |
|---|---|---|---|---|
| tiny | ~75 MB | ⚡⚡⚡⚡ | ★☆☆☆ | Quick drafts |
| base | ~145 MB | ⚡⚡⚡⚡ | ★★☆☆ | Short clips |
| small | ~465 MB | ⚡⚡⚡ | ★★★☆ | Good balance |
| medium | ~1.5 GB | ⚡⚡ | ★★★★ | High quality |
| large-v2 | ~3 GB | ⚡ | ★★★★ | High accuracy |
| large-v3 | ~3 GB | ⚡ | ★★★★ | Best accuracy |
| **turbo** | **~1.6 GB** | **⚡⚡⚡** | **★★★★** | **Default — fast + accurate** |

Models download automatically on first use and cache in `./models/`.

---

## ✨ Improvements over the original repo

| | Original | This fork |
|---|---|---|
| Whisper backend | `openai-whisper` (slow) | `faster-whisper` (4–5× faster) |
| Model loading | Reloads every click | Cached — loads once per session |
| CPU compute type | n/a | `int8` (faster, no silent fallback) |
| Environment | Conda | UV (lightweight, fast) |
| FFmpeg | Manual system install | Bundled or system, auto-detected |
| Launcher | Manual conda steps | Double-click `.bat` |
| Transcript saving | ❌ | ✅ Auto-saved to `./outputs/` |
| SRT subtitle output | ❌ | ✅ Timestamped `.srt` alongside every `.txt` |
| Transcript formatting | Wall of text | Whisper-segmented paragraphs |
| Local file input | ❌ | ✅ Upload mp3, wav, mp4, and more |
| Temp file cleanup | ❌ | ✅ Auto-cleaned (all formats) |
| Language support | 6 fixed options | 10 + auto-detect |
| Whisper Turbo model | ❌ | ✅ (now the default) |
| VAD silence filtering | ❌ | ✅ |
| Saved Outputs tab | ❌ | ✅ Browse, reload, and delete past transcriptions |
| Dark UI | ❌ | ✅ |

---

## 📤 Output Files

Every transcription saves two files to `./outputs/`:

| File | Contents |
|---|---|
| `<title>_<timestamp>.txt` | Plain text transcript with metadata header |
| `<title>_<timestamp>.srt` | Timestamped subtitles with metadata comment block, ready for any video player or editor |

The `.srt` file opens with a comment header like this:

```
; Title    : My Video Title
; Model    : turbo
; Language : en
; Words    : ~3,241
; Date     : 2026-05-09 14:32:01
```

Both files are available as one-click downloads in the app.

---

## 🗂 Saved Outputs Tab

The **🗂 Saved Outputs** tab is fully self-contained — loading a past transcription never affects the YouTube URL or Local File tabs. From here you can:

- **Load** any past transcription — restores the transcript, thumbnail, and download buttons within the tab
- **Delete** any entry — removes the `.txt`, `.srt`, and thumbnail in one click
- **Copy to Clipboard** — copies the transcript with a visual confirmation flash
- **Download** the `.txt` or `.srt` directly from the tab

The list refreshes automatically each time you switch to the tab. If no transcriptions have been saved yet, a placeholder message is shown.

---

## 🛠 Tips

- **`.venv` is always disposable** — delete it and re-run the launcher to reset cleanly
- **FFmpeg on PATH already?** — launcher detects it automatically, no action needed
- **Bundling FFmpeg** — grab `ffmpeg.exe` + `ffprobe.exe` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and drop into `./ffmpeg/`
- **Slow on CPU?** — use `tiny`, `base`, or `small` models. `turbo`+ really wants a GPU
- **Local files** — use the 📂 Local File tab to transcribe any audio or video already on your machine (mp3, wav, m4a, ogg, flac, mp4, webm, mkv, mov)
- **YouTube blocked?** — select your browser in the Cookies dropdown; yt-dlp will borrow its session cookies to bypass bot detection

---

## 📄 License

MIT
