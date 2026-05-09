"""
YouTube Whisper Plus — Downloader
Handles audio + metadata extraction from YouTube URLs using yt-dlp.
"""

import yt_dlp
from pathlib import Path
from dataclasses import dataclass

# ─────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────
@dataclass
class VideoInfo:
    title: str
    thumbnail_path: str | None
    audio_path: str
    duration: int        # seconds
    uploader: str


# ─────────────────────────────────────────────
#  Downloader
# ─────────────────────────────────────────────
def download_audio(
    youtube_url: str,
    downloads_dir: Path,
    cookies_from_browser: str | None = None,
) -> VideoInfo:
    """
    Download audio + thumbnail from a YouTube URL in a single yt-dlp pass.

    Args:
        youtube_url:          YouTube URL to download
        downloads_dir:        Folder to save audio + thumbnail into
        cookies_from_browser: Browser name to pull cookies from e.g. 'firefox',
                              'chrome', 'edge', 'brave'. None = no cookies.
    """

    downloads_dir = Path(downloads_dir)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    audio_out     = downloads_dir / "audio.%(ext)s"
    thumbnail_out = downloads_dir / "thumbnail"

    ydl_opts = {
        # ── Audio ──────────────────────────────
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],

        # ── Thumbnail ──────────────────────────
        "writethumbnail": True,
        "outtmpl": {
            "default": str(audio_out),
            "thumbnail": str(thumbnail_out),
        },

        # ── Behaviour ─────────────────────────
        "quiet": True,
        "no_warnings": True,
        "noprogress": False,
    }

    # ── Cookie support (bypasses YouTube bot detection) ──
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser.lower(),)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    # Resolve final audio path
    audio_path = downloads_dir / "audio.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found after download. Expected: {audio_path}"
        )

    # Find thumbnail
    thumbnail_path = None
    for ext in ("jpg", "jpeg", "webp", "png"):
        candidate = downloads_dir / f"thumbnail.{ext}"
        if candidate.exists():
            thumbnail_path = str(candidate)
            break

    return VideoInfo(
        title=info.get("title", "Unknown Title"),
        thumbnail_path=thumbnail_path,
        audio_path=str(audio_path),
        duration=info.get("duration", 0),
        uploader=info.get("uploader", "Unknown"),
    )


# ─────────────────────────────────────────────
#  Cleanup
# ─────────────────────────────────────────────
def cleanup_download(downloads_dir: Path) -> None:
    """
    Remove all temporary files created during a download pass.

    Covers: audio in any format yt-dlp may leave behind, thumbnails in any
    image format, and any leftover intermediate files (e.g. .webm before
    FFmpeg conversion).
    """
    downloads_dir = Path(downloads_dir)
    if not downloads_dir.exists():
        return

    patterns = (
        # Audio — final and intermediate
        "audio.mp3",
        "audio.webm",
        "audio.m4a",
        "audio.ogg",
        "audio.opus",
        "audio.wav",
        # Thumbnails — all image formats yt-dlp may write
        "thumbnail.jpg",
        "thumbnail.jpeg",
        "thumbnail.webp",
        "thumbnail.png",
    )

    for pattern in patterns:
        target = downloads_dir / pattern
        if target.exists():
            try:
                target.unlink()
            except OSError as e:
                print(f"  ⚠  Could not remove {target.name}: {e}")
