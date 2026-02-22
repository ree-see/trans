# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`trans` is a single-file Python CLI tool (`trans_cli.py`) that transcribes YouTube, TikTok, Twitch videos, and local audio/video files to text. It uses `yt-dlp` for downloading, `faster-whisper` for transcription, and optionally `pyannote-audio` for speaker diarization.

## Commands

```bash
# Setup (uses uv)
./setup.sh              # installs all dependencies via: uv pip install -e ".[all]"

# Install for development
uv pip install -e ".[dev]"  # includes pytest, black, ruff

# Run tests
pytest test_trans.py           # all 52 tests
pytest -v test_trans.py        # verbose
pytest -k "test_url" test_trans.py    # filter by name

# Lint / format
ruff check trans_cli.py
black trans_cli.py

# Run the CLI directly
python trans_cli.py "https://youtube.com/watch?v=..."
trans "https://youtube.com/watch?v=..."   # if installed via pip
```

## Architecture

The entire tool lives in one file: `trans_cli.py` (~1200 lines). The flow is:

1. **`main()`** — parses args, routes each input to either `process_local_file()` or `process_url()`
2. **URL path**: `process_url()` → checks SQLite cache → tries native captions (`extract_native_captions()`) → falls back to download + Whisper (`download_audio_with_progress()` + `transcribe_with_faster_whisper()`)
3. **Local file path**: `process_local_file()` → extracts audio if video (`extract_audio_from_video()`) → `transcribe_local_audio()` → `transcribe_with_faster_whisper()`
4. **Output**: `write_output()` handles txt/srt/vtt/json formats; speaker diarization merges pyannote segments with Whisper segments via `assign_speakers_to_segments()`

### Key design decisions

- **Cache**: SQLite DB at `.cache/transcripts.db`, keyed by platform-prefixed video ID (`yt_`, `tt_`, `tw_`, `twclip_`, `hash_`). Use `--no-cache` or `--clear-cache` to bypass.
- **Whisper backend**: Uses `faster-whisper` (CTranslate2) as the primary transcription engine, not the original OpenAI Whisper CLI. Falls back to the `whisper` CLI if `faster-whisper` isn't installed.
- **TikTok**: Uses `yt-dlp --impersonate chrome-131` for browser impersonation; IP blocks require cookies or a residential IP.
- **YouTube**: Tries native auto-generated captions via `yt-dlp --write-auto-sub` before running Whisper.
- **Optional deps**: `tqdm`, `rich`, `pyannote-audio`, `curl_cffi` are all guarded with `try/except ImportError` at the top of the file.

### Tested functions (offline, no network)

`get_video_id`, `sanitize_filename`, `is_tiktok_url`, `is_twitch_url`, `is_local_file`, `is_audio_file`, `format_timestamp_srt`, `format_timestamp_vtt`, `format_speaker_label`, `assign_speakers_to_segments`

## Dependencies

- **Runtime**: `yt-dlp`, `faster-whisper`, `tqdm`
- **Optional**: `pyannote-audio` (diarization), `curl_cffi` (TikTok), `rich` (pretty output)
- **System**: `ffmpeg` (required for audio extraction/video files), `ffprobe`
- **Build**: `hatchling`
