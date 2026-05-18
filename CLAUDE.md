# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`trans` is a Python CLI tool that transcribes YouTube, TikTok, Twitch videos, and local audio/video files to text. It uses `yt-dlp` for downloading, `faster-whisper` for transcription, and optionally `pyannote-audio` for speaker diarization. Published on PyPI as `boswell`.

## Commands

```bash
# Setup (uses uv)
./setup.sh              # installs all dependencies via: uv pip install -e ".[all]"

# Install for development
uv pip install -e ".[dev]"  # includes pytest, black, ruff

# Run tests (offline; no network)
pytest test_trans.py           # all ~158 tests
pytest -v test_trans.py        # verbose
pytest -k "test_url" test_trans.py    # filter by name

# Opt-in: real-world smoke tests (yt-dlp, Whisper). Slow + needs network.
pytest --run-network test_trans.py

# Lint / format
ruff check trans/
black trans/

# Run the CLI directly
python -m trans "https://youtube.com/watch?v=..."
trans "https://youtube.com/watch?v=..."   # if installed via pip
```

## Architecture

Code lives in the `trans/` package; the historical single-file `trans_cli.py` shim was removed. `python -m trans` is the dev entrypoint via `trans/__main__.py`; the PyPI `console_scripts` entry routes `trans` straight to `trans.cli:app`.

Per-module map:

- `trans/cli.py` — Typer CLI app, `_process_url` / `_process_local` orchestration, batch loop, prefetch worker, cache and config subcommands. Houses the `_VERBOSE` / `_TIKTOK_HELP_SHOWN` module-level toggles that the autouse test fixture resets per-test.
- `trans/cache.py` — SQLite cache. Schema v2 stores canonical Whisper segments + info keyed by `video_id`. Rendering to txt/srt/vtt/json happens at hit time via `formatter.write_output`. Migration policy is destructive on schema mismatch (acceptable pre-1.0). `CacheManager.put` auto-prunes expired rows in the same transaction; `CacheManager.prune` and `CacheManager.stats(ttl_days=…)` expose the active/expired split.
- `trans/config.py` — TOML-backed `Config` dataclass with `[defaults]`, `[cache]`, `[diarization]` sections. `save_config` writes atomically (mkstemp + chmod 0o600 + os.replace) to keep the HF token from leaking via umask/symlink races. `SETTABLE_KEYS` drives the `trans config set` allowlist.
- `trans/downloader.py` — yt-dlp Python-API wrapper. `get_video_info` raises `TikTokIPBlockedError` / `DownloaderError` (no `sys.exit`) so the CLI's batch loop can continue past one bad URL. `extract_native_captions` returns a `NativeCaptureResult(files, segments)` NamedTuple; the parsed VTT segments populate the cache with `source='native'` so a subsequent run renders any format without re-fetching. `_BACKEND_HINT_SHOWN` is a one-shot toggle for the curl_cffi-missing warning.
- `trans/transcriber.py` — `TranscriptionEngine` wraps faster-whisper's `WhisperModel`. Lazy-loaded; `device` and `compute_type` (cpu/int8 default, cuda/auto + int8_float16/float16/float32 supported) are constructor args. `extract_audio_from_video` shells out to ffmpeg for video files.
- `trans/diarizer.py` — Optional pyannote-audio integration. Token lookup precedence: HF_TOKEN env > config > cache file.
- `trans/formatter.py` — txt/srt/vtt/json output writer. Speaker labels rendered when segments carry a `speaker` key.
- `trans/utils.py` — Pure functions: URL parsing (`get_video_id`), filename sanitization, VTT/SRT timestamp formatting, speaker assignment, allowlists (`WHISPER_MODELS`, `WHISPER_DEVICES`, `WHISPER_COMPUTE_TYPES`, `OUTPUT_FORMATS`).

### Flow

1. **URL path**: `_process_url` checks the cache, then tries `extract_native_captions` (caches the parsed segments on success), then falls through to `download_audio` + `TranscriptionEngine.transcribe`. Optional diarization merges pyannote speakers into the segment list. `write_output` renders all requested formats.
2. **Local path**: `_process_local` extracts audio from a video via `extract_audio_from_video` if needed, then runs the same transcribe + diarize + render pipeline.
3. **Batch**: when given multiple URLs, the CLI runs a `ThreadPoolExecutor` prefetch (up to 3 workers) so the sequential transcription loop can reuse pre-downloaded audio. Per-URL failures (including `DownloaderError`) are caught so one bad URL doesn't kill the batch; the summary line reports `N succeeded, M failed`.

### Key design decisions

- **Cache key**: platform-prefixed `video_id` (`yt_`, `tt_`, `tw_`, `twclip_`, `hash_`). `--no-cache` bypasses; `trans cache clear` wipes everything; `trans cache prune` removes only expired rows; `trans cache stats` reports active/expired/total counts.
- **Whisper backend**: `faster-whisper` (CTranslate2). Defaults to cpu/int8; opt into GPU via `--device cuda --compute-type float16`. There is no longer a fallback to the original OpenAI Whisper CLI.
- **TikTok**: `yt-dlp` with `ImpersonateTarget.from_str("chrome-131")` when `curl_cffi` is installed (the `[tiktok]` extra). Without the backend, downloads still attempt but degrade to bot-detectable. IP blocks raise `TikTokIPBlockedError`; the CLI prints the multi-line workarounds block one time per session.
- **YouTube**: tries auto-generated captions first via `extract_native_captions`. Parsed VTT segments get cached with `source='native'`, so subsequent `trans -f json URL` runs render JSON without re-fetching.
- **Optional deps**: `pyannote-audio` (diarization) and `curl_cffi` (TikTok impersonation) are guarded with `try/except ImportError`. `rich` is not a direct dep; it's pulled transitively by `typer`.
- **Verbose mode**: `trans -v <subcommand>` enables tracebacks on stderr while keeping the friendly one-liner on stdout. The `-v` flag must precede the subcommand (Typer callback options don't propagate to subcommand argv).

### Tested functions (offline, no network)

Pure utility functions in `trans.utils` are unit-tested directly. The rest of the package is covered by integration tests in `test_trans.py` that monkeypatch yt-dlp / faster-whisper / pyperclip at the seam, so the suite runs offline.

A small `TestTranscribeNetworkSmoke` block at the bottom of `test_trans.py` covers the Typer `transcribe` entry point end-to-end against real yt-dlp and faster-whisper. It is marked `@pytest.mark.network` and skipped unless `--run-network` is passed; the `conftest.py` at the repo root wires the flag.

## Dependencies

- **Runtime (declared in `pyproject.toml`)**: `yt-dlp`, `faster-whisper`, `typer`, `pyperclip`, `platformdirs`, `tomli` (Python < 3.11 only).
- **Optional**: `pyannote-audio` (`[diarize]`, speaker diarization), `curl_cffi` (`[tiktok]`, browser impersonation — TikTok degrades gracefully to a non-impersonated request when this is missing).
- **Transitive**: `rich` is installed by `typer`; `trans` does not import it directly.
- **System**: `ffmpeg` (required for audio extraction/video files), `ffprobe`.
- **Build**: `hatchling`.
