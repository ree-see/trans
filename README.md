# Trans - Video Transcription Tool

Quick command-line tool to transcribe videos and audio files to text.

## Features

- **Local file support**: Transcribe audio (mp3, wav, m4a, flac, etc.) and video (mp4, mkv, avi, etc.) files directly
- **Multi-platform support**: YouTube, TikTok, and Twitch (VODs and clips)
- **Automatic source selection**: Tries native captions first (YouTube), falls back to Whisper AI
- **Speaker diarization**: Identify who said what (requires pyannote-audio)
- **Multiple output formats**: TXT, SRT, VTT, JSON, or all formats at once
- **Whisper model selection**: Choose from tiny, base, small, medium, or large models
- **Language support**: Auto-detect or specify language (en, es, fr, etc.)
- **Clipboard integration**: Automatically copy transcripts to clipboard (cross-platform)
- **Batch processing**: Process multiple videos in one command — URLs downloaded concurrently
- **Persistent config**: Save your preferred model, format, and output directory
- **Cache management**: Transcripts cached with TTL, inspect or clear via `trans cache`
- **Clean filenames**: Auto-generated names based on video titles
- **Quiet mode**: Minimal output for scripting

## Installation

### Via pip (recommended)

```bash
# Basic installation
pip install boswell

# With speaker diarization support
pip install boswell[diarize]
```

### From source

```bash
git clone https://github.com/ree-see/trans.git
cd trans
uv pip install -e .

# With speaker diarization
uv pip install -e ".[diarize]"
```

### Requirements

- Python 3.9+
- FFmpeg (for audio extraction)

**Install FFmpeg:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

## Usage

All transcription goes through the `transcribe` subcommand (or `trans transcribe`):

### Local Files

```bash
# Audio files
trans transcribe recording.mp3
trans transcribe interview.wav
trans transcribe ~/Downloads/podcast.m4a

# Video files (audio auto-extracted)
trans transcribe meeting.mp4
trans transcribe lecture.mkv

# With speaker identification
trans transcribe --diarize meeting.mp4

# Higher quality model
trans transcribe --model medium conference_talk.mp3
```

### URLs

```bash
# YouTube (uses native captions when available)
trans transcribe "https://youtube.com/watch?v=..."

# Custom output name
trans transcribe -o my_video "https://tiktok.com/@user/video/123"

# Twitch VOD
trans transcribe "https://twitch.tv/videos/123456789"

# Twitch clip
trans transcribe "https://clips.twitch.tv/FunnyClipName"

# Copy to clipboard automatically
trans transcribe -c "https://youtube.com/watch?v=..."
```

### Output Formats

```bash
# Plain text (default)
trans transcribe "URL"

# SRT subtitles
trans transcribe -f srt "URL"

# VTT subtitles
trans transcribe -f vtt "URL"

# JSON with metadata
trans transcribe -f json "URL"

# All formats at once
trans transcribe -f all "URL"
```

### Whisper Models

```bash
trans transcribe -m tiny   "URL"    # Fastest, lower accuracy
trans transcribe -m base   "URL"    # Balanced (default)
trans transcribe -m small  "URL"    # Better accuracy
trans transcribe -m medium "URL"    # High accuracy
trans transcribe -m large  "URL"    # Best accuracy, slowest
```

### Language Options

```bash
trans transcribe "URL"         # Auto-detect (default)
trans transcribe -l en "URL"   # English
trans transcribe -l es "URL"   # Spanish
trans transcribe -l fr "URL"   # French
trans transcribe -l ja "URL"   # Japanese
```

### Speaker Diarization

Identify different speakers in the transcript:

```bash
trans transcribe --diarize "https://youtube.com/watch?v=..."

# Specify number of speakers (improves accuracy)
trans transcribe --diarize --num-speakers 2 "URL"

# With subtitles
trans transcribe -d -f srt "URL"
```

**Output example (txt):**
```
[Speaker 1]
Hello and welcome to the show.
Today we have a special guest.

[Speaker 2]
Thanks for having me!
I'm excited to be here.
```

**Requirements:**
- `pyannote-audio` package: `pip install pyannote-audio`
- HuggingFace token (free): https://huggingface.co/settings/tokens
- Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1

```bash
pip install pyannote-audio
huggingface-cli login          # or: export HF_TOKEN=hf_your_token_here
```

### Batch Processing

```bash
# Multiple inputs (URLs downloaded concurrently, transcribed with shared model)
trans transcribe "URL1" "URL2" "URL3"

# Mix local files and URLs
trans transcribe recording.mp3 "URL1" meeting.mp4

# Quiet batch
trans transcribe -q "URL1" "URL2" "URL3"
```

### Advanced Options

```bash
# Save output to a specific directory
trans transcribe --output-dir ~/transcripts "URL"

# Keep downloaded audio file
trans transcribe -k "URL"

# Add timestamp to filename (prevents overwrites)
trans transcribe -t "URL"

# Skip cache lookup
trans transcribe --no-cache "URL"

# Always use Whisper (skip native caption check)
trans transcribe --force-whisper "URL"

# TikTok with cookies
trans transcribe --cookies cookies.txt "https://tiktok.com/@user/video/123"
```

## Cache Management

Transcripts are cached automatically (default TTL: 30 days):

```bash
# Show cache stats
trans cache stats

# Clear all cached transcripts
trans cache clear
```

## Persistent Configuration

Save your preferences so you don't have to repeat flags:

```bash
# Show current config (and config file path)
trans config show

# Set defaults
trans config set model small
trans config set format srt
trans config set output_dir ~/transcripts
trans config set clipboard true
trans config set quiet true
trans config set cache.ttl_days 60
trans config set diarization.hf_token hf_your_token_here
```

Config is stored in the OS-appropriate location:
- **macOS**: `~/Library/Application Support/trans/config.toml`
- **Linux**: `~/.config/trans/config.toml`
- **Windows**: `%APPDATA%\trans\config.toml`

## Examples

### Quick transcription to clipboard
```bash
trans transcribe -c "https://youtube.com/watch?v=dQw4w9WgXcQ"
```

### Create subtitles for video editing
```bash
trans transcribe -f srt -m small -o my_subtitles "https://youtube.com/watch?v=..."
```

### Transcribe a foreign language video
```bash
trans transcribe -l es -f all "https://youtube.com/watch?v=..."
```

### Batch research videos, quietly
```bash
trans transcribe -q \
  "https://youtube.com/watch?v=video1" \
  "https://youtube.com/watch?v=video2" \
  "https://youtube.com/watch?v=video3"
```

### Set-and-forget config workflow
```bash
trans config set model small
trans config set output_dir ~/transcripts
trans config set clipboard true
# Now every transcription uses these defaults:
trans transcribe "URL"
```

## File Naming

- **Default**: Auto-generated from video/file title → `How_to_Use_Python.txt`
- **Custom**: `trans transcribe -o my_notes "URL"` → `my_notes.txt`
- **With timestamp**: `trans transcribe -t "URL"` → `How_to_Use_Python_20260222_153045.txt`
- **Custom directory**: `trans transcribe --output-dir ~/docs "URL"` → `~/docs/How_to_Use_Python.txt`

## Supported Formats

| Format | Extension | Use case |
|--------|-----------|----------|
| txt | .txt | Plain text, notes |
| srt | .srt | Video editing |
| vtt | .vtt | Web players |
| json | .json | Full metadata + timestamps |
| all | all above | Everything at once |

## Supported File Types

**Audio**: mp3, wav, m4a, flac, ogg, opus, aac, wma
**Video** (audio auto-extracted): mp4, mkv, avi, mov, webm, flv, wmv, m4v, mpeg, mpg

## Command Reference

```
trans [--version] [--help] COMMAND

Commands:
  transcribe   Transcribe video/audio URLs or local files
  cache        Manage the transcript cache
  config       Manage persistent configuration

trans transcribe [OPTIONS] INPUTS...

Options:
  -o, --output PATH        Output base path (no extension, single input only)
  --output-dir DIR         Directory for output files
  -m, --model MODEL        Whisper model: tiny, base, small, medium, large
  -l, --language LANG      Language code (e.g. en, es). Auto-detect if unset.
  -f, --format FORMAT      Output format: txt, srt, vtt, json, all
  -c, --clipboard          Copy transcript to clipboard
  -k, --keep-audio         Keep downloaded audio file
  -t, --timestamp          Add timestamp to output filename
  -q, --quiet              Minimal output (errors only)
  --cookies PATH           Path to cookies.txt for authenticated downloads
  --no-cache               Skip cache lookup
  --force-whisper          Skip native captions, always use Whisper
  -d, --diarize            Enable speaker diarization
  --num-speakers N         Number of speakers (helps diarization accuracy)
```

## Twitch Notes

Twitch videos rarely have native captions, so Whisper is used automatically. For long VODs use `-m tiny` or `-m base` for speed.

## TikTok Notes

TikTok aggressively blocks datacenter IPs. If you see "IP address is blocked":

1. **Use cookies**: Export from your browser with a "Get cookies.txt" extension:
   ```bash
   trans transcribe --cookies cookies.txt "https://tiktok.com/@user/video/123"
   ```
2. **Run from a residential IP** (home internet, not a VPS)
3. **Use a residential VPN**

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No such file or directory: 'ffmpeg'` | `brew install ffmpeg` or `apt install ffmpeg` |
| Whisper is slow | Use a smaller model: `-m tiny` |
| Wrong language detected | Specify explicitly: `-l en` |
| TikTok blocked | See TikTok Notes above |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (52 tests, all offline)
pytest test_trans.py

# Lint / format
ruff check trans/
black trans/
```

## Credits

Built with:
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — Video/audio downloading
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Speech recognition
- [FFmpeg](https://ffmpeg.org/) — Audio processing
- [Typer](https://typer.tiangolo.com/) — CLI framework
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization (optional)
