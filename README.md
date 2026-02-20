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
- **Clipboard integration**: Automatically copy transcripts to clipboard
- **Batch processing**: Process multiple videos in one command
- **Clean filenames**: Auto-generated names based on video titles (no messy timestamps)
- **Audio preservation**: Option to keep downloaded audio files
- **Quiet mode**: Minimal output for scripting

## Installation

### Via pip (recommended)

```bash
# Basic installation
pip install trans-cli

# With speaker diarization support
pip install trans-cli[diarize]

# With all optional features
pip install trans-cli[all]
```

### From source

```bash
# Clone the repository
git clone https://github.com/ree-see/trans.git
cd trans

# Install in development mode
pip install -e .

# Or with optional features
pip install -e ".[all]"
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

### Local Files

```bash
# Audio files
trans recording.mp3
trans interview.wav
trans ~/Downloads/podcast.m4a

# Video files (auto-extracts audio)
trans meeting.mp4
trans lecture.mkv
trans ~/Videos/interview.mov

# With speaker identification
trans --diarize meeting.mp4

# Higher quality model
trans --model medium conference_talk.mp3
```

### URLs

```bash
# Simple transcription (auto-named from video title)
trans "https://youtube.com/watch?v=..."

# Custom output name
trans -o my_video "https://tiktok.com/@user/video/123"

# Twitch VOD
trans "https://twitch.tv/videos/123456789"

# Twitch clip
trans "https://clips.twitch.tv/FunnyClipName"

# Copy to clipboard automatically
trans -c "https://youtube.com/watch?v=..."
```

### Output Formats

```bash
# Plain text (default)
trans "URL"

# SRT subtitles (for video editing)
trans -f srt "URL"

# VTT subtitles (for web players)
trans -f vtt "URL"

# JSON with metadata
trans -f json "URL"

# All formats at once
trans -f all "URL"
```

### Whisper Models

Choose model based on speed vs accuracy trade-off:

```bash
# Fastest, lower accuracy
trans -m tiny "URL"

# Balanced (default)
trans -m base "URL"

# Better accuracy, slower
trans -m small "URL"

# High accuracy
trans -m medium "URL"

# Best accuracy, slowest
trans -m large "URL"
```

### Language Options

```bash
# Auto-detect language (default)
trans "URL"

# Specify language for better accuracy
trans -l en "URL"    # English
trans -l es "URL"    # Spanish
trans -l fr "URL"    # French
trans -l ja "URL"    # Japanese
```

### Speaker Diarization

Identify different speakers in the transcript (who said what):

```bash
# Enable speaker diarization
trans --diarize "https://youtube.com/watch?v=..."

# Specify number of speakers (improves accuracy)
trans --diarize --num-speakers 2 "URL"

# Diarization with subtitles
trans -d -f srt "URL"
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

**Output example (srt/vtt):**
```
1
00:00:00,000 --> 00:00:03,500
[Speaker 1] Hello and welcome to the show.
```

**Requirements:**
- `pyannote-audio` package: `pip install pyannote-audio`
- HuggingFace token (free): https://huggingface.co/settings/tokens
- Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1

**Setup:**
```bash
# Install pyannote-audio
pip install pyannote-audio

# Login to HuggingFace (stores token)
huggingface-cli login

# Or set environment variable
export HF_TOKEN=hf_your_token_here
```

### Advanced Options

```bash
# Keep audio file after transcription
trans -k "URL"

# Add timestamp to filename
trans -t "URL"

# Quiet mode (minimal output)
trans -q "URL"

# Combine options
trans -f srt -m small -l en -c -o "my_video" "URL"
```

### Batch Processing

```bash
# Process multiple videos
trans "URL1" "URL2" "URL3"

# Quiet batch processing
trans -q "URL1" "URL2" "URL3"

# Batch with custom format
trans -f srt "URL1" "URL2" "URL3"
```

## Examples

### Quick transcription for notes
```bash
trans -c "https://youtube.com/watch?v=dQw4w9WgXcQ"
# Transcript is in clipboard, ready to paste
```

### Create subtitles for video editing
```bash
trans -f srt -m small -o "my_subtitles" "https://youtube.com/watch?v=..."
# Creates: my_subtitles.srt
```

### Transcribe foreign language video
```bash
trans -l es -f all "https://youtube.com/watch?v=..."
# Creates all format files with Spanish transcription
```

### Batch process research videos
```bash
trans -q \
  "https://youtube.com/watch?v=video1" \
  "https://youtube.com/watch?v=video2" \
  "https://youtube.com/watch?v=video3"
# Quietly processes all three, auto-named files
```

### Keep audio for podcast archiving
```bash
trans -k -o "podcast_ep_001" "https://youtube.com/watch?v=..."
# Creates: podcast_ep_001.txt and podcast_ep_001.audio.mp3
```

## File Naming

By default, `trans` creates clean filenames from video titles:

- **Without `-o`**: Auto-generated from video title
  - Example: `How_to_Use_Python_for_Data_Science.txt`
- **With `-o`**: Uses your custom name
  - Example: `trans -o my_notes "URL"` → `my_notes.txt`
- **With `-t`**: Adds timestamp to prevent overwrites
  - Example: `How_to_Use_Python_20260114_153045.txt`

## Output Files

Depending on format, you'll get:

- **txt**: Plain text transcript
- **srt**: SubRip subtitles (video editing)
- **vtt**: WebVTT subtitles (web players)
- **json**: Full metadata and timestamps
- **all**: Creates txt, srt, vtt, json, and tsv files

## Requirements

- macOS (uses `pbcopy` for clipboard)
- Python 3.9+
- FFmpeg (installed via setup)
- Internet connection

## Command Options

```
positional arguments:
  URL                   YouTube, TikTok, or Twitch video URL(s)

options:
  -h, --help            Show help message
  -o, --output OUTPUT   Output file path (without extension)
  -m, --model MODEL     Whisper model: tiny, base, small, medium, large
  -l, --language LANG   Language code (e.g., en, es, fr)
  -f, --format FORMAT   Output format: txt, srt, vtt, json, all
  -c, --clipboard       Copy transcript to clipboard
  -k, --keep-audio      Keep downloaded audio file
  -t, --timestamp       Add timestamp to filename
  -q, --quiet           Minimal output (errors only)
  -d, --diarize         Enable speaker diarization (who said what)
  --num-speakers N      Number of speakers (helps diarization accuracy)
  --cookies PATH        Path to cookies.txt file (for TikTok, etc.)
  --force-whisper       Skip native captions, always use Whisper
```

## Twitch Notes

Twitch VODs and clips work well out of the box:

- **VODs**: Full stream recordings (https://twitch.tv/videos/123456789)
- **Clips**: Short highlights (https://clips.twitch.tv/ClipName)

Twitch videos rarely have native captions, so trans will use Whisper transcription. For long VODs, consider:
- Use `-m tiny` or `-m base` for faster transcription
- Use `--force-whisper` if you want to skip caption checks

## TikTok Notes

TikTok aggressively blocks datacenter/VPS IP addresses. If you see "IP address is blocked", try:

1. **Use cookies**: Export cookies from your logged-in browser session using an extension like "Get cookies.txt", then:
   ```bash
   trans --cookies cookies.txt "https://tiktok.com/@user/video/123"
   ```

2. **Run from a residential IP**: TikTok usually works from home internet connections

3. **Use a VPN**: Connect to a residential VPN endpoint

The tool automatically uses browser impersonation (Chrome) when accessing TikTok, but IP-level blocks still apply.

## Supported File Formats

### Audio
- mp3, wav, m4a, flac, ogg, opus, aac, wma

### Video (audio auto-extracted)
- mp4, mkv, avi, mov, webm, flv, wmv, m4v, mpeg, mpg

## Tips

1. **Local files**: No network needed, works offline
2. **Video files**: Audio is automatically extracted using ffmpeg
3. **YouTube videos**: Usually have native captions (faster)
4. **TikTok videos**: Always use Whisper (more accurate for short clips)
5. **Twitch VODs**: Use smaller models for long streams
6. **Long videos**: Use `-m tiny` or `-m base` for speed
7. **Accuracy matters**: Use `-m medium` or `-m large`
8. **Scripting**: Use `-q` for clean output in scripts
9. **Clipboard workflow**: Use `-c -q` for copy-paste workflow
10. **Podcasts/interviews**: Use `-d` (diarize) to identify speakers
11. **Known speakers**: Use `--num-speakers N` for better diarization

## Troubleshooting

### "No such file or directory: 'yt-dlp'"
- Run `./setup.sh` to reinstall dependencies

### "No such file or directory: 'ffmpeg'"
- Run `brew install ffmpeg`

### Whisper is slow
- Use a smaller model: `-m tiny` or `-m base`
- For long videos, consider `-m tiny` for quick drafts

### Wrong language detected
- Specify language explicitly: `-l en` (or your language code)

## Credits

Built with:
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Video/audio downloading
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [FFmpeg](https://ffmpeg.org/) - Audio processing
