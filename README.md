# Trans - Video Transcription Tool

Quick command-line tool to transcribe YouTube, TikTok, and Twitch videos to text.

## Features

- **Multi-platform support**: YouTube, TikTok, and Twitch (VODs and clips)
- **Automatic source selection**: Tries native captions first (YouTube), falls back to Whisper AI
- **Multiple output formats**: TXT, SRT, VTT, JSON, or all formats at once
- **Whisper model selection**: Choose from tiny, base, small, medium, or large models
- **Language support**: Auto-detect or specify language (en, es, fr, etc.)
- **Clipboard integration**: Automatically copy transcripts to clipboard
- **Batch processing**: Process multiple videos in one command
- **Clean filenames**: Auto-generated names based on video titles (no messy timestamps)
- **Audio preservation**: Option to keep downloaded audio files
- **Quiet mode**: Minimal output for scripting

## Installation

```bash
# Clone/download to ~/dev/trans
cd ~/dev/trans

# Run setup
./setup.sh

# The 'trans' command is now available system-wide
```

## Usage

### Basic Usage

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

## Tips

1. **YouTube videos**: Usually have native captions (faster)
2. **TikTok videos**: Always use Whisper (more accurate for short clips)
3. **Twitch VODs**: Use smaller models for long streams
4. **Long videos**: Use `-m tiny` or `-m base` for speed
5. **Accuracy matters**: Use `-m medium` or `-m large`
6. **Scripting**: Use `-q` for clean output in scripts
7. **Clipboard workflow**: Use `-c -q` for copy-paste workflow

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
