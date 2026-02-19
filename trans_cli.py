#!/usr/bin/env python3
"""
Quick transcribe tool for YouTube, TikTok, and Twitch videos.
Tries native captions first, falls back to Whisper transcription.
"""

import argparse
import os
import sys
import subprocess
import json
import re
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False


# Get the paths to executables
# First try to find them in the same environment as this script
import shutil
SCRIPT_DIR = Path(__file__).parent

def find_executable(name, venv_path=None):
    """Find an executable, preferring venv if available."""
    # Check venv first (for development installs)
    if venv_path:
        venv_bin = SCRIPT_DIR / venv_path / 'bin' / name
        if venv_bin.exists():
            return str(venv_bin)
    # Fall back to system PATH
    path = shutil.which(name)
    if path:
        return path
    # Default to just the name (let the OS find it)
    return name

YT_DLP = find_executable('yt-dlp', '.venv')
WHISPER = find_executable('whisper', '.venv')

# Cache directory
CACHE_DIR = SCRIPT_DIR / '.cache'
CACHE_DB = CACHE_DIR / 'transcripts.db'

# Supported Whisper models
WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Supported output formats
OUTPUT_FORMATS = ['txt', 'srt', 'vtt', 'json', 'all']


# ============== Cache Functions ==============

def init_cache():
    """Initialize the cache database."""
    CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            video_id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            transcript TEXT,
            format TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def get_video_id(url):
    """Extract a unique video ID from URL."""
    # YouTube
    if 'youtube.com' in url or 'youtu.be' in url:
        match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        if match:
            return f"yt_{match.group(1)}"
    # TikTok
    if 'tiktok.com' in url:
        match = re.search(r'/video/(\d+)', url)
        if match:
            return f"tt_{match.group(1)}"
    # Twitch VOD
    if 'twitch.tv' in url:
        # VOD format: twitch.tv/videos/123456789
        match = re.search(r'/videos/(\d+)', url)
        if match:
            return f"tw_{match.group(1)}"
        # Clip format: twitch.tv/channel/clip/ClipSlug or clips.twitch.tv/ClipSlug
        match = re.search(r'/clip/([a-zA-Z0-9_-]+)', url) or re.search(r'clips\.twitch\.tv/([a-zA-Z0-9_-]+)', url)
        if match:
            return f"twclip_{match.group(1)}"
    # Fallback: hash the URL
    return f"hash_{hashlib.md5(url.encode()).hexdigest()[:12]}"


def get_cached_transcript(video_id, fmt='txt'):
    """Check if transcript is cached."""
    if not CACHE_DB.exists():
        return None
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.execute(
        'SELECT transcript, title FROM transcripts WHERE video_id = ? AND format = ?',
        (video_id, fmt)
    )
    row = cursor.fetchone()
    conn.close()
    return row if row else None


def cache_transcript(video_id, url, title, transcript, fmt='txt', model=None):
    """Save transcript to cache."""
    init_cache()
    conn = sqlite3.connect(CACHE_DB)
    conn.execute('''
        INSERT OR REPLACE INTO transcripts (video_id, url, title, transcript, format, model)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (video_id, url, title, transcript, fmt, model))
    conn.commit()
    conn.close()


def copy_to_clipboard(text):
    """Copy text to system clipboard."""
    try:
        subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def sanitize_filename(title, max_length=50):
    """Create a safe filename from video title."""
    # Remove special characters, keep alphanumeric, spaces, hyphens, underscores
    safe = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces with underscores
    safe = re.sub(r'\s+', '_', safe)
    # Remove multiple underscores
    safe = re.sub(r'_+', '_', safe)
    # Trim and limit length
    return safe.strip('_')[:max_length]


def is_tiktok_url(url):
    """Check if URL is a TikTok video."""
    return 'tiktok.com' in url or 'vm.tiktok.com' in url


def is_twitch_url(url):
    """Check if URL is a Twitch video (VOD, clip, or stream)."""
    return 'twitch.tv' in url


def download_audio_with_progress(url, output_file, cookies=None, quiet=False):
    """Download audio with a progress bar."""
    cmd = [
        YT_DLP,
        '--extract-audio',
        '--audio-format', 'mp3',
        '--output', output_file,
        '--newline',  # Output progress on new lines for parsing
    ]
    
    # Use browser impersonation for TikTok
    if is_tiktok_url(url):
        cmd.extend(['--impersonate', 'chrome-131'])
    
    # Add cookies if provided
    if cookies:
        cmd.extend(['--cookies', cookies])
    
    cmd.append(url)
    
    if quiet:
        subprocess.run(cmd, capture_output=True, check=True)
        return
    
    # Parse yt-dlp progress output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               text=True, bufsize=1)
    
    pbar = None
    if HAS_TQDM:
        pbar = tqdm(total=100, desc="  Downloading", unit="%", 
                    bar_format='{desc}: {bar:30} {percentage:3.0f}%')
    
    last_percent = 0
    for line in process.stdout:
        # Parse progress like "[download]  50.0% of 5.00MiB"
        if '[download]' in line and '%' in line:
            match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if match:
                percent = float(match.group(1))
                if pbar:
                    pbar.update(percent - last_percent)
                    last_percent = percent
                elif not quiet:
                    sys.stdout.write(f"\r  Downloading: {percent:.0f}%")
                    sys.stdout.flush()
    
    if pbar:
        pbar.close()
    elif not quiet:
        print()  # Newline after progress
    
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def transcribe_audio_with_progress(audio_file, output_base, model='base', language=None,
                                   output_format='txt', quiet=False):
    """Run Whisper transcription with progress indication."""
    cmd = [WHISPER, audio_file, '--model', model]
    
    if language:
        cmd.extend(['--language', language])
    
    # Handle output formats
    if output_format == 'all':
        cmd.extend(['--output_format', 'all'])
    else:
        cmd.extend(['--output_format', output_format])
    
    cmd.extend(['--output_dir', os.path.dirname(output_base) or '.'])
    
    if quiet:
        subprocess.run(cmd, capture_output=True, check=True)
        return
    
    # Run Whisper with progress parsing
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    
    pbar = None
    if HAS_TQDM:
        pbar = tqdm(total=100, desc="  Transcribing", unit="%",
                    bar_format='{desc}: {bar:30} {percentage:3.0f}%')
    
    last_percent = 0
    for line in process.stdout:
        # Whisper outputs progress like "50%|████..." or timing info
        if '%|' in line:
            match = re.search(r'(\d+)%', line)
            if match:
                percent = int(match.group(1))
                if pbar:
                    pbar.update(percent - last_percent)
                    last_percent = percent
                elif not quiet:
                    sys.stdout.write(f"\r  Transcribing: {percent}%")
                    sys.stdout.flush()
    
    if pbar:
        pbar.close()
    elif not quiet:
        print()  # Newline after progress
    
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def transcribe_with_faster_whisper(audio_file, output_base, model='base', language=None,
                                   output_format='txt', quiet=False, diarize=False,
                                   num_speakers=None, hf_token=None):
    """Transcribe using faster-whisper (CTranslate2-based, much faster)."""
    if not HAS_FASTER_WHISPER:
        raise ImportError("faster-whisper not installed")
    
    # Map model names (faster-whisper uses same names)
    model_name = model
    
    if not quiet:
        print(f"  Loading {model_name} model...")
    
    # Load model (will download on first use)
    whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    
    # Get audio duration for progress calculation
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', audio_file],
        capture_output=True, text=True
    )
    total_duration = float(result.stdout.strip()) if result.stdout.strip() else 0
    
    if not quiet:
        print(f"  Transcribing...")
    
    # Transcribe with progress
    pbar = None
    if HAS_TQDM and total_duration > 0 and not quiet:
        pbar = tqdm(total=100, desc="  Progress", unit="%",
                    bar_format='{desc}: {bar:30} {percentage:3.0f}%')
    
    segments_list = []
    last_percent = 0
    
    segments, info = whisper_model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
        vad_filter=False,  # VAD can cause issues with some audio files
    )
    
    # Collect all segments (generator is consumed during iteration)
    for segment in segments:
        seg_data = {
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip()
        }
        segments_list.append(seg_data)
        
        # Update progress based on segment end time
        if pbar and total_duration > 0:
            percent = min(100, (segment.end / total_duration) * 100)
            pbar.update(percent - last_percent)
            last_percent = percent
        elif not quiet:
            # Print inline progress without tqdm
            if total_duration > 0:
                percent = min(100, (segment.end / total_duration) * 100)
                sys.stdout.write(f"\r  Progress: {percent:.0f}%")
                sys.stdout.flush()
    
    if pbar:
        pbar.update(100 - last_percent)  # Ensure we hit 100%
        pbar.close()
    elif not quiet:
        print()  # Newline after progress
    
    if not quiet and not segments_list:
        print("  Warning: No speech detected in audio")
    
    # Run speaker diarization if requested
    if diarize:
        try:
            diarization_segments = run_diarization(audio_file, hf_token, num_speakers, quiet)
            segments_list = assign_speakers_to_segments(segments_list, diarization_segments)
        except Exception as e:
            if not quiet:
                print(f"  Warning: Diarization failed: {e}")
                print("  Continuing without speaker labels...")
    
    # Check if we have speaker labels
    has_speakers = diarize and segments_list and 'speaker' in segments_list[0]
    
    # Generate output files
    if output_format in ['txt', 'all']:
        txt_output = f"{output_base}.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            if has_speakers:
                # Group consecutive segments by speaker for readable output
                current_speaker = None
                for seg in segments_list:
                    speaker = format_speaker_label(seg.get('speaker', 'UNKNOWN'))
                    if speaker != current_speaker:
                        if current_speaker is not None:
                            f.write('\n')
                        f.write(f"[{speaker}]\n")
                        current_speaker = speaker
                    f.write(seg['text'] + '\n')
            else:
                for seg in segments_list:
                    f.write(seg['text'] + '\n')
    
    if output_format in ['srt', 'all']:
        srt_output = f"{output_base}.srt"
        with open(srt_output, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments_list, 1):
                start_time = format_timestamp_srt(seg['start'])
                end_time = format_timestamp_srt(seg['end'])
                text = seg['text']
                if has_speakers:
                    speaker = format_speaker_label(seg.get('speaker', 'UNKNOWN'))
                    text = f"[{speaker}] {text}"
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
    
    if output_format in ['vtt', 'all']:
        vtt_output = f"{output_base}.vtt"
        with open(vtt_output, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for seg in segments_list:
                start_time = format_timestamp_vtt(seg['start'])
                end_time = format_timestamp_vtt(seg['end'])
                text = seg['text']
                if has_speakers:
                    speaker = format_speaker_label(seg.get('speaker', 'UNKNOWN'))
                    # VTT supports <v Speaker> voice tags
                    f.write(f"{start_time} --> {end_time}\n<v {speaker}>{text}\n\n")
                else:
                    f.write(f"{start_time} --> {end_time}\n{text}\n\n")
    
    if output_format in ['json', 'all']:
        json_output = f"{output_base}.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            output_data = {
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'diarization': has_speakers,
                'segments': segments_list
            }
            if has_speakers:
                # Add speaker summary
                speakers = set(seg.get('speaker') for seg in segments_list)
                output_data['speakers'] = [format_speaker_label(s) for s in sorted(speakers)]
            json.dump(output_data, f, indent=2)
    
    return segments_list


def format_timestamp_srt(seconds):
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_hf_token():
    """Get HuggingFace token from environment or cache."""
    # Check environment variable
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token
    
    # Check huggingface-cli token cache
    token_path = Path.home() / '.cache' / 'huggingface' / 'token'
    if token_path.exists():
        return token_path.read_text().strip()
    
    return None


def run_diarization(audio_file, hf_token, num_speakers=None, quiet=False):
    """
    Run speaker diarization using pyannote-audio.
    
    Returns a list of (start, end, speaker) tuples.
    """
    if not HAS_PYANNOTE:
        raise ImportError("pyannote-audio not installed. Run: pip install pyannote-audio")
    
    if not hf_token:
        raise ValueError(
            "HuggingFace token required for speaker diarization.\n"
            "1. Create a token at https://huggingface.co/settings/tokens\n"
            "2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Set HF_TOKEN environment variable or run: huggingface-cli login"
        )
    
    if not quiet:
        print("  Loading diarization model...")
    
    # Load the diarization pipeline
    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    if not quiet:
        print("  Running speaker diarization...")
    
    # Run diarization
    diarization_args = {}
    if num_speakers:
        diarization_args['num_speakers'] = num_speakers
    
    diarization = pipeline(audio_file, **diarization_args)
    
    # Extract speaker segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    
    if not quiet:
        unique_speakers = len(set(s['speaker'] for s in segments))
        print(f"  Detected {unique_speakers} speaker(s)")
    
    return segments


def assign_speakers_to_segments(transcript_segments, diarization_segments):
    """
    Merge transcript segments with speaker labels from diarization.
    
    Uses overlap-based assignment: each transcript segment gets the speaker
    with the most overlap during that time range.
    """
    for t_seg in transcript_segments:
        t_start, t_end = t_seg['start'], t_seg['end']
        
        # Find overlapping diarization segments
        speaker_overlaps = {}
        for d_seg in diarization_segments:
            d_start, d_end = d_seg['start'], d_seg['end']
            
            # Calculate overlap
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                speaker = d_seg['speaker']
                speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap
        
        # Assign speaker with most overlap
        if speaker_overlaps:
            t_seg['speaker'] = max(speaker_overlaps, key=speaker_overlaps.get)
        else:
            t_seg['speaker'] = 'UNKNOWN'
    
    return transcript_segments


def format_speaker_label(speaker_id):
    """Convert pyannote speaker ID (SPEAKER_00) to friendly label (Speaker 1)."""
    if speaker_id == 'UNKNOWN':
        return 'Unknown'
    # Extract number from SPEAKER_XX format
    match = re.search(r'(\d+)', speaker_id)
    if match:
        num = int(match.group(1)) + 1  # 0-indexed to 1-indexed
        return f"Speaker {num}"
    return speaker_id


def format_timestamp_vtt(seconds):
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def get_video_info(url, quiet=False, cookies=None):
    """Get video information using yt-dlp."""
    try:
        cmd = [YT_DLP, '--dump-json', '--no-download']
        
        # Use browser impersonation for TikTok
        if is_tiktok_url(url):
            cmd.extend(['--impersonate', 'chrome-131'])
        
        # Add cookies if provided
        if cookies:
            cmd.extend(['--cookies', cookies])
        
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        
        # Check for TikTok IP block
        if is_tiktok_url(url) and ('IP address is blocked' in error_msg or 'blocked' in error_msg.lower()):
            if not quiet:
                print("✗ TikTok is blocking this server's IP address.")
                print("")
                print("Workarounds:")
                print("  1. Use --cookies to provide cookies from a logged-in browser session")
                print("     Export cookies with a browser extension like 'Get cookies.txt'")
                print("     Then: trans --cookies cookies.txt 'TIKTOK_URL'")
                print("")
                print("  2. Run trans from a residential IP (not a datacenter/VPS)")
                print("")
                print("  3. Use a VPN or proxy with a non-datacenter IP")
            sys.exit(1)
        
        if not quiet:
            print(f"✗ Error fetching video info: {error_msg}")
        sys.exit(1)
    except json.JSONDecodeError:
        if not quiet:
            print("✗ Error parsing video information")
        sys.exit(1)


def extract_native_captions(url, output_file, output_format='txt', quiet=False):
    """Try to extract native captions from YouTube."""
    if not quiet:
        print("→ Checking for native captions...")

    try:
        # Determine subtitle format based on output format
        sub_format = 'vtt' if output_format in ['vtt', 'all'] else 'srt' if output_format == 'srt' else 'vtt'

        cmd = [
            YT_DLP,
            '--write-auto-subs',
            '--write-subs',
            '--sub-lang', 'en',
            '--skip-download',
            '--sub-format', sub_format,
            '--output', output_file,
            url
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # Check if caption file was created
        caption_file = f"{output_file}.en.{sub_format}"

        if os.path.exists(caption_file):
            # If user wants txt or all, convert to plain text
            if output_format in ['txt', 'all']:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Remove formatting
                text_lines = []
                for line in lines:
                    line = line.strip()
                    if (line and
                        not line.startswith('WEBVTT') and
                        not line.startswith('Kind:') and
                        not '-->' in line and
                        not line.isdigit() and
                        not line.startswith('NOTE')):
                        text_lines.append(line)

                # Write plain text
                txt_output = f"{output_file}.txt"
                with open(txt_output, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text_lines))

            # Keep the original format file if requested
            if output_format not in ['all', sub_format]:
                os.remove(caption_file)
            else:
                # Rename to standard extension
                final_name = f"{output_file}.{sub_format}"
                if caption_file != final_name:
                    os.rename(caption_file, final_name)

            return True

        return False
    except subprocess.CalledProcessError:
        return False


def transcribe_with_whisper(url, output_file, model='base', language=None,
                            output_format='txt', keep_audio=False, quiet=False,
                            cookies=None, diarize=False, num_speakers=None):
    """Download audio and transcribe with Whisper."""
    if not quiet:
        backend = "faster-whisper" if HAS_FASTER_WHISPER else "openai-whisper CLI"
        features = []
        if diarize:
            features.append("speaker diarization")
        feature_str = f" + {', '.join(features)}" if features else ""
        print(f"→ Using Whisper transcription ({backend}{feature_str})...")

    audio_file = f"{output_file}.audio.mp3"

    try:
        # Download audio with progress bar
        if not quiet:
            print(f"→ Downloading audio...")
        
        download_audio_with_progress(url, audio_file, cookies, quiet)

        # Transcribe with Whisper
        if not quiet:
            print(f"→ Transcribing with Whisper ({model} model)...")

        if HAS_FASTER_WHISPER:
            # Use faster-whisper (Python API, much faster)
            hf_token = get_hf_token() if diarize else None
            transcribe_with_faster_whisper(audio_file, output_file, model, language,
                                           output_format, quiet, diarize, num_speakers,
                                           hf_token)
        else:
            # Fall back to openai-whisper CLI
            transcribe_audio_with_progress(audio_file, output_file, model, language,
                                           output_format, quiet)

            # Whisper CLI creates files with format: {audio_file_without_ext}.{format}
            base_name = audio_file.replace('.mp3', '')

            # Move output files to desired location
            formats_to_move = [output_format] if output_format != 'all' else ['txt', 'srt', 'vtt', 'json', 'tsv']

            for fmt in formats_to_move:
                whisper_output = f"{base_name}.{fmt}"
                final_output = f"{output_file}.{fmt}"

                if os.path.exists(whisper_output) and whisper_output != final_output:
                    os.rename(whisper_output, final_output)

        # Clean up audio file unless requested to keep
        if not keep_audio and os.path.exists(audio_file):
            os.remove(audio_file)
        elif keep_audio and not quiet:
            print(f"  Audio saved: {audio_file}")

        return True

    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"✗ Error during transcription: {e}")
        # Clean up audio file on error
        if os.path.exists(audio_file):
            os.remove(audio_file)
        return False
    except Exception as e:
        if not quiet:
            print(f"✗ Error during transcription: {e}")
        # Clean up audio file on error
        if os.path.exists(audio_file):
            os.remove(audio_file)
        return False


def process_url(url, args):
    """Process a single URL."""
    # Get video ID for caching
    video_id = get_video_id(url)
    
    # Get cookies path if provided
    cookies = getattr(args, 'cookies', None)
    
    # Check cache first (unless --no-cache)
    if not getattr(args, 'no_cache', False):
        cached = get_cached_transcript(video_id, args.format if args.format != 'all' else 'txt')
        if cached:
            transcript, title = cached
            if not args.quiet:
                print(f"\n💾 Using cached transcript for: {title}")
            
            # Determine output filename
            if args.output:
                output_base = args.output
            else:
                safe_title = sanitize_filename(title)
                if args.timestamp:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_base = f"{safe_title}_{timestamp}"
                else:
                    output_base = safe_title
            
            # Write cached transcript to file
            output_file = f"{output_base}.{args.format if args.format != 'all' else 'txt'}"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            if not args.quiet:
                print(f"✓ Transcript written to {output_file}")
            
            # Copy to clipboard if requested
            if args.clipboard:
                if copy_to_clipboard(transcript):
                    if not args.quiet:
                        print(f"📋 Copied to clipboard")
            
            return True
    
    # Get video info for title
    info = get_video_info(url, args.quiet, cookies)
    video_title = info.get('title', 'video')
    duration = info.get('duration', 0)

    # Determine output filename
    if args.output:
        output_base = args.output
    else:
        # Create safe filename from video title
        safe_title = sanitize_filename(video_title)
        if args.timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_base = f"{safe_title}_{timestamp}"
        else:
            output_base = safe_title

    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"📹 {video_title}")
        if duration:
            mins, secs = divmod(duration, 60)
            print(f"⏱️  Duration: {int(mins)}:{int(secs):02d}")
        print(f"{'='*60}\n")

    # Try native captions first (mainly for YouTube), unless --force-whisper
    if not getattr(args, 'force_whisper', False) and extract_native_captions(url, output_base, args.format, args.quiet):
        output_files = []
        if args.format == 'all':
            output_files = [f"{output_base}.txt", f"{output_base}.vtt"]
        else:
            output_files = [f"{output_base}.{args.format}"]

        if not args.quiet:
            print(f"\n✓ Transcription complete (native captions)")
            for f in output_files:
                if os.path.exists(f):
                    print(f"  → {f}")

        # Cache the transcript
        if not getattr(args, 'no_cache', False):
            txt_file = f"{output_base}.txt"
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    cache_transcript(video_id, url, video_title, f.read(), 'txt')
                if not args.quiet:
                    print(f"💾 Cached for future use")

        # Copy to clipboard if requested
        if args.clipboard and os.path.exists(f"{output_base}.txt"):
            with open(f"{output_base}.txt", 'r') as f:
                if copy_to_clipboard(f.read()):
                    if not args.quiet:
                        print(f"📋 Copied to clipboard")

        return True

    # Fall back to Whisper
    diarize = getattr(args, 'diarize', False)
    num_speakers = getattr(args, 'num_speakers', None)
    
    if transcribe_with_whisper(url, output_base, args.model, args.language,
                               args.format, args.keep_audio, args.quiet, cookies,
                               diarize, num_speakers):
        output_files = []
        if args.format == 'all':
            for ext in ['txt', 'srt', 'vtt', 'json', 'tsv']:
                if os.path.exists(f"{output_base}.{ext}"):
                    output_files.append(f"{output_base}.{ext}")
        else:
            output_files = [f"{output_base}.{args.format}"]

        if not args.quiet:
            print(f"\n✓ Transcription complete (Whisper)")
            for f in output_files:
                if os.path.exists(f):
                    size = os.path.getsize(f)
                    print(f"  → {f} ({size} bytes)")

        # Cache the transcript
        if not getattr(args, 'no_cache', False):
            txt_file = f"{output_base}.txt"
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    cache_transcript(video_id, url, video_title, f.read(), 'txt', args.model)
                if not args.quiet:
                    print(f"💾 Cached for future use")

        # Copy to clipboard if requested
        if args.clipboard and os.path.exists(f"{output_base}.txt"):
            with open(f"{output_base}.txt", 'r') as f:
                if copy_to_clipboard(f.read()):
                    if not args.quiet:
                        print(f"📋 Copied to clipboard")

        return True

    if not args.quiet:
        print("✗ Transcription failed")
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe YouTube, TikTok, or Twitch videos to text.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trans "https://youtube.com/watch?v=..."
  trans -o my_video "https://tiktok.com/..."
  trans "https://twitch.tv/videos/123456789"  # Twitch VOD
  trans "https://clips.twitch.tv/FunnyClipName"  # Twitch clip
  trans --model small --format srt "https://youtube.com/..."
  trans --clipboard --quiet "https://youtube.com/..."
  trans "url1" "url2" "url3"  # Batch processing
        """
    )

    parser.add_argument('urls', nargs='+', metavar='URL',
                       help='YouTube, TikTok, or Twitch video URL(s)')

    parser.add_argument('-o', '--output',
                       help='Output file path (without extension). Only for single URL.')

    parser.add_argument('-m', '--model',
                       choices=WHISPER_MODELS,
                       default='base',
                       help='Whisper model size (default: base)')

    parser.add_argument('-l', '--language',
                       help='Specify language (e.g., en, es, fr). Auto-detect if not set.')

    parser.add_argument('-f', '--format',
                       choices=OUTPUT_FORMATS,
                       default='txt',
                       help='Output format (default: txt)')

    parser.add_argument('-c', '--clipboard',
                       action='store_true',
                       help='Copy transcript to clipboard (txt format only)')

    parser.add_argument('-k', '--keep-audio',
                       action='store_true',
                       help='Keep downloaded audio file')

    parser.add_argument('-t', '--timestamp',
                       action='store_true',
                       help='Add timestamp to output filename')

    parser.add_argument('-q', '--quiet',
                       action='store_true',
                       help='Minimal output (errors only)')

    parser.add_argument('--cookies',
                       help='Path to cookies.txt file (for authenticated downloads, e.g. TikTok)')

    parser.add_argument('--no-cache',
                       action='store_true',
                       help='Skip cache lookup and force fresh transcription')

    parser.add_argument('--clear-cache',
                       action='store_true',
                       help='Clear the transcript cache and exit')

    parser.add_argument('--force-whisper',
                       action='store_true',
                       help='Skip native captions and always use Whisper')

    parser.add_argument('--diarize', '-d',
                       action='store_true',
                       help='Enable speaker diarization (who said what). Requires pyannote-audio and HF token.')

    parser.add_argument('--num-speakers',
                       type=int,
                       help='Number of speakers (helps diarization accuracy). Auto-detect if not set.')

    args = parser.parse_args()
    
    # Check diarization requirements
    if args.diarize:
        if not HAS_PYANNOTE:
            print("✗ Speaker diarization requires pyannote-audio.")
            print("")
            print("Install with:")
            print("  pip install pyannote-audio")
            print("")
            print("You'll also need a HuggingFace token:")
            print("  1. Create token at https://huggingface.co/settings/tokens")
            print("  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  3. Set HF_TOKEN env var or run: huggingface-cli login")
            sys.exit(1)
        
        if not HAS_FASTER_WHISPER:
            print("✗ Speaker diarization requires faster-whisper (not openai-whisper CLI).")
            print("")
            print("Install with:")
            print("  pip install faster-whisper")
            sys.exit(1)
        
        hf_token = get_hf_token()
        if not hf_token:
            print("✗ Speaker diarization requires a HuggingFace token.")
            print("")
            print("Setup:")
            print("  1. Create token at https://huggingface.co/settings/tokens")
            print("  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  3. Set HF_TOKEN environment variable")
            print("     OR run: huggingface-cli login")
            sys.exit(1)
    
    # Handle cache clearing
    if args.clear_cache:
        if CACHE_DB.exists():
            os.remove(CACHE_DB)
            print("✓ Cache cleared")
        else:
            print("Cache is already empty")
        sys.exit(0)

    # Validate arguments
    if args.output and len(args.urls) > 1:
        print("✗ Error: -o/--output can only be used with a single URL")
        sys.exit(1)

    # Process URLs
    success_count = 0
    fail_count = 0

    for url in args.urls:
        try:
            if process_url(url, args):
                success_count += 1
            else:
                fail_count += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            sys.exit(1)
        except Exception as e:
            if not args.quiet:
                print(f"✗ Unexpected error: {e}")
            fail_count += 1

    # Summary for batch processing
    if len(args.urls) > 1 and not args.quiet:
        print(f"\n{'='*60}")
        print(f"Summary: {success_count} succeeded, {fail_count} failed")
        print(f"{'='*60}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()
