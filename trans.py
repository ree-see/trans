#!/usr/bin/env python3
"""
Quick transcribe tool for YouTube and TikTok videos.
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


# Get the paths to executables from the venv
SCRIPT_DIR = Path(__file__).parent
YT_DLP = str(SCRIPT_DIR / '.venv' / 'bin' / 'yt-dlp')
WHISPER = str(SCRIPT_DIR / '.venv' / 'bin' / 'whisper')

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
                            cookies=None):
    """Download audio and transcribe with Whisper."""
    if not quiet:
        print("→ Using Whisper transcription...")

    audio_file = f"{output_file}.audio.mp3"

    try:
        # Download audio
        if not quiet:
            print(f"→ Downloading audio...")

        cmd = [
            YT_DLP,
            '--extract-audio',
            '--audio-format', 'mp3',
            '--output', audio_file,
        ]
        
        # Use browser impersonation for TikTok
        if is_tiktok_url(url):
            cmd.extend(['--impersonate', 'chrome-131'])
        
        # Add cookies if provided
        if cookies:
            cmd.extend(['--cookies', cookies])
        
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Transcribe with Whisper
        if not quiet:
            print(f"→ Transcribing with Whisper ({model} model)...")

        # Prepare Whisper command
        cmd = [WHISPER, audio_file, '--model', model]

        if language:
            cmd.extend(['--language', language])

        # Handle output formats
        if output_format == 'all':
            cmd.extend(['--output_format', 'all'])
        else:
            cmd.extend(['--output_format', output_format])

        cmd.extend(['--output_dir', os.path.dirname(output_file) or '.'])

        # Run Whisper (show progress unless quiet)
        if quiet:
            subprocess.run(cmd, capture_output=True, check=True)
        else:
            subprocess.run(cmd, check=True)

        # Whisper creates files with format: {audio_file_without_ext}.{format}
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

    # Try native captions first (mainly for YouTube)
    if extract_native_captions(url, output_base, args.format, args.quiet):
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
    if transcribe_with_whisper(url, output_base, args.model, args.language,
                               args.format, args.keep_audio, args.quiet, cookies):
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
        description='Transcribe YouTube or TikTok videos to text.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trans "https://youtube.com/watch?v=..."
  trans -o my_video "https://tiktok.com/..."
  trans --model small --format srt "https://youtube.com/..."
  trans --clipboard --quiet "https://youtube.com/..."
  trans "url1" "url2" "url3"  # Batch processing
        """
    )

    parser.add_argument('urls', nargs='+', metavar='URL',
                       help='YouTube or TikTok video URL(s)')

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

    args = parser.parse_args()
    
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
