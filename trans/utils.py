"""Pure utility functions and constants for trans."""

import hashlib
import re
from pathlib import Path

# Supported Whisper models
WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Supported output formats
OUTPUT_FORMATS = ['txt', 'srt', 'vtt', 'json', 'all']

# Supported local file extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.aac', '.wma'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.mpeg', '.mpg'}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def get_video_id(url: str) -> str:
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
        match = re.search(r'/videos/(\d+)', url)
        if match:
            return f"tw_{match.group(1)}"
        match = re.search(r'/clip/([a-zA-Z0-9_-]+)', url) or re.search(
            r'clips\.twitch\.tv/([a-zA-Z0-9_-]+)', url
        )
        if match:
            return f"twclip_{match.group(1)}"
    # Fallback: hash the URL
    return f"hash_{hashlib.md5(url.encode()).hexdigest()[:12]}"


def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Create a safe filename from video title."""
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe)
    safe = re.sub(r'_+', '_', safe)
    return safe.strip('_')[:max_length]


def is_tiktok_url(url: str) -> bool:
    """Check if URL is a TikTok video."""
    return 'tiktok.com' in url or 'vm.tiktok.com' in url


def is_twitch_url(url: str) -> bool:
    """Check if URL is a Twitch video (VOD, clip, or stream)."""
    return 'twitch.tv' in url


def is_local_file(path: str) -> bool:
    """Check if input is a local file path (not a URL)."""
    if path.startswith(('http://', 'https://')):
        return False
    if any(domain in path for domain in ['youtube.com', 'youtu.be', 'tiktok.com', 'twitch.tv']):
        return False
    path_obj = Path(path)
    return path_obj.suffix.lower() in MEDIA_EXTENSIONS


def is_audio_file(path: str) -> bool:
    """Check if file is an audio file (vs video)."""
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_speaker_label(speaker_id: str) -> str:
    """Convert pyannote speaker ID (SPEAKER_00) to friendly label (Speaker 1)."""
    if speaker_id == 'UNKNOWN':
        return 'Unknown'
    match = re.search(r'(\d+)', speaker_id)
    if match:
        num = int(match.group(1)) + 1
        return f"Speaker {num}"
    return speaker_id


def assign_speakers_to_segments(
    transcript_segments: list, diarization_segments: list
) -> list:
    """
    Merge transcript segments with speaker labels from diarization.

    Uses overlap-based assignment: each transcript segment gets the speaker
    with the most overlap during that time range.
    """
    for t_seg in transcript_segments:
        t_start, t_end = t_seg['start'], t_seg['end']

        speaker_overlaps: dict = {}
        for d_seg in diarization_segments:
            d_start, d_end = d_seg['start'], d_seg['end']

            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                speaker = d_seg['speaker']
                speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap

        if speaker_overlaps:
            t_seg['speaker'] = max(speaker_overlaps, key=speaker_overlaps.get)
        else:
            t_seg['speaker'] = 'UNKNOWN'

    return transcript_segments
