"""yt-dlp Python API wrapper for downloading and caption extraction."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import yt_dlp

from .utils import is_tiktok_url


def _base_opts(url: str, cookies: str | None, quiet: bool) -> dict[str, Any]:
    opts: dict[str, Any] = {'quiet': quiet, 'no_warnings': quiet}
    if is_tiktok_url(url):
        opts['impersonate'] = 'chrome-131'
    if cookies:
        opts['cookiefile'] = str(cookies)
    return opts


def get_video_info(url: str, cookies: str | None = None, quiet: bool = False) -> dict[str, Any]:
    """Fetch video metadata without downloading."""
    opts = _base_opts(url, cookies, quiet)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info or {}
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if is_tiktok_url(url) and ('IP address is blocked' in error_msg or 'blocked' in error_msg.lower()):
            if not quiet:
                print("✗ TikTok is blocking this server's IP address.")
                print("")
                print("Workarounds:")
                print("  1. Use --cookies to provide cookies from a logged-in browser session")
                print("     Export cookies with a browser extension like 'Get cookies.txt'")
                print("")
                print("  2. Run trans from a residential IP (not a datacenter/VPS)")
                print("")
                print("  3. Use a VPN or proxy with a non-datacenter IP")
            sys.exit(1)
        if not quiet:
            print(f"✗ Error fetching video info: {error_msg}")
        sys.exit(1)


def download_audio(
    url: str,
    output_path: str,
    cookies: str | None = None,
    quiet: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Download audio from URL. Returns the final file path."""
    opts = _base_opts(url, cookies, quiet)
    opts.update({
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'outtmpl': str(output_path),
    })
    if progress_callback:
        opts['progress_hooks'] = [progress_callback]

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # yt-dlp appends .mp3 when post-processing
    final = str(output_path)
    if not final.endswith('.mp3'):
        final = final + '.mp3'
    if Path(final).exists():
        return final
    # Sometimes the path is left as-is
    return str(output_path)


def extract_native_captions(
    url: str,
    output_path: str,
    output_format: str = 'txt',
    quiet: bool = False,
) -> bool:
    """
    Attempt to extract auto-generated captions.

    Returns True if a caption file was created.
    """
    if not quiet:
        print("→ Checking for native captions...")

    sub_format = 'vtt' if output_format in ('vtt', 'all') else 'srt' if output_format == 'srt' else 'vtt'

    opts = {
        'writeautomaticsub': True,
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'skip_download': True,
        'subtitlesformat': sub_format,
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception:
        return False

    caption_file = f"{output_path}.en.{sub_format}"
    if not os.path.exists(caption_file):
        return False

    # Convert to plain text if requested
    if output_format in ('txt', 'all'):
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        text_lines = []
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith('WEBVTT')
                and not line.startswith('Kind:')
                and '-->' not in line
                and not line.isdigit()
                and not line.startswith('NOTE')
            ):
                text_lines.append(line)

        txt_output = f"{output_path}.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))

    # Clean up or rename caption file
    if output_format not in ('all', sub_format):
        os.remove(caption_file)
    else:
        final_name = f"{output_path}.{sub_format}"
        if caption_file != final_name:
            os.rename(caption_file, final_name)

    return True
