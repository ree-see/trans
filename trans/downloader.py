"""yt-dlp Python API wrapper for downloading and caption extraction."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, NamedTuple

import yt_dlp
from yt_dlp.networking.impersonate import ImpersonateTarget

from .utils import is_tiktok_url


class NativeCaptureResult(NamedTuple):
    """Output of `extract_native_captions`.

    Two channels: the files on disk (txt/vtt/srt as requested) and the
    parsed segment list. Segments are populated when a VTT was downloaded
    and parsed successfully; they're cached so subsequent runs can render
    any format from canonical segments instead of re-fetching.

    Truthiness rule: callers MUST check `result.files` (not `result`) to
    decide whether the native-caption branch succeeded. The NamedTuple
    itself is always truthy.
    """

    files: list[Path]
    segments: list[dict[str, Any]]


_VTT_TIMING_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
)
_VTT_INLINE_TAG_RE = re.compile(r"<[^>]+>")


def _vtt_ts(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(vtt_text: str) -> list[dict[str, Any]]:
    """Parse a WebVTT caption file into canonical segments.

    Strips WEBVTT/Kind/Language/NOTE header lines, ignores cue identifier
    lines, and removes inline timing tags like `<00:00:01.000>` or
    `<c.colorE5E5E5>...</c>` from the text. Returns the same
    `{"start", "end", "text"}` shape that `TranscriptionEngine.transcribe`
    produces, so cache hits render through `formatter.write_output`
    without a separate code path.

    Garbage input returns `[]` rather than raising — callers gate cache
    writes on the result being non-empty.
    """
    if not vtt_text or "-->" not in vtt_text:
        return []
    # Drop CRs so blank-line splitting is uniform.
    blocks = vtt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
    segments: list[dict[str, Any]] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        # Skip the WEBVTT header / Kind / Language / NOTE blocks.
        first = lines[0].strip()
        if (
            first.startswith("WEBVTT")
            or first.startswith("Kind:")
            or first.startswith("Language:")
            or first.startswith("NOTE")
        ):
            continue
        # Find the timing line — sometimes preceded by a cue identifier.
        timing_idx = -1
        for i, ln in enumerate(lines):
            if "-->" in ln:
                timing_idx = i
                break
        if timing_idx < 0:
            continue
        m = _VTT_TIMING_RE.match(lines[timing_idx])
        if not m:
            continue
        start = _vtt_ts(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _vtt_ts(m.group(5), m.group(6), m.group(7), m.group(8))
        text_lines = lines[timing_idx + 1 :]
        if not text_lines:
            continue
        text = " ".join(_VTT_INLINE_TAG_RE.sub("", ln).strip() for ln in text_lines)
        text = text.strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


_BACKEND_HINT_SHOWN = False


class DownloaderError(Exception):
    """Base class for downloader-layer errors users care about.

    Carries a one-sentence message in ``str(exc)``. The CLI layer is
    responsible for any multi-line workaround help (so library callers
    that aren't the CLI can format their own UX).
    """


class TikTokIPBlockedError(DownloaderError):
    """TikTok refused this datacenter/VPS IP. Cookies or residential IP needed."""


def _has_impersonate_backend() -> bool:
    """Return True if an impersonation backend (curl_cffi) is importable.

    yt-dlp's impersonation requires a registered handler; today the only
    one shipped is CurlCFFI, which lives in the [tiktok] extra. If the
    import fails we cannot impersonate, so the caller must omit the
    'impersonate' option entirely rather than letting yt-dlp raise
    "Impersonate target ... is not available" at download time.
    """
    try:
        import curl_cffi  # noqa: F401
    except ImportError:
        return False
    return True


def _base_opts(url: str, cookies: str | None, quiet: bool) -> dict[str, Any]:
    global _BACKEND_HINT_SHOWN
    opts: dict[str, Any] = {"quiet": quiet, "no_warnings": quiet}
    if is_tiktok_url(url):
        if _has_impersonate_backend():
            # yt-dlp >= 2024.07.01 requires an ImpersonateTarget instance here;
            # passing a raw string crashes inside is_supported_target().
            opts["impersonate"] = ImpersonateTarget.from_str("chrome-131")
        elif not quiet and not _BACKEND_HINT_SHOWN:
            # TikTok works without impersonation (just more bot-detectable),
            # so degrade silently — but tell the user once how to fix it.
            # Check-then-set is intentionally lock-free: under cli.py's
            # ThreadPoolExecutor a batch of TikTok URLs without curl_cffi may
            # race and print the hint up to N times. That's cosmetic, not
            # incorrect; a lock here would be overkill for warning output.
            print(
                "Warning: TikTok impersonation backend not installed — "
                "downloads may be blocked or rate-limited."
            )
            print("    Install with: pip install 'boswell[tiktok]'")
            _BACKEND_HINT_SHOWN = True
    if cookies:
        opts["cookiefile"] = str(cookies)
    return opts


def get_video_info(url: str, cookies: str | None = None, quiet: bool = False) -> dict[str, Any]:
    """Fetch video metadata without downloading.

    Raises:
        TikTokIPBlockedError: TikTok blocked this server's IP. The CLI
            renders the multi-line workaround block; library callers can
            render their own UX from the exception type.
        DownloaderError: Any other yt-dlp DownloadError.
    """
    opts = _base_opts(url, cookies, quiet)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info or {}
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if is_tiktok_url(url) and (
            "IP address is blocked" in error_msg or "blocked" in error_msg.lower()
        ):
            raise TikTokIPBlockedError("TikTok is blocking this server's IP address.") from e
        raise DownloaderError(f"Error fetching video info: {error_msg}") from e


def download_audio(
    url: str,
    output_path: str,
    cookies: str | None = None,
    quiet: bool = False,
) -> str:
    """Download audio from URL. Returns the final file path."""
    opts = _base_opts(url, cookies, quiet)
    opts.update(
        {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "outtmpl": str(output_path),
        }
    )

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # yt-dlp appends .mp3 when post-processing
    final = str(output_path)
    if not final.endswith(".mp3"):
        final = final + ".mp3"
    if Path(final).exists():
        return final
    # Sometimes the path is left as-is
    return str(output_path)


def extract_native_captions(
    url: str,
    output_path: str,
    output_format: str = "txt",
    quiet: bool = False,
) -> NativeCaptureResult:
    """
    Attempt to extract auto-generated captions.

    Returns a `NativeCaptureResult(files, segments)`:

    * `files`: caption files created on disk (txt/vtt/srt as requested).
      An empty list signals the caller should fall through to Whisper —
      either because yt-dlp produced no captions, or because
      ``output_format`` isn't one of ``txt``/``vtt``/``srt``/``all``
      (today: ``json``, plus any future format added to
      ``OUTPUT_FORMATS`` that this function doesn't explicitly handle).
    * `segments`: parsed VTT segments in the canonical
      ``{"start", "end", "text"}`` shape. The CLI caches these with
      ``source="native"`` so subsequent runs can render any format
      without re-fetching.

    Truthiness rule: callers MUST check ``result.files`` (not
    ``result``) to decide whether the branch succeeded — a NamedTuple
    with empty fields is still truthy.

    Note: ``output_format == "all"`` returns ``[txt, vtt]`` — the two
    formats derivable from yt-dlp's auto-captions — NOT the full
    four-file set that ``formatter.write_output`` produces on the
    Whisper path. ``--format all`` on a native-captions hit is
    intentionally faster-and-narrower; pass ``--force-whisper`` to
    route through ``write_output`` for srt/json too. Cache hits on a
    native-source row DO render via ``write_output`` and produce every
    requested format.
    """
    if not quiet:
        print("→ Checking for native captions...")

    sub_format = (
        "vtt" if output_format in ("vtt", "all") else "srt" if output_format == "srt" else "vtt"
    )

    opts = {
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": ["en"],
        "skip_download": True,
        "subtitlesformat": sub_format,
        "outtmpl": str(output_path),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception:
        return NativeCaptureResult([], [])

    caption_file = f"{output_path}.en.{sub_format}"
    if not os.path.exists(caption_file):
        return NativeCaptureResult([], [])

    created: list[Path] = []
    segments: list[dict[str, Any]] = []

    # Read the caption file once. Parse VTT into canonical segments so
    # the CLI can cache them and render any format on a hit; SRT goes
    # straight to disk (yt-dlp owns the format).
    with open(caption_file, "r", encoding="utf-8") as f:
        raw = f.read()
    if sub_format == "vtt":
        segments = parse_vtt(raw)

    # Convert to plain text if requested
    if output_format in ("txt", "all"):
        text_lines = []
        for line in raw.splitlines():
            line = line.strip()
            if (
                line
                and not line.startswith("WEBVTT")
                and not line.startswith("Kind:")
                and "-->" not in line
                and not line.isdigit()
                and not line.startswith("NOTE")
            ):
                text_lines.append(line)

        txt_output = f"{output_path}.txt"
        with open(txt_output, "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines))
        created.append(Path(txt_output))

    # Clean up or rename caption file
    if output_format not in ("all", sub_format):
        os.remove(caption_file)
    else:
        final_name = f"{output_path}.{sub_format}"
        if caption_file != final_name:
            os.rename(caption_file, final_name)
        created.append(Path(final_name))

    return NativeCaptureResult(created, segments)
