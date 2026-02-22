"""Typer CLI entry point for trans."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import typer

from . import __version__
from .cache import CacheManager
from .config import Config, SETTABLE_KEYS, load_config, set_config_value, get_config_path
from .diarizer import HAS_PYANNOTE, get_hf_token, run_diarization
from .downloader import download_audio, extract_native_captions, get_video_info
from .formatter import write_output
from .transcriber import HAS_FASTER_WHISPER, TranscriptionEngine, extract_audio_from_video
from .utils import (
    OUTPUT_FORMATS,
    WHISPER_MODELS,
    get_video_id,
    is_local_file,
    is_audio_file,
    sanitize_filename,
)

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

app = typer.Typer(
    name="trans",
    help="Transcribe YouTube, TikTok, Twitch videos and local audio/video files.",
    add_completion=False,
    invoke_without_command=True,
)
cache_app = typer.Typer(help="Manage the transcript cache.")
config_app = typer.Typer(help="Manage persistent configuration.")
app.add_typer(cache_app, name="cache")
app.add_typer(config_app, name="config")


@app.callback()
def _app_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", is_eager=True, help="Show version and exit."
    ),
) -> None:
    """trans — transcribe videos and audio files to text."""
    if version:
        typer.echo(f"trans {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_to_clipboard(text: str, quiet: bool) -> None:
    if not HAS_PYPERCLIP:
        if not quiet:
            typer.echo("⚠️  pyperclip not installed — clipboard copy skipped")
        return
    try:
        pyperclip.copy(text)
        if not quiet:
            typer.echo("📋 Copied to clipboard")
    except Exception as e:
        if not quiet:
            typer.echo(f"⚠️  Clipboard copy failed: {e}")


def _output_base(
    title: str,
    output: str | None,
    output_dir: Path | None,
    timestamp: bool,
    config: Config,
) -> str:
    if output:
        return output
    safe = sanitize_filename(title)
    if timestamp:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe = f"{safe}_{ts}"
    if output_dir:
        return str(output_dir / safe)
    if config.output_dir:
        return str(Path(config.output_dir) / safe)
    return safe


def _resolve(cli_val, config_val, default):
    """Return cli_val if set, else config_val, else default."""
    if cli_val is not None:
        return cli_val
    return config_val if config_val else default


# ---------------------------------------------------------------------------
# URL processing
# ---------------------------------------------------------------------------

def _process_url(
    url: str,
    *,
    output: str | None,
    output_dir: Path | None,
    model: str,
    language: str | None,
    fmt: str,
    clipboard: bool,
    keep_audio: bool,
    timestamp: bool,
    quiet: bool,
    cookies: Path | None,
    no_cache: bool,
    force_whisper: bool,
    diarize: bool,
    num_speakers: int | None,
    engine: TranscriptionEngine,
    cache: CacheManager,
    config: Config,
) -> bool:
    video_id = get_video_id(url)
    cookies_str = str(cookies) if cookies else None

    # Cache lookup
    if not no_cache:
        cached = cache.get(video_id, fmt if fmt != 'all' else 'txt', config.cache.ttl_days)
        if cached:
            transcript, title = cached
            if not quiet:
                typer.echo(f"\n💾 Using cached transcript for: {title}")
            out_base = _output_base(title, output, output_dir, timestamp, config)
            out_fmt = fmt if fmt != 'all' else 'txt'
            out_file = f"{out_base}.{out_fmt}"
            Path(out_file).write_text(transcript, encoding='utf-8')
            if not quiet:
                typer.echo(f"✓ Transcript written to {out_file}")
            if clipboard:
                _copy_to_clipboard(transcript, quiet)
            return True

    # Fetch metadata
    info = get_video_info(url, cookies=cookies_str, quiet=quiet)
    video_title = info.get('title', 'video')
    duration = info.get('duration', 0)
    out_base = _output_base(video_title, output, output_dir, timestamp, config)

    if not quiet:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"📹 {video_title}")
        if duration:
            mins, secs = divmod(duration, 60)
            typer.echo(f"⏱️  Duration: {int(mins)}:{int(secs):02d}")
        typer.echo(f"{'='*60}\n")

    # Try native captions first
    if not force_whisper and extract_native_captions(url, out_base, fmt, quiet):
        if not quiet:
            typer.echo(f"\n✓ Transcription complete (native captions)")
            _print_output_files(out_base, fmt, ['txt', 'vtt'])
        if not no_cache:
            txt_path = Path(f"{out_base}.txt")
            if txt_path.exists():
                cache.put(video_id, url, video_title, txt_path.read_text(encoding='utf-8'), 'txt')
                if not quiet:
                    typer.echo("💾 Cached for future use")
        if clipboard:
            txt_path = Path(f"{out_base}.txt")
            if txt_path.exists():
                _copy_to_clipboard(txt_path.read_text(encoding='utf-8'), quiet)
        return True

    # Download + Whisper
    audio_file = f"{out_base}.audio.mp3"
    try:
        if not quiet:
            typer.echo("→ Downloading audio...")
        final_audio = download_audio(url, audio_file, cookies=cookies_str, quiet=quiet)

        segments, info_dict = engine.transcribe(final_audio, language=language or None, quiet=quiet)

        if diarize:
            hf_token = get_hf_token()
            try:
                diar_segs = run_diarization(final_audio, hf_token, num_speakers, quiet)
                from .utils import assign_speakers_to_segments
                segments = assign_speakers_to_segments(segments, diar_segs)
            except Exception as e:
                if not quiet:
                    typer.echo(f"  Warning: Diarization failed: {e}")
                    typer.echo("  Continuing without speaker labels...")

        created = write_output(segments, out_base, fmt, info=info_dict, diarized=diarize)

        if not keep_audio and Path(final_audio).exists():
            os.remove(final_audio)
        elif keep_audio and not quiet:
            typer.echo(f"  Audio saved: {final_audio}")

        if not quiet:
            typer.echo(f"\n✓ Transcription complete (Whisper)")
            for p in created:
                if p.exists():
                    typer.echo(f"  → {p} ({p.stat().st_size} bytes)")

        if not no_cache:
            txt_path = Path(f"{out_base}.txt")
            if txt_path.exists():
                cache.put(video_id, url, video_title, txt_path.read_text(encoding='utf-8'), 'txt', model)
                if not quiet:
                    typer.echo("💾 Cached for future use")

        if clipboard:
            txt_path = Path(f"{out_base}.txt")
            if txt_path.exists():
                _copy_to_clipboard(txt_path.read_text(encoding='utf-8'), quiet)

        return True

    except Exception as e:
        if not quiet:
            typer.echo(f"✗ Error during transcription: {e}")
        if Path(audio_file).exists():
            os.remove(audio_file)
        return False


# ---------------------------------------------------------------------------
# Local file processing
# ---------------------------------------------------------------------------

def _process_local(
    filepath: str,
    *,
    output: str | None,
    output_dir: Path | None,
    model: str,
    language: str | None,
    fmt: str,
    clipboard: bool,
    timestamp: bool,
    quiet: bool,
    diarize: bool,
    num_speakers: int | None,
    engine: TranscriptionEngine,
    config: Config,
) -> bool:
    fp = Path(filepath)
    if not fp.exists():
        typer.echo(f"✗ File not found: {fp}")
        return False

    title = fp.stem
    out_base = _output_base(title, output, output_dir, timestamp, config)

    # Get duration
    from .transcriber import get_file_duration
    duration = get_file_duration(str(fp))

    if not quiet:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"📁 {fp.name}")
        if duration:
            mins, secs = divmod(duration, 60)
            hours = int(mins) // 60
            mins = int(mins) % 60
            if hours > 0:
                typer.echo(f"⏱️  Duration: {hours}:{mins:02d}:{int(secs):02d}")
            else:
                typer.echo(f"⏱️  Duration: {int(mins)}:{int(secs):02d}")
        typer.echo(f"{'='*60}\n")

    audio_file = str(fp)
    temp_audio = None

    if not is_audio_file(str(fp)):
        temp_audio = f"{out_base}.temp_audio.mp3"
        if not extract_audio_from_video(str(fp), temp_audio, quiet):
            return False
        audio_file = temp_audio

    try:
        segments, info_dict = engine.transcribe(audio_file, language=language or None, quiet=quiet)

        if diarize:
            hf_token = get_hf_token()
            try:
                diar_segs = run_diarization(audio_file, hf_token, num_speakers, quiet)
                from .utils import assign_speakers_to_segments
                segments = assign_speakers_to_segments(segments, diar_segs)
            except Exception as e:
                if not quiet:
                    typer.echo(f"  Warning: Diarization failed: {e}")

        created = write_output(segments, out_base, fmt, info=info_dict, diarized=diarize)

        if not quiet:
            typer.echo(f"\n✓ Transcription complete")
            for p in created:
                if p.exists():
                    typer.echo(f"  → {p} ({p.stat().st_size} bytes)")

        if clipboard:
            txt_path = Path(f"{out_base}.txt")
            if txt_path.exists():
                _copy_to_clipboard(txt_path.read_text(encoding='utf-8'), quiet)

        return True

    except Exception as e:
        if not quiet:
            typer.echo(f"✗ Error during transcription: {e}")
        return False
    finally:
        if temp_audio and Path(temp_audio).exists():
            os.remove(temp_audio)


def _print_output_files(out_base: str, fmt: str, extras: list[str]) -> None:
    formats = ([fmt] if fmt != 'all' else extras)
    for ext in formats:
        p = Path(f"{out_base}.{ext}")
        if p.exists():
            typer.echo(f"  → {p}")


# ---------------------------------------------------------------------------
# Main transcribe command
# ---------------------------------------------------------------------------

@app.command()
def transcribe(
    inputs: list[str] = typer.Argument(..., help="Video/audio URL(s) or local file path(s)"),
    output: str = typer.Option(None, "-o", "--output", help="Output base path (no extension). Single input only."),
    output_dir: Path = typer.Option(None, "--output-dir", help="Directory for output files."),
    model: str = typer.Option(None, "-m", "--model", help=f"Whisper model: {', '.join(WHISPER_MODELS)}"),
    language: str = typer.Option(None, "-l", "--language", help="Language code (e.g. en, es). Auto-detect if unset."),
    format: str = typer.Option(None, "-f", "--format", help=f"Output format: {', '.join(OUTPUT_FORMATS)}"),
    clipboard: bool = typer.Option(None, "-c", "--clipboard", help="Copy transcript to clipboard."),
    keep_audio: bool = typer.Option(False, "-k", "--keep-audio", help="Keep downloaded audio file."),
    timestamp: bool = typer.Option(False, "-t", "--timestamp", help="Add timestamp to output filename."),
    quiet: bool = typer.Option(None, "-q", "--quiet", help="Minimal output (errors only)."),
    cookies: Path = typer.Option(None, "--cookies", help="Path to cookies.txt for authenticated downloads."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip cache lookup and force fresh transcription."),
    force_whisper: bool = typer.Option(False, "--force-whisper", help="Skip native captions, always use Whisper."),
    diarize: bool = typer.Option(False, "-d", "--diarize", help="Enable speaker diarization (requires pyannote-audio)."),
    num_speakers: int = typer.Option(None, "--num-speakers", help="Number of speakers (helps diarization accuracy)."),
) -> None:
    """Transcribe video/audio URLs or local files to text."""

    cfg = load_config()

    # Resolve options: CLI > config > hardcoded default
    eff_model = _resolve(model, cfg.model, 'base')
    eff_format = _resolve(format, cfg.format, 'txt')
    eff_language = _resolve(language, cfg.language, None)
    eff_clipboard = clipboard if clipboard is not None else cfg.clipboard
    eff_quiet = quiet if quiet is not None else cfg.quiet

    # Validate
    if eff_model not in WHISPER_MODELS:
        typer.echo(f"✗ Invalid model '{eff_model}'. Choose from: {', '.join(WHISPER_MODELS)}")
        raise typer.Exit(1)
    if eff_format not in OUTPUT_FORMATS:
        typer.echo(f"✗ Invalid format '{eff_format}'. Choose from: {', '.join(OUTPUT_FORMATS)}")
        raise typer.Exit(1)
    if output and len(inputs) > 1:
        typer.echo("✗ -o/--output can only be used with a single input")
        raise typer.Exit(1)

    # Diarization prerequisite checks
    if diarize:
        if not HAS_PYANNOTE:
            typer.echo("✗ Speaker diarization requires pyannote-audio.")
            typer.echo("  Install: pip install pyannote-audio")
            raise typer.Exit(1)
        if not HAS_FASTER_WHISPER:
            typer.echo("✗ Speaker diarization requires faster-whisper.")
            typer.echo("  Install: pip install faster-whisper")
            raise typer.Exit(1)
        if not get_hf_token():
            typer.echo("✗ Speaker diarization requires a HuggingFace token.")
            typer.echo("  1. Create at https://huggingface.co/settings/tokens")
            typer.echo("  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
            typer.echo("  3. Set HF_TOKEN env var or run: huggingface-cli login")
            raise typer.Exit(1)

    cache = CacheManager()
    engine = TranscriptionEngine(eff_model)

    urls = [i for i in inputs if not is_local_file(i)]
    files = [i for i in inputs if is_local_file(i)]

    success_count = 0
    fail_count = 0

    # Download all URLs concurrently (up to 3), then transcribe sequentially
    downloaded: dict[str, str] = {}  # url -> audio_path
    if urls and not any(is_local_file(u) for u in urls):
        # Pre-download in parallel for batch URL runs
        if len(urls) > 1:
            def _download(url):
                vid_id = get_video_id(url)
                info = get_video_info(url, cookies=str(cookies) if cookies else None, quiet=eff_quiet)
                title = info.get('title', vid_id)
                out_b = _output_base(title, None, output_dir, timestamp, cfg)
                audio_path = f"{out_b}.audio.mp3"
                try:
                    return url, download_audio(url, audio_path, cookies=str(cookies) if cookies else None, quiet=True)
                except Exception:
                    return url, None

            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_download, u): u for u in urls}
                for f in as_completed(futures):
                    url, audio = f.result()
                    if audio:
                        downloaded[url] = audio

    for inp in inputs:
        try:
            if is_local_file(inp):
                ok = _process_local(
                    inp,
                    output=output,
                    output_dir=output_dir,
                    model=eff_model,
                    language=eff_language,
                    fmt=eff_format,
                    clipboard=eff_clipboard,
                    timestamp=timestamp,
                    quiet=eff_quiet,
                    diarize=diarize,
                    num_speakers=num_speakers,
                    engine=engine,
                    config=cfg,
                )
            else:
                ok = _process_url(
                    inp,
                    output=output,
                    output_dir=output_dir,
                    model=eff_model,
                    language=eff_language,
                    fmt=eff_format,
                    clipboard=eff_clipboard,
                    keep_audio=keep_audio,
                    timestamp=timestamp,
                    quiet=eff_quiet,
                    cookies=cookies,
                    no_cache=no_cache,
                    force_whisper=force_whisper,
                    diarize=diarize,
                    num_speakers=num_speakers,
                    engine=engine,
                    cache=cache,
                    config=cfg,
                )
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except KeyboardInterrupt:
            typer.echo("\n\n⚠️  Interrupted by user")
            raise typer.Exit(1)
        except Exception as e:
            if not eff_quiet:
                typer.echo(f"✗ Unexpected error: {e}")
            fail_count += 1

    if len(inputs) > 1 and not eff_quiet:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Summary: {success_count} succeeded, {fail_count} failed")
        typer.echo(f"{'='*60}")

    if fail_count:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Cache subcommands
# ---------------------------------------------------------------------------

@cache_app.command("clear")
def cache_clear() -> None:
    """Delete all cached transcripts."""
    cache = CacheManager()
    count = cache.clear()
    typer.echo(f"✓ Cleared {count} cached transcript(s)")


@cache_app.command("stats")
def cache_stats() -> None:
    """Show cache statistics."""
    cache = CacheManager()
    s = cache.stats()
    typer.echo(f"Entries : {s['count']}")
    typer.echo(f"Size    : {s['size_mb']} MB")
    typer.echo(f"Oldest  : {s['oldest'] or 'n/a'}")
    typer.echo(f"Newest  : {s['newest'] or 'n/a'}")


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------

@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    cfg = load_config()
    path = get_config_path()
    typer.echo(f"Config file: {path}")
    typer.echo("")
    typer.echo(f"model       = {cfg.model}")
    typer.echo(f"format      = {cfg.format}")
    typer.echo(f"language    = {cfg.language or '(auto)'}")
    typer.echo(f"output_dir  = {cfg.output_dir or '(cwd)'}")
    typer.echo(f"clipboard   = {cfg.clipboard}")
    typer.echo(f"quiet       = {cfg.quiet}")
    typer.echo(f"keep_audio  = {cfg.keep_audio}")
    typer.echo(f"cache.ttl_days          = {cfg.cache.ttl_days}")
    typer.echo(f"diarization.hf_token    = {'(set)' if cfg.diarization.hf_token else '(not set)'}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help=f"Config key. Valid: {', '.join(SETTABLE_KEYS)}"),
    value: str = typer.Argument(..., help="Value to set."),
) -> None:
    """Set a persistent configuration value."""
    try:
        set_config_value(key, value)
        typer.echo(f"✓ Set {key} = {value}")
    except ValueError as e:
        typer.echo(f"✗ {e}")
        raise typer.Exit(1)
