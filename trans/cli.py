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


def _segments_to_clipboard_text(segments: list[dict]) -> str:
    """Plain-text rendering of segments that mirrors formatter.write_output's txt
    branch: `[Speaker N]` headers when segments carry speaker labels, otherwise
    a flat newline-join. Keeps cache-hit and cache-miss clipboards identical
    regardless of --format."""
    if not segments:
        return ""
    from .utils import format_speaker_label

    has_speakers = any("speaker" in s for s in segments)
    if not has_speakers:
        return "\n".join(s["text"] for s in segments)
    lines: list[str] = []
    current_speaker: str | None = None
    for seg in segments:
        speaker = format_speaker_label(seg.get("speaker", "UNKNOWN"))
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"[{speaker}]")
            current_speaker = speaker
        lines.append(seg["text"])
    return "\n".join(lines)


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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def _resolve_bool(cli_val: bool | None, cfg_val: bool) -> bool:
    """Return cli_val when the user passed the flag, else cfg_val.

    A bool CLI flag declared as ``Option(False, ...)`` can never be ``None``,
    so it always wins over the config value. Every config-aware bool flag must
    therefore default to ``None`` and route through this helper.
    """
    return cli_val if cli_val is not None else cfg_val


# ---------------------------------------------------------------------------
# URL processing
# ---------------------------------------------------------------------------


def _discard_prefetched(prefetched: dict[str, str] | None, url: str, keep_audio: bool) -> None:
    """Remove a prefetched audio file when a short-circuit path skips transcription."""
    if not prefetched or keep_audio:
        return
    path = prefetched.get(url)
    if path and Path(path).exists():
        try:
            os.remove(path)
        except OSError:
            pass


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
    translate: bool,
    engine: TranscriptionEngine,
    cache: CacheManager,
    config: Config,
    prefetched: dict[str, str] | None = None,
) -> bool:
    video_id = get_video_id(url)
    cookies_str = str(cookies) if cookies else None

    # Cache lookup — schema v2 returns canonical segments; render at hit time.
    if not no_cache:
        hit = cache.get(video_id, config.cache.ttl_days)
        if hit:
            if not quiet:
                typer.echo(f"\n💾 Using cached transcript for: {hit.title}")
            out_base = _output_base(hit.title, output, output_dir, timestamp, config)
            # any() instead of segments[0]: don't assume the first segment is
            # representative — a future segment source could leave gaps.
            diarized = any("speaker" in s for s in hit.segments)
            created = write_output(hit.segments, out_base, fmt, info=hit.info, diarized=diarized)
            if not quiet:
                for p in created:
                    if p.exists():
                        typer.echo(f"  → {p} ({p.stat().st_size} bytes)")
            if clipboard:
                _copy_to_clipboard(_segments_to_clipboard_text(hit.segments), quiet)
            _discard_prefetched(prefetched, url, keep_audio)
            return True

    # Fetch metadata
    info = get_video_info(url, cookies=cookies_str, quiet=quiet)
    video_title = info.get("title", "video")
    duration = info.get("duration", 0)
    out_base = _output_base(video_title, output, output_dir, timestamp, config)

    if not quiet:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"📹 {video_title}")
        if duration:
            mins, secs = divmod(duration, 60)
            typer.echo(f"⏱️  Duration: {int(mins)}:{int(secs):02d}")
        typer.echo(f"{'=' * 60}\n")

    # Try native captions first.
    # Note: native-captions hits are no longer cached. The cache holds canonical
    # Whisper segments; native captions only produce a rendered .txt (or .vtt/.srt
    # via yt-dlp). Caching them would require VTT->segments parsing, which is
    # tracked as a follow-up task (task-cache-native-captions). For now,
    # native-captions runs hit the network each time (~2s warm).
    if not force_whisper and extract_native_captions(url, out_base, fmt, quiet):
        if not quiet:
            typer.echo(f"\n✓ Transcription complete (native captions)")
            _print_output_files(out_base, fmt, ["txt", "vtt"])
        if clipboard:
            txt_path = Path(f"{out_base}.txt")
            if txt_path.exists():
                _copy_to_clipboard(txt_path.read_text(encoding="utf-8"), quiet)
        _discard_prefetched(prefetched, url, keep_audio)
        return True

    # Download + Whisper -- reuse a prefetched audio path when present
    prefetched_path = prefetched.get(url) if prefetched else None
    if (
        prefetched_path
        and Path(prefetched_path).exists()
        and Path(prefetched_path).stat().st_size > 0
    ):
        audio_file = prefetched_path
        prefetch_reused = True
    else:
        # Drop a bogus prefetch (missing or zero-byte) so it doesn't leak when
        # we fall through to a fresh download at a different path.
        _discard_prefetched(prefetched, url, keep_audio)
        audio_file = f"{out_base}.audio.mp3"
        prefetch_reused = False
    try:
        if prefetch_reused:
            if not quiet:
                typer.echo("→ Using pre-downloaded audio")
            final_audio = audio_file
        else:
            if not quiet:
                typer.echo("→ Downloading audio...")
            final_audio = download_audio(url, audio_file, cookies=cookies_str, quiet=quiet)

        segments, info_dict = engine.transcribe(
            final_audio, language=language or None, quiet=quiet, translate=translate
        )

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
            # Cache write failures must never kill a successful transcription —
            # the user already has their output files on disk.
            try:
                cache.put(
                    video_id,
                    url,
                    video_title,
                    segments,
                    info=info_dict,
                    model=model,
                )
                if not quiet:
                    typer.echo("💾 Cached for future use")
            except Exception as e:
                if not quiet:
                    typer.echo(f"⚠️  Cache write failed: {e} (transcription succeeded)")

        if clipboard:
            _copy_to_clipboard(_segments_to_clipboard_text(segments), quiet)

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
    translate: bool,
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
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"📁 {fp.name}")
        if duration:
            mins, secs = divmod(duration, 60)
            hours = int(mins) // 60
            mins = int(mins) % 60
            if hours > 0:
                typer.echo(f"⏱️  Duration: {hours}:{mins:02d}:{int(secs):02d}")
            else:
                typer.echo(f"⏱️  Duration: {int(mins)}:{int(secs):02d}")
        typer.echo(f"{'=' * 60}\n")

    audio_file = str(fp)
    temp_audio = None

    if not is_audio_file(str(fp)):
        temp_audio = f"{out_base}.temp_audio.mp3"
        if not extract_audio_from_video(str(fp), temp_audio, quiet):
            return False
        audio_file = temp_audio

    try:
        segments, info_dict = engine.transcribe(
            audio_file, language=language or None, quiet=quiet, translate=translate
        )

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
                _copy_to_clipboard(txt_path.read_text(encoding="utf-8"), quiet)

        return True

    except Exception as e:
        if not quiet:
            typer.echo(f"✗ Error during transcription: {e}")
        return False
    finally:
        if temp_audio and Path(temp_audio).exists():
            os.remove(temp_audio)


def _print_output_files(out_base: str, fmt: str, extras: list[str]) -> None:
    formats = [fmt] if fmt != "all" else extras
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
    output: str = typer.Option(
        None, "-o", "--output", help="Output base path (no extension). Single input only."
    ),
    output_dir: Path = typer.Option(None, "--output-dir", help="Directory for output files."),
    model: str = typer.Option(
        None, "-m", "--model", help=f"Whisper model: {', '.join(WHISPER_MODELS)}"
    ),
    language: str = typer.Option(
        None, "-l", "--language", help="Language code (e.g. en, es). Auto-detect if unset."
    ),
    format: str = typer.Option(
        None, "-f", "--format", help=f"Output format: {', '.join(OUTPUT_FORMATS)}"
    ),
    clipboard: bool = typer.Option(None, "-c", "--clipboard", help="Copy transcript to clipboard."),
    keep_audio: bool = typer.Option(None, "-k", "--keep-audio", help="Keep downloaded audio file."),
    timestamp: bool = typer.Option(
        False, "-t", "--timestamp", help="Add timestamp to output filename."
    ),
    quiet: bool = typer.Option(None, "-q", "--quiet", help="Minimal output (errors only)."),
    cookies: Path = typer.Option(
        None, "--cookies", help="Path to cookies.txt for authenticated downloads."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Skip cache lookup and force fresh transcription."
    ),
    force_whisper: bool = typer.Option(
        False, "--force-whisper", help="Skip native captions, always use Whisper."
    ),
    diarize: bool = typer.Option(
        False, "-d", "--diarize", help="Enable speaker diarization (requires pyannote-audio)."
    ),
    num_speakers: int = typer.Option(
        None, "--num-speakers", help="Number of speakers (helps diarization accuracy)."
    ),
    translate: bool = typer.Option(
        False, "--translate", help="Translate non-English audio to English."
    ),
) -> None:
    """Transcribe video/audio URLs or local files to text."""

    cfg = load_config()

    # Resolve options: CLI > config > hardcoded default
    eff_model = _resolve(model, cfg.model, "base")
    eff_format = _resolve(format, cfg.format, "txt")
    eff_language = _resolve(language, cfg.language, None)
    eff_clipboard = _resolve_bool(clipboard, cfg.clipboard)
    eff_quiet = _resolve_bool(quiet, cfg.quiet)
    eff_keep_audio = _resolve_bool(keep_audio, cfg.keep_audio)

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
            typer.echo(
                "  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
            typer.echo("  3. Set HF_TOKEN env var or run: huggingface-cli login")
            raise typer.Exit(1)

    cache = CacheManager()
    engine = TranscriptionEngine(eff_model)

    urls = [i for i in inputs if not is_local_file(i)]
    files = [i for i in inputs if is_local_file(i)]

    success_count = 0
    fail_count = 0

    # Pre-download URLs concurrently (up to 3) for batch runs so the
    # sequential transcription loop can reuse the audio files. URLs whose
    # transcripts are already cached are skipped so we don't waste bytes.
    downloaded: dict[str, str] = {}
    if len(urls) > 1:
        cookies_str = str(cookies) if cookies else None
        ttl = cfg.cache.ttl_days

        def _prefetch(url: str) -> tuple[str, str | None]:
            # Schema v2: cache lookup no longer keys on format. A hit on any
            # video_id means we have its segments and can render any format
            # at hit time — so prefetch can skip download regardless of -f.
            if not no_cache and cache.get(get_video_id(url), ttl):
                return url, None
            # get_video_info / download_audio raise SystemExit on hard yt-dlp
            # errors (TikTok IP block, dead URL). Catch it here so one bad URL
            # never kills the orchestrator before any URL gets transcribed.
            try:
                info = get_video_info(url, cookies=cookies_str, quiet=True)
            except (Exception, SystemExit):
                return url, None
            title = info.get("title", get_video_id(url))
            out_b = _output_base(title, None, output_dir, timestamp, cfg)
            audio_path = f"{out_b}.audio.mp3"
            try:
                return url, download_audio(url, audio_path, cookies=cookies_str, quiet=True)
            except (Exception, SystemExit):
                return url, None

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_prefetch, u): u for u in urls}
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
                    translate=translate,
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
                    keep_audio=eff_keep_audio,
                    timestamp=timestamp,
                    quiet=eff_quiet,
                    cookies=cookies,
                    no_cache=no_cache,
                    force_whisper=force_whisper,
                    diarize=diarize,
                    num_speakers=num_speakers,
                    translate=translate,
                    engine=engine,
                    cache=cache,
                    config=cfg,
                    prefetched=downloaded,
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
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Summary: {success_count} succeeded, {fail_count} failed")
        typer.echo(f"{'=' * 60}")

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
