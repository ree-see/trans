#!/usr/bin/env python3
"""Unit tests for trans package."""

import pytest
import sys
import tempfile
import os
from pathlib import Path

# Import functions from trans.utils
from trans.utils import (
    get_video_id,
    sanitize_filename,
    is_tiktok_url,
    is_twitch_url,
    is_local_file,
    is_audio_file,
    format_timestamp_srt,
    format_timestamp_vtt,
    format_speaker_label,
    assign_speakers_to_segments,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MEDIA_EXTENSIONS,
)
from trans.downloader import _base_opts
from yt_dlp.networking.impersonate import ImpersonateTarget


class TestGetVideoId:
    """Tests for get_video_id() URL parsing."""

    def test_youtube_standard_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert get_video_id(url) == "yt_dQw4w9WgXcQ"

    def test_youtube_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert get_video_id(url) == "yt_dQw4w9WgXcQ"

    def test_youtube_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42"
        assert get_video_id(url) == "yt_dQw4w9WgXcQ"

    def test_youtube_mobile_url(self):
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert get_video_id(url) == "yt_dQw4w9WgXcQ"

    def test_tiktok_video_url(self):
        url = "https://www.tiktok.com/@user/video/1234567890123456789"
        assert get_video_id(url) == "tt_1234567890123456789"

    def test_twitch_vod_url(self):
        url = "https://www.twitch.tv/videos/1234567890"
        assert get_video_id(url) == "tw_1234567890"

    def test_twitch_clip_url(self):
        url = "https://www.twitch.tv/channel/clip/AmazingClipName123"
        assert get_video_id(url) == "twclip_AmazingClipName123"

    def test_twitch_clips_domain(self):
        url = "https://clips.twitch.tv/AmazingClipName123"
        assert get_video_id(url) == "twclip_AmazingClipName123"

    def test_unknown_url_uses_hash(self):
        url = "https://example.com/some/video/path"
        video_id = get_video_id(url)
        assert video_id.startswith("hash_")
        assert len(video_id) == 17  # "hash_" + 12 char hash

    def test_same_url_same_hash(self):
        url = "https://example.com/video123"
        assert get_video_id(url) == get_video_id(url)


class TestSanitizeFilename:
    """Tests for sanitize_filename()."""

    def test_basic_title(self):
        assert sanitize_filename("Hello World") == "Hello_World"

    def test_special_characters(self):
        assert sanitize_filename("Video: The Best! @2024") == "Video_The_Best_2024"

    def test_multiple_spaces(self):
        assert sanitize_filename("Hello   World") == "Hello_World"

    def test_multiple_underscores(self):
        assert sanitize_filename("Hello___World") == "Hello_World"

    def test_max_length(self):
        long_title = "A" * 100
        result = sanitize_filename(long_title, max_length=50)
        assert len(result) == 50

    def test_custom_max_length(self):
        long_title = "Hello World This Is A Long Title"
        result = sanitize_filename(long_title, max_length=10)
        assert len(result) == 10

    def test_strips_leading_trailing_underscores(self):
        assert sanitize_filename("  Hello World  ") == "Hello_World"
        assert sanitize_filename("_Hello_") == "Hello"

    def test_unicode_preserved(self):
        # Alphanumeric includes unicode letters
        result = sanitize_filename("Café Video")
        assert "Caf" in result


class TestUrlDetection:
    """Tests for URL detection functions."""

    def test_is_tiktok_url_standard(self):
        assert is_tiktok_url("https://www.tiktok.com/@user/video/123")
        assert is_tiktok_url("https://tiktok.com/video")

    def test_is_tiktok_url_vm_shortlink(self):
        assert is_tiktok_url("https://vm.tiktok.com/abc123")

    def test_is_tiktok_url_false(self):
        assert not is_tiktok_url("https://youtube.com/watch?v=123")
        assert not is_tiktok_url("https://example.com")

    def test_is_twitch_url_standard(self):
        assert is_twitch_url("https://www.twitch.tv/videos/123")
        assert is_twitch_url("https://twitch.tv/channel")

    def test_is_twitch_url_false(self):
        assert not is_twitch_url("https://youtube.com/watch?v=123")
        assert not is_twitch_url("https://tiktok.com/video")


class TestLocalFileDetection:
    """Tests for is_local_file() and is_audio_file()."""

    def test_is_local_file_audio(self):
        assert is_local_file("podcast.mp3")
        assert is_local_file("/path/to/audio.wav")
        assert is_local_file("./music.flac")

    def test_is_local_file_video(self):
        assert is_local_file("video.mp4")
        assert is_local_file("/home/user/movie.mkv")
        assert is_local_file("lecture.webm")

    def test_is_local_file_url_false(self):
        assert not is_local_file("https://youtube.com/watch?v=123")
        assert not is_local_file("http://example.com/video.mp4")

    def test_is_local_file_known_domains_false(self):
        assert not is_local_file("youtube.com/video")
        assert not is_local_file("tiktok.com/@user/video/123")

    def test_is_local_file_unknown_extension_false(self):
        assert not is_local_file("document.pdf")
        assert not is_local_file("image.png")

    def test_is_audio_file_true(self):
        assert is_audio_file("song.mp3")
        assert is_audio_file("recording.wav")
        assert is_audio_file("podcast.m4a")
        assert is_audio_file("music.flac")
        assert is_audio_file("audio.ogg")
        assert is_audio_file("voice.opus")

    def test_is_audio_file_video_false(self):
        assert not is_audio_file("video.mp4")
        assert not is_audio_file("movie.mkv")
        assert not is_audio_file("clip.webm")


class TestTimestampFormatting:
    """Tests for timestamp formatting functions."""

    def test_format_timestamp_srt_zero(self):
        assert format_timestamp_srt(0) == "00:00:00,000"

    def test_format_timestamp_srt_seconds(self):
        assert format_timestamp_srt(5) == "00:00:05,000"

    def test_format_timestamp_srt_minutes(self):
        assert format_timestamp_srt(125) == "00:02:05,000"

    def test_format_timestamp_srt_hours(self):
        assert format_timestamp_srt(3661) == "01:01:01,000"

    def test_format_timestamp_srt_milliseconds(self):
        assert format_timestamp_srt(1.5) == "00:00:01,500"
        assert format_timestamp_srt(1.234) == "00:00:01,234"

    def test_format_timestamp_vtt_zero(self):
        assert format_timestamp_vtt(0) == "00:00:00.000"

    def test_format_timestamp_vtt_seconds(self):
        assert format_timestamp_vtt(5) == "00:00:05.000"

    def test_format_timestamp_vtt_minutes(self):
        assert format_timestamp_vtt(125) == "00:02:05.000"

    def test_format_timestamp_vtt_hours(self):
        assert format_timestamp_vtt(3661) == "01:01:01.000"

    def test_format_timestamp_vtt_milliseconds(self):
        assert format_timestamp_vtt(1.5) == "00:00:01.500"
        assert format_timestamp_vtt(1.234) == "00:00:01.234"

    def test_srt_vs_vtt_delimiter(self):
        # SRT uses comma, VTT uses period
        srt = format_timestamp_srt(1.5)
        vtt = format_timestamp_vtt(1.5)
        assert "," in srt and "." not in srt.split(",")[1]
        assert "." in vtt


class TestSpeakerLabels:
    """Tests for speaker label formatting."""

    def test_format_speaker_label_standard(self):
        assert format_speaker_label("SPEAKER_00") == "Speaker 1"
        assert format_speaker_label("SPEAKER_01") == "Speaker 2"
        assert format_speaker_label("SPEAKER_09") == "Speaker 10"

    def test_format_speaker_label_unknown(self):
        assert format_speaker_label("UNKNOWN") == "Unknown"

    def test_format_speaker_label_passthrough(self):
        # Non-standard format passes through
        assert format_speaker_label("CustomSpeaker") == "CustomSpeaker"


class TestSpeakerAssignment:
    """Tests for assign_speakers_to_segments()."""

    def test_simple_assignment(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 5.0, "end": 10.0, "text": "World"},
        ]
        diarization = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]
        result = assign_speakers_to_segments(transcript, diarization)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_overlapping_speakers_majority_wins(self):
        # Transcript segment overlaps with two speakers, longer overlap wins
        transcript = [
            {"start": 0.0, "end": 10.0, "text": "Long segment"},
        ]
        diarization = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},  # 3 seconds
            {"start": 3.0, "end": 10.0, "speaker": "SPEAKER_01"},  # 7 seconds
        ]
        result = assign_speakers_to_segments(transcript, diarization)
        assert result[0]["speaker"] == "SPEAKER_01"  # Longer overlap

    def test_no_overlap_assigns_unknown(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Gap segment"},
        ]
        diarization = [
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00"},
        ]
        result = assign_speakers_to_segments(transcript, diarization)
        assert result[0]["speaker"] == "UNKNOWN"

    def test_empty_diarization(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "No speakers"},
        ]
        result = assign_speakers_to_segments(transcript, [])
        assert result[0]["speaker"] == "UNKNOWN"


class TestExtensionSets:
    """Tests for extension constants."""

    def test_audio_extensions_complete(self):
        expected = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
        assert AUDIO_EXTENSIONS == expected

    def test_video_extensions_complete(self):
        expected = {
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".webm",
            ".flv",
            ".wmv",
            ".m4v",
            ".mpeg",
            ".mpg",
        }
        assert VIDEO_EXTENSIONS == expected

    def test_media_extensions_is_union(self):
        assert MEDIA_EXTENSIONS == AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

    def test_no_overlap_audio_video(self):
        # Audio and video extensions should be disjoint
        assert AUDIO_EXTENSIONS.isdisjoint(VIDEO_EXTENSIONS)


class TestDownloaderImpersonate:
    """Tests for _base_opts impersonation handling.

    Regression guard for the yt-dlp >= 2024.07.01 API change: the
    ``impersonate`` option must be an ``ImpersonateTarget`` instance,
    not a raw string. A string here crashes with an empty-message
    AssertionError inside yt-dlp's is_supported_target().
    """

    def test_tiktok_url_gets_impersonate_target(self, monkeypatch):
        # Pin backend-present branch independent of install matrix
        # (avoids test breaking under `pip install -e ".[dev]"` without [tiktok]).
        monkeypatch.setitem(sys.modules, "curl_cffi", object())
        opts = _base_opts(
            "https://www.tiktok.com/@user/video/1234567890123456789",
            cookies=None,
            quiet=True,
        )
        assert "impersonate" in opts
        assert isinstance(opts["impersonate"], ImpersonateTarget)
        assert str(opts["impersonate"]) == "chrome-131"

    def test_non_tiktok_url_has_no_impersonate(self):
        opts = _base_opts(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            cookies=None,
            quiet=True,
        )
        assert "impersonate" not in opts

    def test_tiktok_without_backend_skips_impersonate(self, monkeypatch):
        # When curl_cffi is missing, _base_opts must omit the 'impersonate'
        # key so yt-dlp does not raise "Impersonate target ... is not available"
        # at download time.
        monkeypatch.setitem(sys.modules, "curl_cffi", None)
        opts = _base_opts(
            "https://www.tiktok.com/@user/video/1234567890123456789",
            cookies=None,
            quiet=True,
        )
        assert "impersonate" not in opts

    def test_tiktok_without_backend_prints_hint(self, monkeypatch, capsys):
        import trans.downloader

        # Reset the once-per-process flag so the hint actually prints here.
        monkeypatch.setattr(trans.downloader, "_BACKEND_HINT_SHOWN", False)
        monkeypatch.setitem(sys.modules, "curl_cffi", None)
        _base_opts(
            "https://www.tiktok.com/@user/video/1234567890123456789",
            cookies=None,
            quiet=False,
        )
        captured = capsys.readouterr()
        assert "boswell[tiktok]" in captured.out
        assert "backend not installed" in captured.out


class TestBatchPrefetch:
    """Tests that `_process_url` honors a `prefetched` dict of pre-downloaded audio paths.

    Stubs target `trans.cli.*` rather than `trans.downloader.*` because
    `trans/cli.py` does `from .downloader import download_audio, ...`,
    which binds the names into `trans.cli`'s namespace. Patching the
    `trans.downloader` attribute has no effect on `_process_url`'s lookups.
    """

    @staticmethod
    def _build_fixtures(tmp_path, monkeypatch):
        from trans import cli as cli_mod
        from trans.cache import CacheManager
        from trans.config import Config

        calls = {"download": 0}

        def fake_download(url, output_path, cookies=None, quiet=False, **_):
            calls["download"] += 1
            final = output_path if str(output_path).endswith(".mp3") else f"{output_path}.mp3"
            Path(final).write_bytes(b"\x00" * 1024)
            return final

        def fake_get_video_info(url, cookies=None, quiet=False):
            return {"title": "fake_video", "duration": 1}

        monkeypatch.setattr(cli_mod, "download_audio", fake_download)
        monkeypatch.setattr(cli_mod, "get_video_info", fake_get_video_info)
        monkeypatch.setattr(cli_mod, "extract_native_captions", lambda *a, **kw: False)

        class FakeEngine:
            def transcribe(self, audio, *, language=None, quiet=False, translate=False):
                return [{"start": 0.0, "end": 1.0, "text": "hi"}], {"language": "en"}

        cache = CacheManager(db_path=tmp_path / "test_cache.db")
        cfg = Config()

        common_kwargs: dict = dict(
            output=None,
            output_dir=tmp_path,
            model="base",
            language=None,
            fmt="txt",
            clipboard=False,
            keep_audio=False,
            timestamp=False,
            quiet=True,
            cookies=None,
            no_cache=False,
            force_whisper=False,
            diarize=False,
            num_speakers=None,
            translate=False,
            engine=FakeEngine(),
            cache=cache,
            config=cfg,
        )
        return calls, common_kwargs

    def test_prefetched_audio_is_reused(self, tmp_path, monkeypatch):
        """When prefetched[url] points at an existing non-empty file, skip download_audio."""
        from trans.cli import _process_url

        calls, kwargs = self._build_fixtures(tmp_path, monkeypatch)
        prefetched_audio = tmp_path / "prefetched.audio.mp3"
        prefetched_audio.write_bytes(b"\x00" * 2048)

        url = "https://youtube.com/watch?v=fakefakefake"
        ok = _process_url(url, prefetched={url: str(prefetched_audio)}, **kwargs)

        assert ok is True
        assert calls["download"] == 0, "download_audio must not be called when prefetch is usable"

    def test_missing_prefetch_falls_back_to_download(self, tmp_path, monkeypatch):
        """A missing prefetched path falls through to download_audio."""
        from trans.cli import _process_url

        calls, kwargs = self._build_fixtures(tmp_path, monkeypatch)
        url = "https://youtube.com/watch?v=fakefakefake"
        ok = _process_url(
            url,
            prefetched={url: str(tmp_path / "nonexistent.mp3")},
            **kwargs,
        )

        assert ok is True
        assert calls["download"] == 1

    def test_prefetch_skipped_when_none(self, tmp_path, monkeypatch):
        """`prefetched=None` preserves current behavior (always downloads on cache miss)."""
        from trans.cli import _process_url

        calls, kwargs = self._build_fixtures(tmp_path, monkeypatch)
        url = "https://youtube.com/watch?v=fakefakefake"
        ok = _process_url(url, prefetched=None, **kwargs)

        assert ok is True
        assert calls["download"] == 1


class TestConfigBoolRoundTrip:
    """Round-trip every bool config key: write via set_config_value, reload, assert.

    Catches the family of bugs where a SETTABLE_KEYS bool key drifts out of sync
    between the Config dataclass, the TOML writer, and the typed-value parser in
    set_config_value.
    """

    @pytest.mark.parametrize(
        "key,attr",
        [
            ("clipboard", "clipboard"),
            ("quiet", "quiet"),
            ("keep_audio", "keep_audio"),
        ],
    )
    def test_bool_key_round_trip(self, tmp_path, key, attr):
        from trans.config import load_config, set_config_value

        cfg_path = tmp_path / "config.toml"
        set_config_value(key, "true", path=cfg_path)
        assert getattr(load_config(path=cfg_path), attr) is True
        set_config_value(key, "false", path=cfg_path)
        assert getattr(load_config(path=cfg_path), attr) is False


class TestConfigKeepAudioResolution:
    """Regression guard for the silently-ignored `keep_audio` config key.

    Pre-fix: `keep_audio: bool = typer.Option(False, ...)` short-circuits the
    `_resolve_bool(cli, cfg)` pattern because the CLI default is never `None`.
    Setting `cfg.keep_audio = true` had no effect — the bug this fix removes.
    """

    @staticmethod
    def _setup_stubs(tmp_path, monkeypatch, *, cfg_keep_audio: bool):
        from typer.testing import CliRunner

        from trans import cli as cli_mod
        from trans.cache import CacheManager
        from trans.config import Config

        monkeypatch.setattr(cli_mod, "load_config", lambda: Config(keep_audio=cfg_keep_audio))

        def fake_download(url, output_path, cookies=None, quiet=False, **_):
            final = output_path if str(output_path).endswith(".mp3") else f"{output_path}.mp3"
            Path(final).write_bytes(b"\x00" * 1024)
            return final

        monkeypatch.setattr(cli_mod, "download_audio", fake_download)
        monkeypatch.setattr(
            cli_mod,
            "get_video_info",
            lambda url, cookies=None, quiet=False: {"title": "stub", "duration": 1},
        )
        monkeypatch.setattr(cli_mod, "extract_native_captions", lambda *a, **kw: False)

        class StubEngine:
            def transcribe(self, audio, *, language=None, quiet=False, translate=False):
                return (
                    [{"start": 0.0, "end": 1.0, "text": "x"}],
                    {"language": "en"},
                )

        monkeypatch.setattr(cli_mod, "TranscriptionEngine", lambda model: StubEngine())
        monkeypatch.setattr(
            cli_mod, "CacheManager", lambda: CacheManager(db_path=tmp_path / "c.db")
        )
        return CliRunner(), cli_mod

    def test_config_keep_audio_true_retains_audio_without_flag(self, tmp_path, monkeypatch):
        """cfg.keep_audio=True + no -k flag must retain the audio file."""
        runner, cli_mod = self._setup_stubs(tmp_path, monkeypatch, cfg_keep_audio=True)
        result = runner.invoke(
            cli_mod.app,
            [
                "transcribe",
                "--output-dir",
                str(tmp_path),
                "--quiet",
                "https://www.youtube.com/watch?v=fakefakefake",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "stub.audio.mp3").exists(), (
            "config keep_audio=true was silently ignored — audio was deleted"
        )

    def test_config_keep_audio_false_default_removes_audio(self, tmp_path, monkeypatch):
        """Positive control: cfg.keep_audio=False + no -k → audio removed.

        Proves the stub fixture writes/deletes a real path at the asserted
        location, so the True-case test can't pass for the wrong reason.
        """
        runner, cli_mod = self._setup_stubs(tmp_path, monkeypatch, cfg_keep_audio=False)
        result = runner.invoke(
            cli_mod.app,
            [
                "transcribe",
                "--output-dir",
                str(tmp_path),
                "--quiet",
                "https://www.youtube.com/watch?v=fakefakefake",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not (tmp_path / "stub.audio.mp3").exists(), (
            "fixture sanity: audio must be removed when keep_audio is False"
        )

    def test_cli_flag_overrides_config_false(self, tmp_path, monkeypatch):
        """CLI `-k` must beat `cfg.keep_audio=False` — the other half of the precedence rule."""
        runner, cli_mod = self._setup_stubs(tmp_path, monkeypatch, cfg_keep_audio=False)
        result = runner.invoke(
            cli_mod.app,
            [
                "transcribe",
                "--output-dir",
                str(tmp_path),
                "-k",
                "--quiet",
                "https://www.youtube.com/watch?v=fakefakefake",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "stub.audio.mp3").exists(), (
            "explicit -k must override cfg.keep_audio=False"
        )


class TestPackageImports:
    """Regression guards on the public package surface."""

    def test_package_imports_cleanly(self):
        """Every public submodule must remain importable in any env with the declared required deps."""
        import importlib

        import trans  # noqa: F401

        for mod in (
            "trans.cli",
            "trans.config",
            "trans.cache",
            "trans.downloader",
            "trans.transcriber",
            "trans.utils",
            "trans.diarizer",
            "trans.formatter",
        ):
            importlib.import_module(mod)


class TestCacheRoundTrip:
    """Cache schema v2: store segments + info; render at hit time.

    Validates the contract introduced by task-cache-format-contract — the cache
    holds the canonical segment list for a video, not a rendered text blob.
    """

    @staticmethod
    def _sample_segments():
        return [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "Second segment"},
        ]

    @staticmethod
    def _sample_info():
        return {"language": "en", "language_probability": 0.99, "duration": 5.0}

    def test_put_get_round_trip(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        segs = self._sample_segments()
        info = self._sample_info()
        cache.put("yt_abc", "https://youtu.be/abc", "My Video", segs, info=info)
        hit = cache.get("yt_abc", ttl_days=30)
        assert hit is not None
        assert hit.segments == segs
        assert hit.info == info
        assert hit.title == "My Video"
        assert hit.source == "whisper"

    def test_cache_hit_renders_all_formats(self, tmp_path):
        from trans.cache import CacheManager
        from trans.formatter import write_output

        cache = CacheManager(db_path=tmp_path / "c.db")
        segs = self._sample_segments()
        cache.put("yt_abc", "url", "Title", segs, info=self._sample_info())
        hit = cache.get("yt_abc", ttl_days=30)
        base = str(tmp_path / "out")
        write_output(hit.segments, base, "all", info=hit.info)
        for ext in ("txt", "srt", "vtt", "json"):
            p = Path(f"{base}.{ext}")
            assert p.exists(), f"{ext} not written"
            assert p.stat().st_size > 0

    def test_cache_hit_renders_requested_format_only(self, tmp_path):
        from trans.cache import CacheManager
        from trans.formatter import write_output

        cache = CacheManager(db_path=tmp_path / "c.db")
        segs = self._sample_segments()
        cache.put("yt_abc", "url", "Title", segs)
        hit = cache.get("yt_abc", ttl_days=30)
        base = str(tmp_path / "out")
        write_output(hit.segments, base, "srt")
        assert Path(f"{base}.srt").exists()
        for ext in ("txt", "vtt", "json"):
            assert not Path(f"{base}.{ext}").exists()

    def test_cache_get_miss_returns_none(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        assert cache.get("yt_nonexistent", ttl_days=30) is None

    def test_cache_get_expired_returns_none(self, tmp_path):
        import sqlite3

        from trans.cache import CacheManager

        db = tmp_path / "c.db"
        cache = CacheManager(db_path=db)
        cache.put("yt_old", "url", "Old", self._sample_segments())
        # Backdate to outside TTL window
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE transcripts SET created_at = datetime('now', '-60 days') "
            "WHERE video_id = 'yt_old'"
        )
        conn.commit()
        conn.close()
        assert cache.get("yt_old", ttl_days=30) is None

    def test_cache_get_diarized_segments_preserved(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        segs = [
            {"start": 0.0, "end": 2.0, "text": "Hi", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "text": "Hello", "speaker": "SPEAKER_01"},
        ]
        cache.put("yt_dia", "url", "Diarized", segs)
        hit = cache.get("yt_dia", ttl_days=30)
        assert hit.segments[0]["speaker"] == "SPEAKER_00"
        assert hit.segments[1]["speaker"] == "SPEAKER_01"

    def test_schema_mismatch_resets(self, tmp_path):
        import sqlite3

        from trans.cache import CacheManager

        db = tmp_path / "c.db"
        # Simulate old schema v1: transcript text column, user_version=1, one row
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE transcripts ("
            "video_id TEXT PRIMARY KEY, url TEXT, title TEXT, transcript TEXT, "
            "format TEXT, model TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO transcripts (video_id, url, title, transcript, format) "
            "VALUES ('yt_old', 'u', 't', 'text', 'txt')"
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        cache = CacheManager(db_path=db)
        # Touch a public method to trigger lazy schema-ensure
        assert cache.get("yt_old", ttl_days=30) is None
        # Verify new schema is in place
        conn = sqlite3.connect(db)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        cols = [r[1] for r in conn.execute("PRAGMA table_info(transcripts)")]
        conn.close()
        assert version == 2
        assert "segments_json" in cols
        assert "transcript" not in cols

    def test_schema_migration_from_unversioned_db(self, tmp_path):
        """A DB file with no transcripts table and user_version=0 migrates cleanly.

        Edge case: a previous trans crashed mid-init, or a user deleted the
        transcripts table manually. The migration must tolerate the absence of
        the table when computing the discarded-row count.
        """
        import sqlite3

        from trans.cache import CacheManager

        db = tmp_path / "c.db"
        # Create the DB file but no transcripts table; user_version stays at 0.
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE other (x INTEGER)")
        conn.commit()
        conn.close()

        cache = CacheManager(db_path=db)
        # First public call must not raise.
        assert cache.get("yt_anything", ttl_days=30) is None
        conn = sqlite3.connect(db)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        cols = [r[1] for r in conn.execute("PRAGMA table_info(transcripts)")]
        conn.close()
        assert version == 2
        assert "segments_json" in cols

    def test_put_with_none_info_round_trips(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        cache.put("yt_noinfo", "url", "Title", self._sample_segments(), info=None)
        hit = cache.get("yt_noinfo", ttl_days=30)
        assert hit is not None
        assert hit.info is None

    def test_put_numpy_floats_serialize(self, tmp_path):
        np = pytest.importorskip("numpy")
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        segs = [
            {"start": np.float64(0.0), "end": np.float64(2.5), "text": "Hi"},
        ]
        info = {
            "language": "en",
            "language_probability": np.float32(0.99),
            "duration": np.float64(5.0),
        }
        cache.put("yt_numpy", "url", "Title", segs, info=info)
        hit = cache.get("yt_numpy", ttl_days=30)
        assert isinstance(hit.segments[0]["start"], float)
        assert isinstance(hit.segments[0]["end"], float)
        assert isinstance(hit.info["language_probability"], float)
        assert isinstance(hit.info["duration"], float)

    def test_lazy_db_creation(self, tmp_path):
        from trans.cache import CacheManager

        db = tmp_path / "nope.db"
        CacheManager(db_path=db)
        assert not db.exists(), "constructing CacheManager must not touch the filesystem"

    def test_native_captions_no_longer_cached(self, tmp_path, monkeypatch):
        """Native-captions path must NOT write to the cache (post-contract behavior)."""
        from trans import cli as cli_mod
        from trans.cache import CacheManager
        from trans.config import Config

        # Stub the native-captions write so it returns success without network.
        def fake_native(url, out_base, fmt, quiet):
            Path(f"{out_base}.txt").write_text("captioned content", encoding="utf-8")
            return True

        monkeypatch.setattr(cli_mod, "extract_native_captions", fake_native)
        monkeypatch.setattr(
            cli_mod,
            "get_video_info",
            lambda url, cookies=None, quiet=False: {"title": "Native", "duration": 1},
        )

        class StubEngine:
            def transcribe(self, *a, **kw):
                raise AssertionError("native-captions path should short-circuit")

        cache = CacheManager(db_path=tmp_path / "c.db")
        cfg = Config()
        url = "https://www.youtube.com/watch?v=fakefakefake"
        ok = cli_mod._process_url(
            url,
            output=None,
            output_dir=tmp_path,
            model="base",
            language=None,
            fmt="txt",
            clipboard=False,
            keep_audio=False,
            timestamp=False,
            quiet=True,
            cookies=None,
            no_cache=False,
            force_whisper=False,
            diarize=False,
            num_speakers=None,
            translate=False,
            engine=StubEngine(),
            cache=cache,
            config=cfg,
        )
        assert ok is True
        # Critical assertion: native-captions success must not populate the cache
        from trans.utils import get_video_id

        assert cache.get(get_video_id(url), ttl_days=30) is None

    def test_clipboard_copies_segments_text_on_cache_hit(self, tmp_path, monkeypatch):
        """B-1 fix: -c on a cache hit copies plain-text join regardless of --format."""
        from trans import cli as cli_mod
        from trans.cache import CacheManager
        from trans.config import Config

        clipboard_calls: list[str] = []

        def fake_copy(text: str, quiet: bool) -> None:
            clipboard_calls.append(text)

        monkeypatch.setattr(cli_mod, "_copy_to_clipboard", fake_copy)

        cache = CacheManager(db_path=tmp_path / "c.db")
        segs = [
            {"start": 0.0, "end": 1.0, "text": "alpha"},
            {"start": 1.0, "end": 2.0, "text": "beta"},
        ]
        url = "https://www.youtube.com/watch?v=fakefakefake"
        from trans.utils import get_video_id

        cache.put(get_video_id(url), url, "Title", segs)

        class UnreachableEngine:
            def transcribe(self, *a, **kw):
                raise AssertionError("cache hit must short-circuit transcription")

        ok = cli_mod._process_url(
            url,
            output=None,
            output_dir=tmp_path,
            model="base",
            language=None,
            fmt="srt",  # NOT txt — but clipboard must still receive plain text
            clipboard=True,
            keep_audio=False,
            timestamp=False,
            quiet=True,
            cookies=None,
            no_cache=False,
            force_whisper=False,
            diarize=False,
            num_speakers=None,
            translate=False,
            engine=UnreachableEngine(),
            cache=cache,
            config=Config(),
        )
        assert ok is True
        assert clipboard_calls == ["alpha\nbeta"]

    def test_clipboard_diarized_keeps_speaker_headers(self, tmp_path):
        """Diarized segments on a cache-hit clipboard should mirror the txt format,
        i.e. include [Speaker N] headers rather than a flat join."""
        from trans.cli import _segments_to_clipboard_text

        segs = [
            {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.0, "text": "world", "speaker": "SPEAKER_01"},
        ]
        text = _segments_to_clipboard_text(segs)
        # Two speaker blocks separated by a blank line, same shape as formatter txt.
        assert text == "[Speaker 1]\nhi\nhello\n\n[Speaker 2]\nworld"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
