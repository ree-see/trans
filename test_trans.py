#!/usr/bin/env python3
"""Unit tests for trans package."""

import json as _json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


@pytest.fixture(autouse=True)
def _reset_module_toggles():
    """Reset per-session warning/verbose flags so test order is irrelevant.

    Three module-level flags accumulate state across a process:
    `trans.downloader._BACKEND_HINT_SHOWN`, `trans.cli._TIKTOK_HELP_SHOWN`,
    and `trans.cli._VERBOSE`. Without this fixture the second TikTok-batch
    test in a run would silently skip the help block, and a verbose test
    that fails to clean up would leak verbose mode into unrelated tests.
    """
    import trans.cli as _cli
    import trans.downloader as _dl

    _dl._BACKEND_HINT_SHOWN = False
    if hasattr(_cli, "_TIKTOK_HELP_SHOWN"):
        _cli._TIKTOK_HELP_SHOWN = False
    if hasattr(_cli, "_VERBOSE"):
        _cli._VERBOSE = False
    yield


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

    def test_cookies_path_sets_cookiefile(self, tmp_path):
        # _base_opts only stores the path string; it doesn't read the file.
        cookies = tmp_path / "cookies.txt"
        opts = _base_opts(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            cookies=str(cookies),
            quiet=True,
        )
        assert opts["cookiefile"] == str(cookies)

    def test_no_cookies_omits_cookiefile(self):
        opts = _base_opts(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            cookies=None,
            quiet=True,
        )
        assert "cookiefile" not in opts


class TestExtractNativeCaptions:
    """Cover the three output_format branches of extract_native_captions.

    The function mixes a yt-dlp download call with file-shuffling logic
    (write the .txt, rename the .vtt, or delete the source). Mock the
    YoutubeDL context manager so the test exercises just the file logic.

    Branches:
      - txt: writes .txt + removes .en.vtt source
      - vtt: renames .en.vtt -> .vtt, no .txt
      - all: writes .txt AND renames .en.vtt -> .vtt
    """

    _VTT_FIXTURE = (
        "WEBVTT\n"
        "Kind: captions\n"
        "Language: en\n"
        "\n"
        "NOTE This is a comment block\n"
        "\n"
        "1\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "Hello world\n"
        "\n"
        "2\n"
        "00:00:02.000 --> 00:00:04.000\n"
        "This is a test\n"
    )

    @staticmethod
    def _mock_ydl(output_path, fixture, sub_format="vtt"):
        """Build a MagicMock that satisfies yt_dlp.YoutubeDL(opts)'s context
        manager protocol and writes the fixture to ``output_path.en.{sub_format}``
        on .download(). The sub_format arg lets the srt branch test the real
        rename path (yt-dlp would write .en.srt when subtitlesformat=srt)."""

        def write_caption(_urls):
            Path(f"{output_path}.en.{sub_format}").write_text(fixture)

        instance = MagicMock()
        instance.download.side_effect = write_caption
        mock_ctor = MagicMock()
        mock_ctor.return_value.__enter__.return_value = instance
        return mock_ctor

    def test_txt_format_strips_vtt_to_txt_and_removes_source(self, tmp_path):
        from trans.downloader import extract_native_captions

        output_path = str(tmp_path / "video")
        with patch(
            "trans.downloader.yt_dlp.YoutubeDL",
            new=self._mock_ydl(output_path, self._VTT_FIXTURE, "vtt"),
        ):
            created = extract_native_captions(
                "https://example.com/v", output_path, output_format="txt", quiet=True
            )
        assert created.files == [Path(f"{output_path}.txt")]
        # Segments come from parsing the source VTT — non-empty since the
        # fixture has two cues.
        assert len(created.segments) == 2
        assert created.segments[0]["text"] == "Hello world"
        txt = Path(f"{output_path}.txt")
        assert txt.exists()
        body = txt.read_text()
        # Real text survives.
        assert "Hello world" in body
        assert "This is a test" in body
        # The 5 documented filter branches strip these:
        assert "WEBVTT" not in body
        assert "Kind:" not in body
        assert "NOTE" not in body
        assert "-->" not in body
        # Pure-digit cue numbers stripped (but text containing digits is fine).
        for line in body.splitlines():
            assert not line.isdigit()
        # Source has no "Language:" filter — line leaks through. Pinning the
        # current contract; if a future change adds a Language filter, flip
        # this to `not in body` and update the strip-filter docstring.
        assert "Language: en" in body
        # Source caption file removed (cleanup branch).
        assert not Path(f"{output_path}.en.vtt").exists()

    def test_vtt_format_renames_caption_no_txt(self, tmp_path):
        from trans.downloader import extract_native_captions

        output_path = str(tmp_path / "video")
        with patch(
            "trans.downloader.yt_dlp.YoutubeDL",
            new=self._mock_ydl(output_path, self._VTT_FIXTURE, "vtt"),
        ):
            created = extract_native_captions(
                "https://example.com/v", output_path, output_format="vtt", quiet=True
            )
        assert created.files == [Path(f"{output_path}.vtt")]
        assert len(created.segments) == 2  # VTT was parsed pre-rename.
        assert not Path(f"{output_path}.en.vtt").exists()  # renamed
        assert not Path(f"{output_path}.txt").exists()  # not written

    def test_srt_format_renames_caption_no_txt(self, tmp_path):
        """srt path mirrors vtt: rename .en.srt -> .srt, no .txt written.
        Regression test for the print mismatch — caller now sees the real file."""
        from trans.downloader import extract_native_captions

        output_path = str(tmp_path / "video")
        with patch(
            "trans.downloader.yt_dlp.YoutubeDL",
            new=self._mock_ydl(output_path, self._VTT_FIXTURE, "srt"),
        ):
            created = extract_native_captions(
                "https://example.com/v", output_path, output_format="srt", quiet=True
            )
        assert created.files == [Path(f"{output_path}.srt")]
        # SRT branch doesn't parse — yt-dlp owns that format. Segments stay empty.
        assert created.segments == []
        assert not Path(f"{output_path}.en.srt").exists()
        assert not Path(f"{output_path}.txt").exists()
        assert not Path(f"{output_path}.vtt").exists()

    def test_all_format_writes_txt_and_renames_vtt(self, tmp_path):
        from trans.downloader import extract_native_captions

        output_path = str(tmp_path / "video")
        with patch(
            "trans.downloader.yt_dlp.YoutubeDL",
            new=self._mock_ydl(output_path, self._VTT_FIXTURE, "vtt"),
        ):
            created = extract_native_captions(
                "https://example.com/v", output_path, output_format="all", quiet=True
            )
        # txt is written first, then sub is renamed — pin the order.
        assert created.files == [Path(f"{output_path}.txt"), Path(f"{output_path}.vtt")]
        assert len(created.segments) == 2
        assert not Path(f"{output_path}.en.vtt").exists()  # renamed, not removed

    def test_json_format_returns_empty(self, tmp_path):
        """json isn't derivable from VTT here — caller must fall through to Whisper.
        Today this silently returned True with zero files; new contract: empty list."""
        from trans.downloader import extract_native_captions

        output_path = str(tmp_path / "video")
        with patch(
            "trans.downloader.yt_dlp.YoutubeDL",
            new=self._mock_ydl(output_path, self._VTT_FIXTURE, "vtt"),
        ):
            created = extract_native_captions(
                "https://example.com/v", output_path, output_format="json", quiet=True
            )
        assert created.files == []
        # json branch still parses the VTT before deciding it has nothing to
        # write — segments are present but discarded by the empty-files check.
        assert len(created.segments) == 2
        assert not Path(f"{output_path}.txt").exists()
        assert not Path(f"{output_path}.vtt").exists()
        assert not Path(f"{output_path}.json").exists()

    def test_missing_caption_file_returns_empty(self, tmp_path):
        """If yt-dlp succeeds but no caption file appears (video has no subs),
        the function returns an empty list without raising — caller falls through."""
        from trans.downloader import extract_native_captions

        output_path = str(tmp_path / "video")
        instance = MagicMock()  # download is a no-op; writes nothing
        mock_ctor = MagicMock()
        mock_ctor.return_value.__enter__.return_value = instance
        with patch("trans.downloader.yt_dlp.YoutubeDL", new=mock_ctor):
            created = extract_native_captions(
                "https://example.com/v", output_path, output_format="txt", quiet=True
            )
        assert created.files == []
        assert created.segments == []
        assert not Path(f"{output_path}.txt").exists()


class TestParseVtt:
    """`parse_vtt` is the new VTT->segments parser used by cache writes."""

    def test_basic_two_cues(self):
        from trans.downloader import parse_vtt

        vtt = (
            "WEBVTT\n"
            "Kind: captions\n"
            "Language: en\n"
            "\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Hello world\n"
            "\n"
            "00:00:02.000 --> 00:00:04.500\n"
            "Second cue\n"
        )
        segs = parse_vtt(vtt)
        assert len(segs) == 2
        assert segs[0] == {"start": 0.0, "end": 2.0, "text": "Hello world"}
        assert segs[1] == {"start": 2.0, "end": 4.5, "text": "Second cue"}

    def test_strips_inline_timing_tags(self):
        """yt-dlp auto-captions sometimes embed `<00:00:01.000>` and `<c>` tags."""
        from trans.downloader import parse_vtt

        vtt = (
            "WEBVTT\n"
            "\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "<00:00:01.000><c.colorE5E5E5>Hello</c> <00:00:02.000><c>world</c>\n"
        )
        segs = parse_vtt(vtt)
        assert len(segs) == 1
        assert "<" not in segs[0]["text"]
        assert "Hello" in segs[0]["text"]
        assert "world" in segs[0]["text"]

    def test_empty_input_returns_empty_list(self):
        from trans.downloader import parse_vtt

        assert parse_vtt("") == []
        assert parse_vtt("not a vtt file at all") == []

    def test_garbage_blocks_skipped_silently(self):
        """A malformed cue block doesn't crash the parser; surrounding ones survive."""
        from trans.downloader import parse_vtt

        vtt = (
            "WEBVTT\n"
            "\n"
            "00:00:00.000 --> 00:00:01.000\n"
            "ok\n"
            "\n"
            "this block has no timing line\n"
            "\n"
            "00:00:02.000 --> 00:00:03.000\n"
            "also ok\n"
        )
        segs = parse_vtt(vtt)
        assert [s["text"] for s in segs] == ["ok", "also ok"]


class TestDownloaderErrors:
    """Typed exceptions in trans.downloader replace the old sys.exit(1) paths.

    Library callers (and the CLI's batch handler) can now distinguish
    TikTok IP blocks from generic download failures without parsing strings
    or catching SystemExit.
    """

    def test_get_video_info_raises_tiktok_blocked_on_ip_block(self, monkeypatch):
        from trans import downloader as dl_mod

        class FakeYDL:
            def __init__(self, *_a, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def extract_info(self, *_a, **_kw):
                raise dl_mod.yt_dlp.utils.DownloadError(
                    "ERROR: [TikTok] 123: Your IP address is blocked from accessing this post"
                )

        monkeypatch.setattr(dl_mod.yt_dlp, "YoutubeDL", FakeYDL)

        with pytest.raises(dl_mod.TikTokIPBlockedError):
            dl_mod.get_video_info("https://www.tiktok.com/@u/video/123", quiet=True)

    def test_get_video_info_raises_downloader_error_on_generic_failure(self, monkeypatch):
        from trans import downloader as dl_mod

        class FakeYDL:
            def __init__(self, *_a, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def extract_info(self, *_a, **_kw):
                raise dl_mod.yt_dlp.utils.DownloadError("ERROR: video unavailable")

        monkeypatch.setattr(dl_mod.yt_dlp, "YoutubeDL", FakeYDL)

        # Plain DownloaderError, not TikTokIPBlockedError, on a non-TikTok URL.
        with pytest.raises(dl_mod.DownloaderError) as exc_info:
            dl_mod.get_video_info("https://www.youtube.com/watch?v=abc12345678", quiet=True)
        assert not isinstance(exc_info.value, dl_mod.TikTokIPBlockedError)

    def test_batch_continues_after_one_url_raises_downloader_error(self, monkeypatch, tmp_path):
        """Per the task spec: a DownloaderError on URL 1 must NOT kill URLs 2+.

        Pre-fix, downloader.py called sys.exit(1) which raised SystemExit
        and bypassed the batch loop's `except Exception` clause, aborting
        the whole batch. The typed exception now flows through the
        DownloaderError handler.
        """
        from typer.testing import CliRunner

        from trans import cli as cli_mod
        from trans.downloader import DownloaderError

        calls = {"url1": 0, "url2": 0}

        def fake_get_video_info(url, **_kw):
            if "111" in url:
                calls["url1"] += 1
                raise DownloaderError("Error fetching video info: simulated")
            calls["url2"] += 1
            return {"title": "ok", "duration": 1.0}

        def fake_extract_native_captions(url, out, fmt, quiet=True):
            # Pretend URL 2 has perfectly serviceable native captions.
            from trans.downloader import NativeCaptureResult

            txt = Path(f"{out}.txt")
            txt.write_text("hello", encoding="utf-8")
            return NativeCaptureResult([txt], [])

        monkeypatch.setattr(cli_mod, "get_video_info", fake_get_video_info)
        monkeypatch.setattr(cli_mod, "extract_native_captions", fake_extract_native_captions)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        url1 = "https://example.com/111"
        url2 = "https://example.com/222"
        result = runner.invoke(cli_mod.app, ["transcribe", url1, url2, "--no-cache"])

        # One URL failed (exit 1) but the second still ran. The prefetch
        # loop calls get_video_info once per URL silently; the main loop
        # then re-runs each URL — so the failing URL gets hit twice, the
        # succeeding one twice too. The contract under test is "URL 2 was
        # processed", not the exact call count.
        assert result.exit_code == 1
        assert calls["url1"] >= 1
        assert calls["url2"] >= 1
        assert "Summary: 1 succeeded, 1 failed" in result.stdout


class TestVerboseFlag:
    """`-v/--verbose` prints tracebacks to stderr; stdout stays clean."""

    @staticmethod
    def _run(monkeypatch, tmp_path, args, raise_in_native_captions=True):
        from typer.testing import CliRunner

        from trans import cli as cli_mod

        def fake_get_video_info(url, **_kw):
            return {"title": "ok", "duration": 1.0}

        def fake_extract_native_captions(url, out, fmt, quiet=True):
            from trans.downloader import NativeCaptureResult

            if raise_in_native_captions:
                raise RuntimeError("boom-from-extract")
            txt = Path(f"{out}.txt")
            txt.write_text("hi", encoding="utf-8")
            return NativeCaptureResult([txt], [])

        monkeypatch.setattr(cli_mod, "get_video_info", fake_get_video_info)
        monkeypatch.setattr(cli_mod, "extract_native_captions", fake_extract_native_captions)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        return runner.invoke(cli_mod.app, args)

    def test_verbose_prints_traceback_to_stderr(self, monkeypatch, tmp_path):
        # CliRunner separates stdout and stderr on result.stdout / result.stderr
        # so we can assert the stream-separation contract directly.
        result = self._run(
            monkeypatch,
            tmp_path,
            ["-v", "transcribe", "https://example.com/v", "--no-cache"],
        )
        # Friendly one-liner on stdout via typer.echo.
        assert "boom-from-extract" in result.stdout
        # Traceback on stderr — and NOT on stdout (preserves machine-parseable stdout).
        assert "Traceback" in result.stderr
        assert "boom-from-extract" in result.stderr
        assert "Traceback" not in result.stdout
        assert result.exit_code == 1

    def test_no_verbose_omits_traceback(self, monkeypatch, tmp_path):
        result = self._run(
            monkeypatch,
            tmp_path,
            ["transcribe", "https://example.com/v", "--no-cache"],
        )
        # Friendly one-liner present; traceback absent from both streams.
        assert "boom-from-extract" in result.stdout
        assert "Traceback" not in result.stderr
        assert "Traceback" not in result.stdout
        assert result.exit_code == 1


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
        from trans.downloader import NativeCaptureResult as _NCR

        monkeypatch.setattr(cli_mod, "extract_native_captions", lambda *a, **kw: _NCR([], []))

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

    def test_native_captions_prints_actual_files(self, tmp_path, monkeypatch, capsys):
        """Regression: the native-captions success branch must print the file(s) the
        downloader actually created, not a hardcoded `['txt', 'vtt']` extras list.

        Bug-as-was: `-f srt URL` on a native-captions hit printed `.srt` correctly
        (single-format path), but `-f all URL` and `-f json URL` broke. After the
        refactor, the cli iterates the list returned by `extract_native_captions`
        — this test pins that contract under the `quiet=False` print loop, where
        the bug actually lives. Single-format `srt` is the cleanest probe: any
        leakage of hardcoded `txt`/`vtt` shows up immediately in stdout.
        """
        from trans import cli as cli_mod
        from trans.cli import _process_url

        calls, kwargs = self._build_fixtures(tmp_path, monkeypatch)
        kwargs["quiet"] = False  # the bug lives in the `if not quiet:` print loop
        kwargs["fmt"] = "srt"

        def fake_native(url, out_base, fmt, quiet):
            from trans.downloader import NativeCaptureResult

            p = Path(f"{out_base}.srt")
            p.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
            return NativeCaptureResult([p], [])

        monkeypatch.setattr(cli_mod, "extract_native_captions", fake_native)

        url = "https://youtube.com/watch?v=fakefakefake"
        ok = _process_url(url, **kwargs)
        assert ok is True

        out = capsys.readouterr().out
        # The actual file is reported.
        assert ".srt" in out
        # Hardcoded `["txt", "vtt"]` extras list is gone — neither extension leaks
        # into stdout when the user requested `-f srt`.
        assert ".txt" not in out
        assert ".vtt" not in out


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
        from trans.downloader import NativeCaptureResult as _NCR

        monkeypatch.setattr(cli_mod, "extract_native_captions", lambda *a, **kw: _NCR([], []))

        class StubEngine:
            def transcribe(self, audio, *, language=None, quiet=False, translate=False):
                return (
                    [{"start": 0.0, "end": 1.0, "text": "x"}],
                    {"language": "en"},
                )

        monkeypatch.setattr(cli_mod, "TranscriptionEngine", lambda model, **_kw: StubEngine())
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

    def test_no_keep_audio_beats_config_true(self, tmp_path, monkeypatch):
        """`--no-keep-audio` must turn off keep_audio when cfg has it on.

        Without the slash-form CLI declaration, Typer wouldn't expose a
        `--no-keep-audio` flag — `cfg.keep_audio=true` would be permanent
        until the user edits the config file. The slash form gives a
        per-invocation escape hatch.
        """
        runner, cli_mod = self._setup_stubs(tmp_path, monkeypatch, cfg_keep_audio=True)
        result = runner.invoke(
            cli_mod.app,
            [
                "transcribe",
                "--output-dir",
                str(tmp_path),
                "--no-keep-audio",
                "--quiet",
                "https://www.youtube.com/watch?v=fakefakefake",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not (tmp_path / "stub.audio.mp3").exists(), (
            "--no-keep-audio must override cfg.keep_audio=True"
        )


class TestNoFlagOverrides:
    """`--no-clipboard` and `--no-quiet` paired with config truthy values.

    `--no-keep-audio` is covered by `TestConfigKeepAudioResolution` (same
    fixture, same shape) — these two tests round out the three
    config-aware bool flags.
    """

    @staticmethod
    def _setup_stubs(tmp_path, monkeypatch, *, cfg_clipboard=False, cfg_quiet=False):
        from typer.testing import CliRunner

        from trans import cli as cli_mod
        from trans.cache import CacheManager
        from trans.config import Config

        monkeypatch.setattr(
            cli_mod,
            "load_config",
            lambda: Config(clipboard=cfg_clipboard, quiet=cfg_quiet),
        )

        copies: list[str] = []
        monkeypatch.setattr(cli_mod, "HAS_PYPERCLIP", True)

        class FakePyperclip:
            def copy(self, text):
                copies.append(text)

        monkeypatch.setattr(cli_mod, "pyperclip", FakePyperclip())

        # Native captions branch returns one txt file so clipboard reads it.
        def fake_native(url, out, fmt, quiet=True):
            from trans.downloader import NativeCaptureResult

            txt = Path(f"{out}.txt")
            txt.write_text("hello\nworld\n", encoding="utf-8")
            return NativeCaptureResult([txt], [])

        monkeypatch.setattr(cli_mod, "extract_native_captions", fake_native)
        monkeypatch.setattr(
            cli_mod, "get_video_info", lambda url, **_: {"title": "stub", "duration": 1}
        )
        monkeypatch.setattr(
            cli_mod, "CacheManager", lambda: CacheManager(db_path=tmp_path / "c.db")
        )
        return CliRunner(), cli_mod, copies

    def test_no_clipboard_beats_config_true(self, tmp_path, monkeypatch):
        runner, cli_mod, copies = self._setup_stubs(tmp_path, monkeypatch, cfg_clipboard=True)
        result = runner.invoke(
            cli_mod.app,
            [
                "transcribe",
                "--output-dir",
                str(tmp_path),
                "--no-clipboard",
                "--quiet",
                "https://www.youtube.com/watch?v=fakefakefake",
            ],
        )
        assert result.exit_code == 0, result.output
        assert copies == [], "--no-clipboard must suppress the clipboard copy"

    def test_clipboard_runs_when_config_true_and_no_override(self, tmp_path, monkeypatch):
        """Positive control — cfg=True with no override still copies."""
        runner, cli_mod, copies = self._setup_stubs(tmp_path, monkeypatch, cfg_clipboard=True)
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
        assert copies == ["hello\nworld\n"], "cfg.clipboard=True must trigger a copy"

    def test_no_quiet_beats_config_true(self, tmp_path, monkeypatch):
        """`--no-quiet` must surface the informational output even when cfg.quiet=True."""
        runner, cli_mod, _ = self._setup_stubs(tmp_path, monkeypatch, cfg_quiet=True)
        result = runner.invoke(
            cli_mod.app,
            [
                "transcribe",
                "--output-dir",
                str(tmp_path),
                "--no-quiet",
                "https://www.youtube.com/watch?v=fakefakefake",
            ],
        )
        assert result.exit_code == 0, result.output
        # The "→ Checking for native captions..." line is one of the first
        # informational prints. If quiet=True wins, it doesn't appear.
        assert "Transcription complete" in result.output, (
            "--no-quiet must restore informational output even when cfg.quiet=True"
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

    def test_native_captions_cached_on_put(self, tmp_path, monkeypatch):
        """Native-captions success must populate the cache with source='native'.

        Flipped from `test_native_captions_no_longer_cached` after
        task-cache-native-captions landed. The CLI now caches the parsed
        VTT segments so a subsequent run renders any format from canonical
        segments without re-fetching.
        """
        from trans import cli as cli_mod
        from trans.cache import CacheManager
        from trans.config import Config
        from trans.downloader import NativeCaptureResult

        # Stub the native-captions write so it returns success without network.
        # New contract: NativeCaptureResult(files, segments).
        parsed_segments = [
            {"start": 0.0, "end": 2.0, "text": "captioned content"},
        ]

        def fake_native(url, out_base, fmt, quiet):
            p = Path(f"{out_base}.txt")
            p.write_text("captioned content", encoding="utf-8")
            return NativeCaptureResult([p], parsed_segments)

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
            quiet=False,
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
        # Critical: native-captions success populates the cache with source='native'.
        from trans.utils import get_video_id

        hit = cache.get(get_video_id(url), ttl_days=30)
        assert hit is not None
        assert hit.source == "native"
        assert hit.segments == parsed_segments

    def test_native_captions_empty_segments_skip_cache_write(self, tmp_path, monkeypatch):
        """Malformed VTT yields empty segments — must NOT pollute the cache."""
        from trans import cli as cli_mod
        from trans.cache import CacheManager
        from trans.config import Config
        from trans.downloader import NativeCaptureResult

        def fake_native(url, out_base, fmt, quiet):
            p = Path(f"{out_base}.txt")
            p.write_text("only-text-no-segments", encoding="utf-8")
            return NativeCaptureResult([p], [])  # empty segments

        monkeypatch.setattr(cli_mod, "extract_native_captions", fake_native)
        monkeypatch.setattr(
            cli_mod,
            "get_video_info",
            lambda url, cookies=None, quiet=False: {"title": "x", "duration": 1},
        )

        class StubEngine:
            def transcribe(self, *a, **kw):
                raise AssertionError("unreachable")

        cache = CacheManager(db_path=tmp_path / "c.db")
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
            config=Config(),
        )
        assert ok is True
        from trans.utils import get_video_id

        assert cache.get(get_video_id(url), ttl_days=30) is None, (
            "empty segments must not result in a cache row"
        )

    def test_native_captions_offline_replay_from_cache(self, tmp_path, monkeypatch):
        """After a native-captions run, a second run hits the cache even if extract crashes."""
        from trans import cli as cli_mod
        from trans.cache import CacheManager
        from trans.config import Config
        from trans.downloader import NativeCaptureResult

        parsed_segments = [{"start": 0.0, "end": 1.0, "text": "replay"}]

        def fake_native(url, out_base, fmt, quiet):
            p = Path(f"{out_base}.txt")
            p.write_text("replay", encoding="utf-8")
            return NativeCaptureResult([p], parsed_segments)

        monkeypatch.setattr(cli_mod, "extract_native_captions", fake_native)
        monkeypatch.setattr(
            cli_mod,
            "get_video_info",
            lambda url, cookies=None, quiet=False: {"title": "Replay", "duration": 1},
        )

        class UnreachableEngine:
            def transcribe(self, *a, **kw):
                raise AssertionError("must not transcribe on cache hit")

        cache = CacheManager(db_path=tmp_path / "c.db")
        url = "https://www.youtube.com/watch?v=fakefakefake"
        kwargs = dict(
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
            engine=UnreachableEngine(),
            cache=cache,
            config=Config(),
        )
        # First run: populate cache.
        assert cli_mod._process_url(url, **kwargs) is True

        # Now simulate "extract crashes" (no network). The cache hit should
        # short-circuit before extract_native_captions is even called.
        def crashing_extract(*a, **kw):
            raise RuntimeError("would have hit the network")

        monkeypatch.setattr(cli_mod, "extract_native_captions", crashing_extract)
        # Re-render to a fresh path so we can assert the output appeared.
        assert cli_mod._process_url(url, **kwargs) is True

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


class TestWriteOutput:
    """Direct tests for trans.formatter.write_output.

    Cache round-trip tests cover this indirectly; this class pins down the
    formatter's per-format behavior and the diarized branches so the contract
    is testable without going through CacheManager.
    """

    @staticmethod
    def _plain_segments():
        return [
            {"start": 0.0, "end": 2.5, "text": "first line"},
            {"start": 2.5, "end": 5.0, "text": "second line"},
        ]

    @staticmethod
    def _diarized_segments():
        return [
            {"start": 0.0, "end": 1.0, "text": "alpha", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "beta", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.0, "text": "gamma", "speaker": "SPEAKER_01"},
            {"start": 3.0, "end": 4.0, "text": "delta", "speaker": "SPEAKER_00"},
        ]

    @pytest.mark.parametrize(
        "fmt,ext",
        [
            ("txt", ".txt"),
            ("srt", ".srt"),
            ("vtt", ".vtt"),
            ("json", ".json"),
        ],
    )
    def test_each_format_non_diarized(self, tmp_path, fmt, ext):
        from trans.formatter import write_output

        base = str(tmp_path / "out")
        created = write_output(self._plain_segments(), base, fmt)
        assert len(created) == 1
        assert created[0].suffix == ext
        body = created[0].read_text()
        assert "first line" in body
        assert "second line" in body
        if fmt == "srt":
            assert "00:00:00,000 --> 00:00:02,500" in body  # comma decimal
            assert body.startswith("1\n")  # 1-indexed sequence at the top
        elif fmt == "vtt":
            assert body.startswith("WEBVTT\n\n")
            assert "00:00:00.000 --> 00:00:02.500" in body  # dot decimal
        elif fmt == "json":
            parsed = _json.loads(body)
            assert parsed["diarization"] is False
            assert len(parsed["segments"]) == 2

    def test_format_all_writes_four_files(self, tmp_path):
        from trans.formatter import write_output

        base = str(tmp_path / "out")
        created = write_output(self._plain_segments(), base, "all")
        # Order isn't a documented contract; assert membership not sequence.
        assert len(created) == 4
        assert {p.suffix for p in created} == {".txt", ".srt", ".vtt", ".json"}
        for p in created:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_txt_groups_same_speaker_runs(self, tmp_path):
        """4 segments alternating A,A,B,A => 3 speaker headers, not 4."""
        from trans.formatter import write_output

        base = str(tmp_path / "out")
        created = write_output(self._diarized_segments(), base, "txt", diarized=True)
        body = created[0].read_text()
        assert body.count("[Speaker 1]") == 2  # opening run + reappearance after Speaker 2
        assert body.count("[Speaker 2]") == 1
        # Consecutive same-speaker segments share one header.
        assert "[Speaker 1]\nalpha\nbeta\n" in body

    def test_srt_prefixes_speaker_when_diarized(self, tmp_path):
        from trans.formatter import write_output

        base = str(tmp_path / "out")
        created = write_output(self._diarized_segments(), base, "srt", diarized=True)
        body = created[0].read_text()
        assert "[Speaker 1] alpha" in body
        assert "[Speaker 1] beta" in body
        assert "[Speaker 2] gamma" in body
        assert "[Speaker 1] delta" in body
        # 1-indexed sequence numbers, one per segment.
        assert body.startswith("1\n")
        assert "\n4\n" in body

    def test_vtt_uses_voice_tag_when_diarized(self, tmp_path):
        from trans.formatter import write_output

        base = str(tmp_path / "out")
        created = write_output(self._diarized_segments(), base, "vtt", diarized=True)
        body = created[0].read_text()
        assert body.startswith("WEBVTT\n\n")
        assert "<v Speaker 1>alpha" in body
        assert "<v Speaker 2>gamma" in body
        # Non-diarized run shouldn't sneak the voice tag in.
        plain = write_output(self._plain_segments(), str(tmp_path / "plain"), "vtt", diarized=False)
        assert "<v " not in plain[0].read_text()

    def test_json_includes_info_when_provided(self, tmp_path):
        from trans.formatter import write_output

        base = str(tmp_path / "out")
        info = {"language": "en", "language_probability": 0.99, "duration": 12.3}
        created = write_output(self._plain_segments(), base, "json", info=info)
        parsed = _json.loads(created[0].read_text())
        assert parsed["language"] == "en"
        assert parsed["language_probability"] == 0.99
        assert parsed["duration"] == 12.3
        assert parsed["diarization"] is False

        # Without info, those keys are absent.
        plain = write_output(self._plain_segments(), str(tmp_path / "noinfo"), "json")
        parsed_plain = _json.loads(plain[0].read_text())
        assert "language" not in parsed_plain
        assert "language_probability" not in parsed_plain
        assert "duration" not in parsed_plain

        # Empty info dict is falsy; source uses `if info:`, so the keys are
        # absent same as None. Guards against a refactor to `is not None`.
        empty = write_output(self._plain_segments(), str(tmp_path / "emptyinfo"), "json", info={})
        parsed_empty = _json.loads(empty[0].read_text())
        assert "language" not in parsed_empty
        assert "language_probability" not in parsed_empty
        assert "duration" not in parsed_empty

    def test_json_speakers_list_when_diarized(self, tmp_path):
        """Sort is on raw pyannote IDs; zero-padded form (SPEAKER_00..) means
        alphabetical == numeric. Non-padded IDs would alphabetize wrong."""
        from trans.formatter import write_output

        base = str(tmp_path / "out")
        created = write_output(self._diarized_segments(), base, "json", diarized=True)
        parsed = _json.loads(created[0].read_text())
        assert parsed["diarization"] is True
        assert parsed["speakers"] == ["Speaker 1", "Speaker 2"]

    def test_empty_segments_writes_files_without_crash(self, tmp_path):
        from trans.formatter import write_output

        base = str(tmp_path / "out")
        created = write_output([], base, "all")
        assert len(created) == 4
        for p in created:
            assert p.exists()
        # VTT still gets its header; JSON is well-formed empty.
        vtt = next(p for p in created if p.suffix == ".vtt")
        assert vtt.read_text() == "WEBVTT\n\n"
        js = next(p for p in created if p.suffix == ".json")
        parsed = _json.loads(js.read_text())
        assert parsed == {"diarization": False, "segments": []}


class TestConfigPersistence:
    """load/save round-trip and resilience to missing/malformed files."""

    def test_load_missing_file_returns_defaults(self, tmp_path):
        from trans.config import Config, load_config

        cfg = load_config(path=tmp_path / "nope.toml")
        assert cfg == Config()

    def test_load_malformed_toml_returns_defaults(self, tmp_path):
        """Silent-fallback contract: a corrupt config must not crash the CLI.

        If we ever want to warn on malformed TOML, this test must change.
        """
        from trans.config import Config, load_config

        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text("this is [garbage and not = valid toml @@@\n")
        assert load_config(path=cfg_path) == Config()

    def test_round_trip_all_fields(self, tmp_path):
        from trans.config import (
            CacheConfig,
            Config,
            DiarizationConfig,
            load_config,
            save_config,
        )

        cfg_path = tmp_path / "config.toml"
        original = Config(
            model="large",
            format="srt",
            language="es",
            output_dir="/tmp/out",
            clipboard=True,
            quiet=True,
            keep_audio=True,
            cache=CacheConfig(ttl_days=7),
            diarization=DiarizationConfig(hf_token="hf_abc123"),
        )
        save_config(original, path=cfg_path)
        loaded = load_config(path=cfg_path)
        assert loaded == original
        # type(...) is int — not isinstance, which would silently pass for bool.
        assert type(loaded.cache.ttl_days) is int


class TestConfigSetValue:
    """set_config_value typed-coercion, dotted-key routing, error handling."""

    def test_unknown_key_raises_value_error(self, tmp_path):
        from trans.config import set_config_value

        with pytest.raises(ValueError, match="Unknown config key"):
            set_config_value("bogus.key", "x", path=tmp_path / "config.toml")

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("YES", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("anything", False),
            ("", False),
        ],
    )
    def test_bool_parsing_variants(self, tmp_path, raw, expected):
        from trans.config import load_config, set_config_value

        cfg_path = tmp_path / "config.toml"
        cfg = set_config_value("clipboard", raw, path=cfg_path)
        assert cfg.clipboard is expected
        # Round-trip too — the TOML writer must preserve the bool, not the raw string.
        assert load_config(path=cfg_path).clipboard is expected

    def test_int_parsing_for_ttl_days(self, tmp_path):
        """Contract guard: cache.ttl_days survives as int through set + round-trip.

        save_config writes the bare int via f-string (no quotes); load_config
        reads it back via tomllib which produces an int. Expected green on
        first run — if this fails, the bug is in test or in save_config drift,
        not in tomllib.
        """
        from trans.config import load_config, set_config_value

        cfg_path = tmp_path / "config.toml"
        cfg = set_config_value("cache.ttl_days", "30", path=cfg_path)
        assert cfg.cache.ttl_days == 30
        assert type(cfg.cache.ttl_days) is int

        loaded = load_config(path=cfg_path)
        assert loaded.cache.ttl_days == 30
        assert type(loaded.cache.ttl_days) is int

    def test_int_parsing_invalid_raises(self, tmp_path):
        from trans.config import set_config_value

        with pytest.raises(ValueError):
            set_config_value("cache.ttl_days", "thirty", path=tmp_path / "config.toml")

    def test_nested_key_routes_to_subsection(self, tmp_path):
        from trans.config import Config, load_config, set_config_value

        cfg_path = tmp_path / "config.toml"
        cfg = set_config_value("cache.ttl_days", "5", path=cfg_path)
        assert cfg.cache.ttl_days == 5
        # Must NOT have set a flat attr on Config (which would shadow the subsection).
        assert not hasattr(Config(), "ttl_days")

        cfg = set_config_value("diarization.hf_token", "tok_xyz", path=cfg_path)
        assert cfg.diarization.hf_token == "tok_xyz"
        # Both subsections persist together across set_config_value calls.
        loaded = load_config(path=cfg_path)
        assert loaded.cache.ttl_days == 5
        assert loaded.diarization.hf_token == "tok_xyz"


class TestClipboardErrorHandling:
    """The narrow `except pyperclip.PyperclipException` keeps the friendly
    warning path for pyperclip's own errors (no xclip on headless Linux,
    Windows clipboard race, etc.) while letting genuinely unexpected
    exceptions surface to the verbose handler.
    """

    def test_pyperclip_exception_prints_warning_and_continues(self, monkeypatch):
        import pyperclip

        from trans import cli as cli_mod

        monkeypatch.setattr(cli_mod, "HAS_PYPERCLIP", True)

        class _FakePyperclip:
            PyperclipException = pyperclip.PyperclipException

            def copy(self, _text):
                raise pyperclip.PyperclipException("no clipboard backend")

        monkeypatch.setattr(cli_mod, "pyperclip", _FakePyperclip())

        # Should not raise — the warning prints and execution continues.
        cli_mod._copy_to_clipboard("hello", quiet=False)


class TestPythonDashMEntrypoint:
    """`python -m trans` must work end-to-end via trans/__main__.py.

    Replaces the legacy `trans_cli.py` shim. The PyPI `console_scripts`
    entry already routes `trans` directly to `trans.cli:app`; this
    test guards the dev invocation.
    """

    def test_python_dash_m_trans_version(self, tmp_path):
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "trans", "--version"],
            capture_output=True,
            text=True,
            cwd=tmp_path,  # CWD-independent
        )
        assert result.returncode == 0, result.stderr
        # The version line gets prefixed with "trans " from `_app_callback`.
        assert "trans " in result.stdout


class TestDeviceComputeFlags:
    """`--device` / `--compute-type` CLI flags, config keys, and engine wiring."""

    def test_engine_accepts_device_and_compute_type(self):
        """Construction must not crash — the model is lazy-loaded, so no
        faster-whisper invocation happens until `.model` is accessed."""
        from trans.transcriber import TranscriptionEngine

        engine = TranscriptionEngine("base", device="cpu", compute_type="int8")
        assert engine.device == "cpu"
        assert engine.compute_type == "int8"
        assert engine._model is None  # still lazy

    def test_config_round_trip_for_device_and_compute_type(self, tmp_path):
        from trans.config import load_config, set_config_value

        cfg_path = tmp_path / "config.toml"
        set_config_value("device", "cuda", path=cfg_path)
        set_config_value("compute_type", "float16", path=cfg_path)
        loaded = load_config(path=cfg_path)
        assert loaded.device == "cuda"
        assert loaded.compute_type == "float16"

    def test_invalid_device_rejected_by_cli(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from trans import cli as cli_mod

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            cli_mod.app,
            ["transcribe", "--device", "tpu", "--no-cache", "https://example.com/v"],
        )
        assert result.exit_code == 1
        assert "Invalid device" in result.stdout


class TestCacheManagerLifecycle:
    """Lifecycle ops: clear, stats, put-overwrite.

    Complements TestCacheRoundTrip (which covers store/retrieve). Defends the
    lazy-init invariant — methods on a never-touched DB must not silently
    create the file.
    """

    @staticmethod
    def _sample_segments(text="Hello"):
        return [{"start": 0.0, "end": 1.0, "text": text}]

    def test_clear_returns_row_count(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")

        # Never-touched DB: clear() short-circuits without creating the file.
        # Public contract — internal flag (_schema_ensured) is incidental.
        assert cache.clear() == 0
        assert not (tmp_path / "c.db").exists()

        cache.put("yt_a", "u1", "T1", self._sample_segments("a"))
        cache.put("yt_b", "u2", "T2", self._sample_segments("b"))
        assert cache.clear() == 2
        assert cache.get("yt_a") is None
        assert cache.get("yt_b") is None
        # Subsequent clear on an empty table returns 0 (table exists, no rows).
        assert cache.clear() == 0

    def test_stats_shape_after_writes(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")

        # Never-touched: zero-state without creating the file (same public
        # contract as clear() above). Schema includes the active/expired
        # split — both zero on an empty DB.
        stats = cache.stats()
        assert stats == {
            "count": 0,
            "count_active": 0,
            "count_expired": 0,
            "size_mb": 0.0,
            "oldest": None,
            "newest": None,
        }
        assert not (tmp_path / "c.db").exists()

        cache.put("yt_a", "u1", "T1", self._sample_segments("a"))
        cache.put("yt_b", "u2", "T2", self._sample_segments("b"))
        stats = cache.stats()
        assert set(stats.keys()) == {
            "count",
            "count_active",
            "count_expired",
            "size_mb",
            "oldest",
            "newest",
        }
        assert stats["count"] == 2
        # Without ttl_days, the active/expired split defaults to count/0.
        assert stats["count_active"] == 2
        assert stats["count_expired"] == 0
        assert stats["size_mb"] > 0
        assert stats["oldest"] is not None and stats["newest"] is not None
        assert stats["oldest"] <= stats["newest"]

    def test_put_overwrites_existing_row(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        cache.put("yt_x", "u1", "First", self._sample_segments("alpha"))
        cache.put("yt_x", "u2", "Second", self._sample_segments("beta"))

        hit = cache.get("yt_x")
        assert hit is not None
        assert hit.title == "Second"
        assert hit.segments == [{"start": 0.0, "end": 1.0, "text": "beta"}]
        assert cache.stats()["count"] == 1


class TestCachePrune:
    """`prune` + auto-prune-on-put + stats() active/expired split."""

    @staticmethod
    def _insert_backdated(cache, video_id: str, days_old: int) -> None:
        """Insert a row with a synthetic `created_at` so we can simulate aging.

        `cache.put` uses CURRENT_TIMESTAMP; bypass that by going straight to
        the connection for these probe rows.
        """
        import sqlite3

        cache._ensure_schema()
        with sqlite3.connect(cache._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO transcripts "
                "(video_id, url, title, segments_json, info_json, model, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', ?))",
                (
                    video_id,
                    "u",
                    "T",
                    '[{"start": 0.0, "end": 1.0, "text": "x"}]',
                    None,
                    "base",
                    "whisper",
                    f"-{days_old} days",
                ),
            )

    def test_prune_removes_only_expired_rows(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        self._insert_backdated(cache, "yt_old1", 60)
        self._insert_backdated(cache, "yt_old2", 45)
        self._insert_backdated(cache, "yt_fresh", 1)

        removed = cache.prune(ttl_days=30)
        assert removed == 2

        # The fresh row survives.
        assert cache.get("yt_fresh", ttl_days=30) is not None
        # The old rows are gone — gone enough that even a query with a
        # very-permissive ttl can't see them.
        assert cache.get("yt_old1", ttl_days=10000) is None
        assert cache.get("yt_old2", ttl_days=10000) is None

    def test_prune_on_empty_db_returns_zero(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        # Don't ensure schema first — confirms the no-file shortcut works.
        assert cache.prune(ttl_days=7) == 0

    def test_put_auto_prunes_expired(self, tmp_path):
        """Every put() opportunistically evicts rows older than ttl_days.

        Without this, the SQLite file grows unbounded — `get()` filters by
        TTL on read but never deletes. The auto-prune keeps the file
        bounded between manual `trans cache prune` invocations.
        """
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        for i in range(5):
            self._insert_backdated(cache, f"yt_old{i}", 60)

        # One fresh put with ttl_days=30 should sweep all 5 expired rows.
        cache.put(
            "yt_fresh",
            "u",
            "Fresh",
            [{"start": 0.0, "end": 1.0, "text": "hi"}],
            ttl_days=30,
        )

        stats = cache.stats()
        assert stats["count"] == 1
        assert stats["count_active"] == 1

    def test_stats_splits_active_and_expired(self, tmp_path):
        from trans.cache import CacheManager

        cache = CacheManager(db_path=tmp_path / "c.db")
        self._insert_backdated(cache, "yt_old", 60)
        self._insert_backdated(cache, "yt_fresh", 1)

        stats = cache.stats(ttl_days=30)
        assert stats["count"] == 2
        assert stats["count_active"] == 1
        assert stats["count_expired"] == 1


class TestConfigFilePermissions:
    """`save_config` must produce a 0o600 file and stay tight across overwrites.

    Threat model: single-user CLI on a shared Unix host. The token in
    `diarization.hf_token` is plaintext; the file mode is the only barrier
    against a sibling-process-as-same-user read.
    """

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX permission bits are advisory on Windows",
    )
    def test_save_config_writes_mode_0o600(self, tmp_path):
        import os
        import stat

        from trans.config import Config, save_config

        cfg_path = tmp_path / "config.toml"
        save_config(Config(), path=cfg_path)
        mode = stat.S_IMODE(os.stat(cfg_path).st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX permission bits are advisory on Windows",
    )
    def test_save_config_chmod_runs_on_overwrite(self, tmp_path):
        """Defends against the regression where the chmod only fires at create."""
        import os
        import stat

        from trans.config import Config, save_config

        cfg_path = tmp_path / "config.toml"
        save_config(Config(), path=cfg_path)
        os.chmod(cfg_path, 0o644)
        save_config(Config(), path=cfg_path)
        mode = stat.S_IMODE(os.stat(cfg_path).st_mode)
        assert mode == 0o600, f"overwrite did not re-tighten mode (got {oct(mode)})"

    def test_save_config_atomic_no_tempfile_left_behind(self, tmp_path):
        """Atomic-write contract: parent dir must contain only the final file."""
        from trans.config import Config, save_config

        cfg_path = tmp_path / "config.toml"
        save_config(Config(), path=cfg_path)
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["config.toml"], f"tempfile leaked into config dir: {names}"


class TestReadHfCacheToken:
    """Direct coverage for the cache-file source. Without these the precedence
    tests monkeypatch the helper away — a typo in the path string would
    silently pass CI and break only in production.
    """

    def test_returns_token_when_file_exists(self, tmp_path, monkeypatch):
        from trans.diarizer import _read_hf_cache_token

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cache_dir = tmp_path / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True)
        (cache_dir / "token").write_text("hf_real_token\n")

        assert _read_hf_cache_token() == "hf_real_token"

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        from trans.diarizer import _read_hf_cache_token

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _read_hf_cache_token() is None

    def test_returns_none_on_permission_error(self, tmp_path, monkeypatch):
        """Unreadable cache file must degrade quietly, not bubble OSError up."""
        from trans import diarizer

        def raise_perm(*_a, **_kw):
            raise PermissionError("nope")

        monkeypatch.setattr(diarizer.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(diarizer.Path, "read_text", raise_perm)
        assert diarizer._read_hf_cache_token() is None


class TestGetHfTokenPrecedence:
    """`get_hf_token` source precedence: env > config > cache."""

    @staticmethod
    def _clear_envs(monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    def test_hf_token_env_wins_over_config_and_cache(self, monkeypatch):
        from trans import diarizer

        monkeypatch.setenv("HF_TOKEN", "env_tok")
        monkeypatch.setattr(diarizer, "_read_hf_cache_token", lambda: "cache_tok")
        assert diarizer.get_hf_token("cfg_tok") == "env_tok"

    def test_hugging_face_hub_token_env_wins_over_config(self, monkeypatch):
        from trans import diarizer

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub_tok")
        monkeypatch.setattr(diarizer, "_read_hf_cache_token", lambda: "cache_tok")
        assert diarizer.get_hf_token("cfg_tok") == "hub_tok"

    def test_config_wins_when_no_env(self, monkeypatch):
        from trans import diarizer

        self._clear_envs(monkeypatch)
        monkeypatch.setattr(diarizer, "_read_hf_cache_token", lambda: "cache_tok")
        assert diarizer.get_hf_token("cfg_tok") == "cfg_tok"

    def test_cache_used_when_no_env_or_config(self, monkeypatch):
        from trans import diarizer

        self._clear_envs(monkeypatch)
        monkeypatch.setattr(diarizer, "_read_hf_cache_token", lambda: "cache_tok")
        assert diarizer.get_hf_token("") == "cache_tok"

    def test_returns_none_when_nothing_set(self, monkeypatch):
        from trans import diarizer

        self._clear_envs(monkeypatch)
        monkeypatch.setattr(diarizer, "_read_hf_cache_token", lambda: None)
        assert diarizer.get_hf_token("") is None


class TestTranscribeGateUsesConfigToken:
    """End-to-end: the diarization gate at `cli.transcribe` must honor
    `cfg.diarization.hf_token`. Catches the regression where someone refactors
    the gate back to a zero-arg `get_hf_token()` call.
    """

    @staticmethod
    def _setup(tmp_path, monkeypatch, *, hf_token: str):
        from typer.testing import CliRunner

        from trans import cli as cli_mod
        from trans import diarizer as diar_mod
        from trans.config import Config, DiarizationConfig

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr(diar_mod, "_read_hf_cache_token", lambda: None)

        monkeypatch.setattr(cli_mod, "HAS_PYANNOTE", True)
        monkeypatch.setattr(cli_mod, "HAS_FASTER_WHISPER", True)
        monkeypatch.setattr(
            cli_mod,
            "load_config",
            lambda: Config(diarization=DiarizationConfig(hf_token=hf_token)),
        )
        return CliRunner(), cli_mod

    def test_gate_does_not_fire_when_config_token_set(self, tmp_path, monkeypatch):
        runner, cli_mod = self._setup(tmp_path, monkeypatch, hf_token="hf_cfg")
        result = runner.invoke(
            cli_mod.app,
            [
                "transcribe",
                "--diarize",
                "--output-dir",
                str(tmp_path),
                "--quiet",
                "https://www.youtube.com/watch?v=fakefakefake",
            ],
        )
        assert "Speaker diarization requires a HuggingFace token" not in result.output, (
            f"gate fired despite cfg.diarization.hf_token being set:\n{result.output}"
        )

    def test_gate_fires_when_no_token_anywhere(self, tmp_path, monkeypatch):
        """Positive control: empty config + cleared envs + no cache → gate must fire."""
        runner, cli_mod = self._setup(tmp_path, monkeypatch, hf_token="")
        result = runner.invoke(
            cli_mod.app,
            [
                "transcribe",
                "--diarize",
                "--output-dir",
                str(tmp_path),
                "--quiet",
                "https://www.youtube.com/watch?v=fakefakefake",
            ],
        )
        assert result.exit_code == 1
        assert "Speaker diarization requires a HuggingFace token" in result.output


class TestTranscribeNetworkSmoke:
    """Opt-in smoke tests that hit real services.

    Skipped by default. Run with: `pytest --run-network test_trans.py`.

    Per-function tests bypass `app()` and mock at the yt-dlp / faster-whisper
    seams, so a regression in Typer wiring (option resolution, exit codes,
    argument parsing) inside the `transcribe` entry point can slip through
    unnoticed. These two tests drive the real stack end-to-end.
    """

    # Rick Astley "Never Gonna Give You Up" — public on YouTube since 2009
    # with native English captions, also the URL the offline `get_video_id`
    # tests pin to. `extract_native_captions` accepts both author-uploaded
    # and auto-generated subtitle tracks, so either source satisfies the
    # test. If the video ever dies, swap for any other long-lived clip
    # with native captions — assertions are lenient (exit code +
    # native-captions branch fired + non-empty .txt).
    _STABLE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    @staticmethod
    def _isolate_user_state(cli_mod, monkeypatch, tmp_path):
        """Point load_config and CacheManager at hermetic fakes so the smoke
        tests can't read or write the user's real config / cache trees."""
        from trans.cache import CacheManager
        from trans.config import Config

        monkeypatch.setattr(cli_mod, "load_config", lambda: Config())
        monkeypatch.setattr(
            cli_mod,
            "CacheManager",
            lambda: CacheManager(db_path=tmp_path / "c.db"),
        )

    @pytest.mark.network
    def test_transcribe_youtube_url_native_captions(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from trans import cli as cli_mod

        self._isolate_user_state(cli_mod, monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli_mod.app,
            ["transcribe", self._STABLE_URL, "--no-cache", "-f", "txt"],
        )
        # Lenient: exit 0 + a non-empty .txt somewhere under tmp_path.
        assert result.exit_code == 0, (
            f"transcribe failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        # Pin the *native-captions* branch — without this, a future URL swap
        # to a no-captions clip would silently fall through to Whisper and
        # the test would still pass, contradicting its own name.
        assert "native captions" in result.stdout, (
            f"native-captions branch did not fire: stdout={result.stdout!r}"
        )
        txts = list(tmp_path.glob("*.txt"))
        assert txts, f"no .txt produced; cwd={list(tmp_path.iterdir())}"
        assert txts[0].stat().st_size > 0, "txt is empty"

    @pytest.mark.network
    def test_transcribe_local_audio_file(self, tmp_path, monkeypatch):
        """Exercise the local-file branch end-to-end: ffmpeg-generated silence
        in, faster-whisper out. Silent audio is fine — we only assert the
        wiring works, not that any words came out.
        """
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")

        from typer.testing import CliRunner

        from trans import cli as cli_mod

        self._isolate_user_state(cli_mod, monkeypatch, tmp_path)

        wav = tmp_path / "silence.wav"
        # 1s of mono 16kHz silence. Cheap; Whisper-tiny still loads but
        # transcribes to empty or near-empty text.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=mono:sample_rate=16000",
                "-t",
                "1",
                str(wav),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert wav.exists() and wav.stat().st_size > 0

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        # `--no-cache` is a no-op for local files (CacheManager is URL-only),
        # omitted here for clarity.
        result = runner.invoke(
            cli_mod.app,
            ["transcribe", str(wav), "-m", "tiny", "-f", "txt"],
        )
        assert result.exit_code == 0, (
            f"transcribe failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        out = tmp_path / "silence.txt"
        assert out.exists(), f"expected {out}, found {list(tmp_path.iterdir())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
