#!/usr/bin/env python3
"""Unit tests for trans_cli.py"""

import pytest
import tempfile
import os
from pathlib import Path

# Import functions from trans_cli
from trans_cli import (
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
        expected = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"}
        assert VIDEO_EXTENSIONS == expected

    def test_media_extensions_is_union(self):
        assert MEDIA_EXTENSIONS == AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

    def test_no_overlap_audio_video(self):
        # Audio and video extensions should be disjoint
        assert AUDIO_EXTENSIONS.isdisjoint(VIDEO_EXTENSIONS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
