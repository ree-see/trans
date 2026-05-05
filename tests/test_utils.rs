//! Port of test_trans.py — exercises the pure utility functions.

use trans::utils::{
    assign_speakers_to_segments, format_speaker_label, format_timestamp_srt, format_timestamp_vtt,
    get_video_id, is_audio_file, is_local_file, is_tiktok_url, is_twitch_url, sanitize_filename,
    DiarizationSegment, TranscriptSegment, AUDIO_EXTENSIONS, MEDIA_EXTENSIONS, VIDEO_EXTENSIONS,
};

// ---------------------------------------------------------------------------
// TestGetVideoId
// ---------------------------------------------------------------------------

#[test]
fn youtube_standard_url() {
    assert_eq!(
        get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
        "yt_dQw4w9WgXcQ"
    );
}

#[test]
fn youtube_short_url() {
    assert_eq!(
        get_video_id("https://youtu.be/dQw4w9WgXcQ"),
        "yt_dQw4w9WgXcQ"
    );
}

#[test]
fn youtube_with_extra_params() {
    assert_eq!(
        get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42"),
        "yt_dQw4w9WgXcQ"
    );
}

#[test]
fn youtube_mobile_url() {
    assert_eq!(
        get_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ"),
        "yt_dQw4w9WgXcQ"
    );
}

#[test]
fn tiktok_video_url() {
    assert_eq!(
        get_video_id("https://www.tiktok.com/@user/video/1234567890123456789"),
        "tt_1234567890123456789"
    );
}

#[test]
fn twitch_vod_url() {
    assert_eq!(
        get_video_id("https://www.twitch.tv/videos/1234567890"),
        "tw_1234567890"
    );
}

#[test]
fn twitch_clip_url() {
    assert_eq!(
        get_video_id("https://www.twitch.tv/channel/clip/AmazingClipName123"),
        "twclip_AmazingClipName123"
    );
}

#[test]
fn twitch_clips_domain() {
    assert_eq!(
        get_video_id("https://clips.twitch.tv/AmazingClipName123"),
        "twclip_AmazingClipName123"
    );
}

#[test]
fn unknown_url_uses_hash() {
    let id = get_video_id("https://example.com/some/video/path");
    assert!(id.starts_with("hash_"));
    assert_eq!(id.len(), 17); // "hash_" + 12 char hash
}

#[test]
fn same_url_same_hash() {
    let url = "https://example.com/video123";
    assert_eq!(get_video_id(url), get_video_id(url));
}

// ---------------------------------------------------------------------------
// TestSanitizeFilename
// ---------------------------------------------------------------------------

#[test]
fn basic_title() {
    assert_eq!(sanitize_filename("Hello World", 50), "Hello_World");
}

#[test]
fn special_characters() {
    assert_eq!(
        sanitize_filename("Video: The Best! @2024", 50),
        "Video_The_Best_2024"
    );
}

#[test]
fn multiple_spaces() {
    assert_eq!(sanitize_filename("Hello   World", 50), "Hello_World");
}

#[test]
fn multiple_underscores() {
    assert_eq!(sanitize_filename("Hello___World", 50), "Hello_World");
}

#[test]
fn max_length_50() {
    let long = "A".repeat(100);
    let res = sanitize_filename(&long, 50);
    assert_eq!(res.chars().count(), 50);
}

#[test]
fn custom_max_length() {
    let res = sanitize_filename("Hello World This Is A Long Title", 10);
    assert_eq!(res.chars().count(), 10);
}

#[test]
fn strips_leading_trailing_underscores() {
    assert_eq!(sanitize_filename("  Hello World  ", 50), "Hello_World");
    assert_eq!(sanitize_filename("_Hello_", 50), "Hello");
}

#[test]
fn unicode_preserved() {
    let res = sanitize_filename("Café Video", 50);
    assert!(res.contains("Caf"));
}

// ---------------------------------------------------------------------------
// TestUrlDetection
// ---------------------------------------------------------------------------

#[test]
fn is_tiktok_url_standard() {
    assert!(is_tiktok_url("https://www.tiktok.com/@user/video/123"));
    assert!(is_tiktok_url("https://tiktok.com/video"));
}

#[test]
fn is_tiktok_url_vm_shortlink() {
    assert!(is_tiktok_url("https://vm.tiktok.com/abc123"));
}

#[test]
fn is_tiktok_url_false() {
    assert!(!is_tiktok_url("https://youtube.com/watch?v=123"));
    assert!(!is_tiktok_url("https://example.com"));
}

#[test]
fn is_twitch_url_standard() {
    assert!(is_twitch_url("https://www.twitch.tv/videos/123"));
    assert!(is_twitch_url("https://twitch.tv/channel"));
}

#[test]
fn is_twitch_url_false() {
    assert!(!is_twitch_url("https://youtube.com/watch?v=123"));
    assert!(!is_twitch_url("https://tiktok.com/video"));
}

// ---------------------------------------------------------------------------
// TestLocalFileDetection
// ---------------------------------------------------------------------------

#[test]
fn is_local_file_audio() {
    assert!(is_local_file("podcast.mp3"));
    assert!(is_local_file("/path/to/audio.wav"));
    assert!(is_local_file("./music.flac"));
}

#[test]
fn is_local_file_video() {
    assert!(is_local_file("video.mp4"));
    assert!(is_local_file("/home/user/movie.mkv"));
    assert!(is_local_file("lecture.webm"));
}

#[test]
fn is_local_file_url_false() {
    assert!(!is_local_file("https://youtube.com/watch?v=123"));
    assert!(!is_local_file("http://example.com/video.mp4"));
}

#[test]
fn is_local_file_known_domains_false() {
    assert!(!is_local_file("youtube.com/video"));
    assert!(!is_local_file("tiktok.com/@user/video/123"));
}

#[test]
fn is_local_file_unknown_extension_false() {
    assert!(!is_local_file("document.pdf"));
    assert!(!is_local_file("image.png"));
}

#[test]
fn is_audio_file_true() {
    assert!(is_audio_file("song.mp3"));
    assert!(is_audio_file("recording.wav"));
    assert!(is_audio_file("podcast.m4a"));
    assert!(is_audio_file("music.flac"));
    assert!(is_audio_file("audio.ogg"));
    assert!(is_audio_file("voice.opus"));
}

#[test]
fn is_audio_file_video_false() {
    assert!(!is_audio_file("video.mp4"));
    assert!(!is_audio_file("movie.mkv"));
    assert!(!is_audio_file("clip.webm"));
}

// ---------------------------------------------------------------------------
// TestTimestampFormatting
// ---------------------------------------------------------------------------

#[test]
fn format_timestamp_srt_zero() {
    assert_eq!(format_timestamp_srt(0.0), "00:00:00,000");
}

#[test]
fn format_timestamp_srt_seconds() {
    assert_eq!(format_timestamp_srt(5.0), "00:00:05,000");
}

#[test]
fn format_timestamp_srt_minutes() {
    assert_eq!(format_timestamp_srt(125.0), "00:02:05,000");
}

#[test]
fn format_timestamp_srt_hours() {
    assert_eq!(format_timestamp_srt(3661.0), "01:01:01,000");
}

#[test]
fn format_timestamp_srt_milliseconds() {
    assert_eq!(format_timestamp_srt(1.5), "00:00:01,500");
    assert_eq!(format_timestamp_srt(1.234), "00:00:01,234");
}

#[test]
fn format_timestamp_vtt_zero() {
    assert_eq!(format_timestamp_vtt(0.0), "00:00:00.000");
}

#[test]
fn format_timestamp_vtt_seconds() {
    assert_eq!(format_timestamp_vtt(5.0), "00:00:05.000");
}

#[test]
fn format_timestamp_vtt_minutes() {
    assert_eq!(format_timestamp_vtt(125.0), "00:02:05.000");
}

#[test]
fn format_timestamp_vtt_hours() {
    assert_eq!(format_timestamp_vtt(3661.0), "01:01:01.000");
}

#[test]
fn format_timestamp_vtt_milliseconds() {
    assert_eq!(format_timestamp_vtt(1.5), "00:00:01.500");
    assert_eq!(format_timestamp_vtt(1.234), "00:00:01.234");
}

#[test]
fn srt_vs_vtt_delimiter() {
    let srt = format_timestamp_srt(1.5);
    let vtt = format_timestamp_vtt(1.5);
    assert!(srt.contains(','));
    assert!(!srt.split(',').nth(1).unwrap().contains('.'));
    assert!(vtt.contains('.'));
}

// ---------------------------------------------------------------------------
// TestSpeakerLabels
// ---------------------------------------------------------------------------

#[test]
fn format_speaker_label_standard() {
    assert_eq!(format_speaker_label("SPEAKER_00"), "Speaker 1");
    assert_eq!(format_speaker_label("SPEAKER_01"), "Speaker 2");
    assert_eq!(format_speaker_label("SPEAKER_09"), "Speaker 10");
}

#[test]
fn format_speaker_label_unknown() {
    assert_eq!(format_speaker_label("UNKNOWN"), "Unknown");
}

#[test]
fn format_speaker_label_passthrough() {
    assert_eq!(format_speaker_label("CustomSpeaker"), "CustomSpeaker");
}

// ---------------------------------------------------------------------------
// TestSpeakerAssignment
// ---------------------------------------------------------------------------

fn t(start: f64, end: f64, text: &str) -> TranscriptSegment {
    TranscriptSegment {
        start,
        end,
        text: text.to_string(),
        speaker: None,
    }
}

fn d(start: f64, end: f64, speaker: &str) -> DiarizationSegment {
    DiarizationSegment {
        start,
        end,
        speaker: speaker.to_string(),
    }
}

#[test]
fn simple_assignment() {
    let mut transcript = vec![t(0.0, 5.0, "Hello"), t(5.0, 10.0, "World")];
    let diarization = vec![d(0.0, 5.0, "SPEAKER_00"), d(5.0, 10.0, "SPEAKER_01")];
    assign_speakers_to_segments(&mut transcript, &diarization);
    assert_eq!(transcript[0].speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(transcript[1].speaker.as_deref(), Some("SPEAKER_01"));
}

#[test]
fn overlapping_speakers_majority_wins() {
    let mut transcript = vec![t(0.0, 10.0, "Long segment")];
    let diarization = vec![d(0.0, 3.0, "SPEAKER_00"), d(3.0, 10.0, "SPEAKER_01")];
    assign_speakers_to_segments(&mut transcript, &diarization);
    assert_eq!(transcript[0].speaker.as_deref(), Some("SPEAKER_01"));
}

#[test]
fn no_overlap_assigns_unknown() {
    let mut transcript = vec![t(0.0, 5.0, "Gap segment")];
    let diarization = vec![d(10.0, 15.0, "SPEAKER_00")];
    assign_speakers_to_segments(&mut transcript, &diarization);
    assert_eq!(transcript[0].speaker.as_deref(), Some("UNKNOWN"));
}

#[test]
fn empty_diarization() {
    let mut transcript = vec![t(0.0, 5.0, "No speakers")];
    assign_speakers_to_segments(&mut transcript, &[]);
    assert_eq!(transcript[0].speaker.as_deref(), Some("UNKNOWN"));
}

// ---------------------------------------------------------------------------
// TestExtensionSets
// ---------------------------------------------------------------------------

#[test]
fn audio_extensions_complete() {
    let expected: std::collections::HashSet<&str> = [
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ]
    .into_iter()
    .collect();
    let actual: std::collections::HashSet<&str> = AUDIO_EXTENSIONS.iter().copied().collect();
    assert_eq!(actual, expected);
}

#[test]
fn video_extensions_complete() {
    let expected: std::collections::HashSet<&str> = [
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg",
    ]
    .into_iter()
    .collect();
    let actual: std::collections::HashSet<&str> = VIDEO_EXTENSIONS.iter().copied().collect();
    assert_eq!(actual, expected);
}

#[test]
fn media_extensions_is_union() {
    let union: std::collections::HashSet<&str> = AUDIO_EXTENSIONS
        .iter()
        .chain(VIDEO_EXTENSIONS.iter())
        .copied()
        .collect();
    let actual: std::collections::HashSet<&str> = MEDIA_EXTENSIONS.iter().copied().collect();
    assert_eq!(actual, union);
}

#[test]
fn no_overlap_audio_video() {
    let intersect: std::collections::HashSet<&str> = AUDIO_EXTENSIONS
        .iter()
        .filter(|e| VIDEO_EXTENSIONS.contains(*e))
        .copied()
        .collect();
    assert!(intersect.is_empty());
}
