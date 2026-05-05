//! Pure utility functions and constants for trans.

use md5::{Digest, Md5};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

pub const WHISPER_MODELS: &[&str] = &["tiny", "base", "small", "medium", "large"];
pub const OUTPUT_FORMATS: &[&str] = &["txt", "srt", "vtt", "json", "all"];

pub static AUDIO_EXTENSIONS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"]
        .into_iter()
        .collect()
});

pub static VIDEO_EXTENSIONS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg",
    ]
    .into_iter()
    .collect()
});

pub static MEDIA_EXTENSIONS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    AUDIO_EXTENSIONS
        .iter()
        .chain(VIDEO_EXTENSIONS.iter())
        .copied()
        .collect()
});

static RE_YT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})").unwrap());
static RE_TT: Lazy<Regex> = Lazy::new(|| Regex::new(r"/video/(\d+)").unwrap());
static RE_TW_VOD: Lazy<Regex> = Lazy::new(|| Regex::new(r"/videos/(\d+)").unwrap());
static RE_TW_CLIP: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"/clip/([a-zA-Z0-9_-]+)").unwrap());
static RE_TW_CLIPS_DOMAIN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"clips\.twitch\.tv/([a-zA-Z0-9_-]+)").unwrap());
static RE_NON_WORD: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\w\s-]").unwrap());
static RE_SPACES: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static RE_UNDERSCORES: Lazy<Regex> = Lazy::new(|| Regex::new(r"_+").unwrap());
static RE_SPEAKER_NUM: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)").unwrap());

/// Extract a unique video ID from URL.
pub fn get_video_id(url: &str) -> String {
    if url.contains("youtube.com") || url.contains("youtu.be") {
        if let Some(c) = RE_YT.captures(url) {
            return format!("yt_{}", &c[1]);
        }
    }
    if url.contains("tiktok.com") {
        if let Some(c) = RE_TT.captures(url) {
            return format!("tt_{}", &c[1]);
        }
    }
    if url.contains("twitch.tv") {
        if let Some(c) = RE_TW_VOD.captures(url) {
            return format!("tw_{}", &c[1]);
        }
        if let Some(c) = RE_TW_CLIP
            .captures(url)
            .or_else(|| RE_TW_CLIPS_DOMAIN.captures(url))
        {
            return format!("twclip_{}", &c[1]);
        }
    }
    let mut hasher = Md5::new();
    hasher.update(url.as_bytes());
    let digest = hasher.finalize();
    let hex_str = hex::encode(digest);
    format!("hash_{}", &hex_str[..12])
}

/// Create a safe filename from video title.
pub fn sanitize_filename(title: &str, max_length: usize) -> String {
    let s = RE_NON_WORD.replace_all(title, "");
    let s = RE_SPACES.replace_all(&s, "_");
    let s = RE_UNDERSCORES.replace_all(&s, "_");
    let trimmed = s.trim_matches('_').to_string();
    let mut chars: Vec<char> = trimmed.chars().collect();
    if chars.len() > max_length {
        chars.truncate(max_length);
    }
    chars.into_iter().collect()
}

pub fn is_tiktok_url(url: &str) -> bool {
    url.contains("tiktok.com") || url.contains("vm.tiktok.com")
}

pub fn is_twitch_url(url: &str) -> bool {
    url.contains("twitch.tv")
}

pub fn is_local_file(path: &str) -> bool {
    if path.starts_with("http://") || path.starts_with("https://") {
        return false;
    }
    for d in ["youtube.com", "youtu.be", "tiktok.com", "twitch.tv"] {
        if path.contains(d) {
            return false;
        }
    }
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_lowercase()));
    match ext {
        Some(e) => MEDIA_EXTENSIONS.contains(e.as_str()),
        None => false,
    }
}

pub fn is_audio_file(path: &str) -> bool {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_lowercase()));
    match ext {
        Some(e) => AUDIO_EXTENSIONS.contains(e.as_str()),
        None => false,
    }
}

pub fn format_timestamp_srt(seconds: f64) -> String {
    let total = seconds.max(0.0);
    let hours = (total / 3600.0).floor() as u64;
    let minutes = ((total % 3600.0) / 60.0).floor() as u64;
    let secs = (total % 60.0).floor() as u64;
    let millis = ((total - total.floor()) * 1000.0).floor() as u64;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

pub fn format_timestamp_vtt(seconds: f64) -> String {
    let total = seconds.max(0.0);
    let hours = (total / 3600.0).floor() as u64;
    let minutes = ((total % 3600.0) / 60.0).floor() as u64;
    let secs = (total % 60.0).floor() as u64;
    let millis = ((total - total.floor()) * 1000.0).floor() as u64;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
}

pub fn format_speaker_label(speaker_id: &str) -> String {
    if speaker_id == "UNKNOWN" {
        return "Unknown".to_string();
    }
    if let Some(c) = RE_SPEAKER_NUM.captures(speaker_id) {
        if let Ok(n) = c[1].parse::<u64>() {
            return format!("Speaker {}", n + 1);
        }
    }
    speaker_id.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DiarizationSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

/// Merge transcript segments with speaker labels from diarization.
/// Each transcript segment gets the speaker with the most overlap.
pub fn assign_speakers_to_segments(
    transcript_segments: &mut [TranscriptSegment],
    diarization_segments: &[DiarizationSegment],
) {
    for t_seg in transcript_segments.iter_mut() {
        let (t_start, t_end) = (t_seg.start, t_seg.end);
        let mut overlaps: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for d_seg in diarization_segments {
            let overlap_start = t_start.max(d_seg.start);
            let overlap_end = t_end.min(d_seg.end);
            let overlap = (overlap_end - overlap_start).max(0.0);
            if overlap > 0.0 {
                *overlaps.entry(d_seg.speaker.clone()).or_insert(0.0) += overlap;
            }
        }
        if let Some((sp, _)) =
            overlaps.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            t_seg.speaker = Some(sp);
        } else {
            t_seg.speaker = Some("UNKNOWN".to_string());
        }
    }
}
