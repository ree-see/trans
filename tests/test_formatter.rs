//! Output formatter tests.

use serde_json::Value;
use tempfile::tempdir;
use trans::formatter::{write_output, TranscriptionInfo};
use trans::utils::TranscriptSegment;

fn seg(start: f64, end: f64, text: &str, speaker: Option<&str>) -> TranscriptSegment {
    TranscriptSegment {
        start,
        end,
        text: text.to_string(),
        speaker: speaker.map(String::from),
    }
}

#[test]
fn writes_txt() {
    let tmp = tempdir().unwrap();
    let base = tmp.path().join("out").to_string_lossy().to_string();
    let segs = vec![seg(0.0, 1.0, "hello", None), seg(1.0, 2.0, "world", None)];
    let created = write_output(&segs, &base, "txt", None, false).unwrap();
    assert_eq!(created.len(), 1);
    let body = std::fs::read_to_string(&created[0]).unwrap();
    assert_eq!(body, "hello\nworld\n");
}

#[test]
fn writes_srt_with_timestamps() {
    let tmp = tempdir().unwrap();
    let base = tmp.path().join("out").to_string_lossy().to_string();
    let segs = vec![seg(0.0, 1.5, "hi", None)];
    write_output(&segs, &base, "srt", None, false).unwrap();
    let body = std::fs::read_to_string(format!("{}.srt", base)).unwrap();
    assert!(body.contains("1\n00:00:00,000 --> 00:00:01,500\nhi\n\n"));
}

#[test]
fn writes_vtt_header() {
    let tmp = tempdir().unwrap();
    let base = tmp.path().join("out").to_string_lossy().to_string();
    let segs = vec![seg(0.0, 1.5, "hi", None)];
    write_output(&segs, &base, "vtt", None, false).unwrap();
    let body = std::fs::read_to_string(format!("{}.vtt", base)).unwrap();
    assert!(body.starts_with("WEBVTT\n\n"));
    assert!(body.contains("00:00:00.000 --> 00:00:01.500"));
}

#[test]
fn writes_json_with_info() {
    let tmp = tempdir().unwrap();
    let base = tmp.path().join("out").to_string_lossy().to_string();
    let segs = vec![seg(0.0, 1.0, "hello", None)];
    let info = TranscriptionInfo {
        language: "en".to_string(),
        language_probability: 0.99,
        duration: 1.0,
    };
    write_output(&segs, &base, "json", Some(&info), false).unwrap();
    let body = std::fs::read_to_string(format!("{}.json", base)).unwrap();
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["language"], "en");
    assert_eq!(v["segments"][0]["text"], "hello");
    assert_eq!(v["diarization"], false);
}

#[test]
fn writes_all_creates_four_files() {
    let tmp = tempdir().unwrap();
    let base = tmp.path().join("out").to_string_lossy().to_string();
    let segs = vec![seg(0.0, 1.0, "hi", None)];
    let created = write_output(&segs, &base, "all", None, false).unwrap();
    assert_eq!(created.len(), 4);
    for ext in ["txt", "srt", "vtt", "json"] {
        assert!(std::path::Path::new(&format!("{}.{}", base, ext)).exists());
    }
}

#[test]
fn diarized_txt_groups_by_speaker() {
    let tmp = tempdir().unwrap();
    let base = tmp.path().join("out").to_string_lossy().to_string();
    let segs = vec![
        seg(0.0, 1.0, "hello", Some("SPEAKER_00")),
        seg(1.0, 2.0, "world", Some("SPEAKER_00")),
        seg(2.0, 3.0, "again", Some("SPEAKER_01")),
    ];
    write_output(&segs, &base, "txt", None, true).unwrap();
    let body = std::fs::read_to_string(format!("{}.txt", base)).unwrap();
    assert!(body.contains("[Speaker 1]\nhello\nworld\n"));
    assert!(body.contains("[Speaker 2]\nagain\n"));
}
