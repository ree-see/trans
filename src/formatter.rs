//! Output file writing for transcript segments.

use crate::utils::{
    format_speaker_label, format_timestamp_srt, format_timestamp_vtt, TranscriptSegment,
};
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct TranscriptionInfo {
    pub language: String,
    pub language_probability: f64,
    pub duration: f64,
}

/// Write transcript segments to one or more output files.
pub fn write_output(
    segments: &[TranscriptSegment],
    output_base: &str,
    fmt: &str,
    info: Option<&TranscriptionInfo>,
    diarized: bool,
) -> Result<Vec<PathBuf>> {
    let has_speakers =
        diarized && !segments.is_empty() && segments[0].speaker.is_some();
    let mut created: Vec<PathBuf> = Vec::new();

    if fmt == "txt" || fmt == "all" {
        let path = PathBuf::from(format!("{}.txt", output_base));
        let mut out = String::new();
        if has_speakers {
            let mut current: Option<String> = None;
            for seg in segments {
                let raw = seg.speaker.as_deref().unwrap_or("UNKNOWN");
                let speaker = format_speaker_label(raw);
                if Some(&speaker) != current.as_ref() {
                    if current.is_some() {
                        out.push('\n');
                    }
                    out.push_str(&format!("[{}]\n", speaker));
                    current = Some(speaker);
                }
                out.push_str(&seg.text);
                out.push('\n');
            }
        } else {
            for seg in segments {
                out.push_str(&seg.text);
                out.push('\n');
            }
        }
        std::fs::write(&path, out)?;
        created.push(path);
    }

    if fmt == "srt" || fmt == "all" {
        let path = PathBuf::from(format!("{}.srt", output_base));
        let mut out = String::new();
        for (i, seg) in segments.iter().enumerate() {
            let start = format_timestamp_srt(seg.start);
            let end = format_timestamp_srt(seg.end);
            let mut text = seg.text.clone();
            if has_speakers {
                let raw = seg.speaker.as_deref().unwrap_or("UNKNOWN");
                let sp = format_speaker_label(raw);
                text = format!("[{}] {}", sp, text);
            }
            out.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, start, end, text));
        }
        std::fs::write(&path, out)?;
        created.push(path);
    }

    if fmt == "vtt" || fmt == "all" {
        let path = PathBuf::from(format!("{}.vtt", output_base));
        let mut out = String::from("WEBVTT\n\n");
        for seg in segments {
            let start = format_timestamp_vtt(seg.start);
            let end = format_timestamp_vtt(seg.end);
            if has_speakers {
                let raw = seg.speaker.as_deref().unwrap_or("UNKNOWN");
                let sp = format_speaker_label(raw);
                out.push_str(&format!("{} --> {}\n<v {}>{}\n\n", start, end, sp, seg.text));
            } else {
                out.push_str(&format!("{} --> {}\n{}\n\n", start, end, seg.text));
            }
        }
        std::fs::write(&path, out)?;
        created.push(path);
    }

    if fmt == "json" || fmt == "all" {
        let path = PathBuf::from(format!("{}.json", output_base));
        let mut data: Map<String, Value> = Map::new();
        data.insert("diarization".to_string(), json!(has_speakers));
        let segs_json: Vec<Value> = segments
            .iter()
            .map(|s| {
                let mut m = Map::new();
                m.insert("start".to_string(), json!(s.start));
                m.insert("end".to_string(), json!(s.end));
                m.insert("text".to_string(), json!(s.text));
                if let Some(sp) = &s.speaker {
                    m.insert("speaker".to_string(), json!(sp));
                }
                Value::Object(m)
            })
            .collect();
        data.insert("segments".to_string(), Value::Array(segs_json));
        if let Some(info) = info {
            data.insert("language".to_string(), json!(info.language));
            data.insert(
                "language_probability".to_string(),
                json!(info.language_probability),
            );
            data.insert("duration".to_string(), json!(info.duration));
        }
        if has_speakers {
            let unique: BTreeSet<String> = segments
                .iter()
                .filter_map(|s| s.speaker.clone())
                .collect();
            let labels: Vec<Value> = unique
                .into_iter()
                .map(|s| json!(format_speaker_label(&s)))
                .collect();
            data.insert("speakers".to_string(), Value::Array(labels));
        }
        let json_str = serde_json::to_string_pretty(&Value::Object(data))?;
        std::fs::write(&path, json_str)?;
        created.push(path);
    }

    Ok(created)
}
