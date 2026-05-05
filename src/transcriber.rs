//! Whisper transcription engine — shells out to a Whisper CLI.
//!
//! Mirrors the Python `transcriber.py` which uses faster-whisper.
//! We invoke either `whisper-ctranslate2` (a direct CLI wrapper around
//! faster-whisper, preferred when available) or fall back to OpenAI's
//! `whisper` CLI. Both accept the same core flags and emit JSON output
//! with `segments[].start/end/text` and `language` fields.

use crate::formatter::TranscriptionInfo;
use crate::utils::TranscriptSegment;
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Returns duration in seconds via ffprobe, or 0 on failure.
pub fn get_file_duration(audio_file: &str) -> f64 {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_file,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();

    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            s.parse::<f64>().unwrap_or(0.0)
        }
        Err(_) => 0.0,
    }
}

/// Extract audio track from a video file using ffmpeg.
pub fn extract_audio_from_video(video_path: &str, output_audio: &str, quiet: bool) -> bool {
    if !quiet {
        println!("→ Extracting audio from video...");
    }
    let result = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            output_audio,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output();

    match result {
        Ok(o) if o.status.success() => true,
        Ok(o) => {
            if !quiet {
                eprintln!(
                    "✗ Failed to extract audio: {}",
                    String::from_utf8_lossy(&o.stderr)
                );
            }
            false
        }
        Err(e) => {
            if !quiet {
                eprintln!("✗ Failed to invoke ffmpeg: {}", e);
            }
            false
        }
    }
}

fn whisper_binary() -> &'static str {
    if which_in_path("whisper-ctranslate2") {
        "whisper-ctranslate2"
    } else {
        "whisper"
    }
}

fn which_in_path(name: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return true;
            }
        }
    }
    false
}

pub struct TranscriptionEngine {
    pub model_name: String,
}

impl TranscriptionEngine {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }

    /// Transcribe audio file. Returns (segments, info).
    pub fn transcribe(
        &self,
        audio_file: &str,
        language: Option<&str>,
        quiet: bool,
        translate: bool,
    ) -> Result<(Vec<TranscriptSegment>, TranscriptionInfo)> {
        if !quiet {
            println!("  Loading {} model...", self.model_name);
        }

        let total_duration = get_file_duration(audio_file);

        if !quiet {
            println!("  Transcribing...");
        }

        let bin = whisper_binary();
        let tmp = tempdir_in_cwd("trans-whisper")?;

        let task = if translate { "translate" } else { "transcribe" };
        let mut args: Vec<String> = vec![
            audio_file.to_string(),
            "--model".to_string(),
            self.model_name.clone(),
            "--task".to_string(),
            task.to_string(),
            "--beam_size".to_string(),
            "5".to_string(),
            "--output_format".to_string(),
            "json".to_string(),
            "--output_dir".to_string(),
            tmp.to_string_lossy().to_string(),
        ];
        if let Some(lang) = language.filter(|s| !s.is_empty()) {
            args.push("--language".to_string());
            args.push(lang.to_string());
        }
        if quiet {
            args.push("--verbose".to_string());
            args.push("False".to_string());
        }

        let out = Command::new(bin)
            .args(&args)
            .stdout(if quiet { Stdio::null() } else { Stdio::inherit() })
            .stderr(if quiet { Stdio::null() } else { Stdio::inherit() })
            .output()
            .map_err(|e| anyhow!("failed to run {}: {} (is whisper installed?)", bin, e))?;

        if !out.status.success() {
            return Err(anyhow!(
                "{} exited with status {}",
                bin,
                out.status.code().unwrap_or(-1)
            ));
        }

        // Find the produced JSON in tmp dir.
        let json_path = locate_json_in(&tmp, audio_file)?;
        let raw = std::fs::read_to_string(&json_path)
            .with_context(|| format!("reading whisper JSON {:?}", json_path))?;
        let v: Value = serde_json::from_str(&raw)
            .with_context(|| format!("parsing whisper JSON {:?}", json_path))?;

        let language_str = v
            .get("language")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let language_probability = v
            .get("language_probability")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);

        let mut segments_list: Vec<TranscriptSegment> = Vec::new();
        let mut last_percent: i64 = 0;
        if let Some(arr) = v.get("segments").and_then(|s| s.as_array()) {
            for seg in arr {
                let start = seg.get("start").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let end = seg.get("end").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let text = seg
                    .get("text")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                segments_list.push(TranscriptSegment {
                    start,
                    end,
                    text,
                    speaker: None,
                });
                if !quiet && total_duration > 0.0 {
                    let percent = ((end / total_duration) * 100.0).min(100.0) as i64;
                    if percent != last_percent {
                        let _ = std::io::stdout().write_all(
                            format!("\r  Progress: {}%", percent).as_bytes(),
                        );
                        let _ = std::io::stdout().flush();
                        last_percent = percent;
                    }
                }
            }
        }

        if !quiet && total_duration > 0.0 {
            let _ = std::io::stdout().write_all(b"\r  Progress: 100%\n");
            let _ = std::io::stdout().flush();
        }

        if !quiet && segments_list.is_empty() {
            println!("  Warning: No speech detected in audio");
        }

        // Clean up temp dir.
        let _ = std::fs::remove_dir_all(&tmp);

        let info = TranscriptionInfo {
            language: language_str,
            language_probability,
            duration: total_duration,
        };
        Ok((segments_list, info))
    }
}

fn locate_json_in(dir: &Path, audio_file: &str) -> Result<PathBuf> {
    // whisper writes <stem>.json based on the input filename.
    let stem = Path::new(audio_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let preferred = dir.join(format!("{}.json", stem));
    if preferred.exists() {
        return Ok(preferred);
    }
    // Fallback: any .json in the dir.
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("json") {
            return Ok(p);
        }
    }
    Err(anyhow!("whisper produced no JSON output in {:?}", dir))
}

fn tempdir_in_cwd(prefix: &str) -> Result<PathBuf> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = base.join(format!("{}-{}-{}", prefix, pid, nanos));
    std::fs::create_dir_all(&dir).context("create whisper temp dir")?;
    Ok(dir)
}
