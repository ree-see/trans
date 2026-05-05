//! Speaker diarization via pyannote-audio (shells out to a Python helper).
//!
//! Mirrors `diarizer.py`. Since pyannote-audio has no Rust port, we invoke
//! a small Python script that runs the pipeline and emits JSON. Detection
//! of pyannote uses `python -c "import pyannote.audio"`.

use crate::utils::DiarizationSegment;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Stdio};

pub fn has_pyannote() -> bool {
    Command::new("python3")
        .args(["-c", "import pyannote.audio"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn get_hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.is_empty() {
            return Some(t);
        }
    }
    if let Ok(t) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        if !t.is_empty() {
            return Some(t);
        }
    }
    if let Some(home) = dirs::home_dir() {
        let token_path: PathBuf = home.join(".cache").join("huggingface").join("token");
        if token_path.exists() {
            if let Ok(s) = std::fs::read_to_string(&token_path) {
                let t = s.trim().to_string();
                if !t.is_empty() {
                    return Some(t);
                }
            }
        }
    }
    None
}

/// Run speaker diarization. Returns a list of {start, end, speaker} segments.
pub fn run_diarization(
    audio_file: &str,
    hf_token: &str,
    num_speakers: Option<u32>,
    quiet: bool,
) -> Result<Vec<DiarizationSegment>> {
    if !has_pyannote() {
        return Err(anyhow!(
            "pyannote-audio not installed. Run: pip install pyannote-audio"
        ));
    }
    if hf_token.is_empty() {
        return Err(anyhow!(
            "HuggingFace token required for speaker diarization.\n\
             1. Create a token at https://huggingface.co/settings/tokens\n\
             2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1\n\
             3. Set HF_TOKEN environment variable or run: huggingface-cli login"
        ));
    }

    if !quiet {
        println!("  Loading diarization model...");
    }

    let script = r#"
import json
import sys
from pyannote.audio import Pipeline

audio_file = sys.argv[1]
hf_token = sys.argv[2]
num_speakers = int(sys.argv[3]) if sys.argv[3] != "0" else None

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token,
)
kwargs = {}
if num_speakers:
    kwargs["num_speakers"] = num_speakers
diarization = pipeline(audio_file, **kwargs)
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
sys.stdout.write(json.dumps(segments))
"#;

    if !quiet {
        println!("  Running speaker diarization...");
    }

    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(audio_file)
        .arg(hf_token)
        .arg(num_speakers.map(|n| n.to_string()).unwrap_or_else(|| "0".to_string()))
        .stdout(Stdio::piped())
        .stderr(if quiet { Stdio::null() } else { Stdio::inherit() })
        .output()
        .map_err(|e| anyhow!("failed to invoke python3 for diarization: {}", e))?;

    if !out.status.success() {
        return Err(anyhow!(
            "diarization helper exited with status {}",
            out.status.code().unwrap_or(-1)
        ));
    }

    let parsed: Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| anyhow!("could not parse diarization JSON: {}", e))?;
    let mut segments: Vec<DiarizationSegment> = Vec::new();
    if let Some(arr) = parsed.as_array() {
        for s in arr {
            let start = s.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let end = s.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let speaker = s
                .get("speaker")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN")
                .to_string();
            segments.push(DiarizationSegment { start, end, speaker });
        }
    }

    if !quiet {
        let unique: std::collections::HashSet<&str> =
            segments.iter().map(|s| s.speaker.as_str()).collect();
        println!("  Detected {} speaker(s)", unique.len());
    }

    Ok(segments)
}
