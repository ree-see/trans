//! yt-dlp shell-out wrapper for downloading and caption extraction.

use crate::utils::is_tiktok_url;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Default)]
pub struct VideoInfo {
    pub title: String,
    pub duration: u64,
}

fn base_args(url: &str, cookies: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    if is_tiktok_url(url) {
        args.push("--impersonate".to_string());
        args.push("chrome-131".to_string());
    }
    if let Some(c) = cookies {
        args.push("--cookies".to_string());
        args.push(c.to_string());
    }
    args
}

/// Fetch video metadata without downloading (yt-dlp -j).
pub fn get_video_info(url: &str, cookies: Option<&str>, quiet: bool) -> Result<VideoInfo> {
    let mut args = base_args(url, cookies);
    args.push("-j".to_string());
    args.push("--no-warnings".to_string());
    args.push(url.to_string());

    let out = Command::new("yt-dlp")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| anyhow!("failed to run yt-dlp: {} (is yt-dlp installed?)", e))?;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).to_string();
        if is_tiktok_url(url)
            && (err.contains("IP address is blocked") || err.to_lowercase().contains("blocked"))
        {
            if !quiet {
                eprintln!("✗ TikTok is blocking this server's IP address.");
                eprintln!();
                eprintln!("Workarounds:");
                eprintln!(
                    "  1. Use --cookies to provide cookies from a logged-in browser session"
                );
                eprintln!("     Export cookies with a browser extension like 'Get cookies.txt'");
                eprintln!();
                eprintln!("  2. Run trans from a residential IP (not a datacenter/VPS)");
                eprintln!();
                eprintln!("  3. Use a VPN or proxy with a non-datacenter IP");
            }
            std::process::exit(1);
        }
        if !quiet {
            eprintln!("✗ Error fetching video info: {}", err);
        }
        std::process::exit(1);
    }

    let json: Value =
        serde_json::from_slice(&out.stdout).map_err(|e| anyhow!("yt-dlp JSON parse: {}", e))?;
    let title = json
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("video")
        .to_string();
    let duration = json
        .get("duration")
        .and_then(|v| v.as_f64())
        .map(|d| d as u64)
        .unwrap_or(0);
    Ok(VideoInfo { title, duration })
}

/// Download audio from URL, returns final file path.
pub fn download_audio(
    url: &str,
    output_path: &str,
    cookies: Option<&str>,
    quiet: bool,
) -> Result<String> {
    let mut args = base_args(url, cookies);
    args.extend([
        "-f".to_string(),
        "bestaudio/best".to_string(),
        "-x".to_string(),
        "--audio-format".to_string(),
        "mp3".to_string(),
        "-o".to_string(),
        output_path.to_string(),
    ]);
    if quiet {
        args.push("--quiet".to_string());
        args.push("--no-warnings".to_string());
    }
    args.push(url.to_string());

    let status = Command::new("yt-dlp")
        .args(&args)
        .status()
        .map_err(|e| anyhow!("failed to run yt-dlp: {}", e))?;
    if !status.success() {
        return Err(anyhow!("yt-dlp exited with non-zero status"));
    }

    // yt-dlp appends .mp3 when post-processing
    let mut final_path = output_path.to_string();
    if !final_path.ends_with(".mp3") {
        final_path = format!("{}.mp3", final_path);
    }
    if Path::new(&final_path).exists() {
        return Ok(final_path);
    }
    Ok(output_path.to_string())
}

/// Attempt to extract auto-generated captions. Returns true if a caption file was created.
pub fn extract_native_captions(
    url: &str,
    output_path: &str,
    output_format: &str,
    quiet: bool,
) -> bool {
    if !quiet {
        println!("→ Checking for native captions...");
    }

    let sub_format = match output_format {
        "vtt" | "all" => "vtt",
        "srt" => "srt",
        _ => "vtt",
    };

    let args = vec![
        "--write-auto-sub".to_string(),
        "--write-sub".to_string(),
        "--sub-langs".to_string(),
        "en".to_string(),
        "--skip-download".to_string(),
        "--sub-format".to_string(),
        sub_format.to_string(),
        "-o".to_string(),
        output_path.to_string(),
        "--quiet".to_string(),
        "--no-warnings".to_string(),
        url.to_string(),
    ];

    let status = Command::new("yt-dlp")
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if status.is_err() {
        return false;
    }

    let caption_file = format!("{}.en.{}", output_path, sub_format);
    if !Path::new(&caption_file).exists() {
        return false;
    }

    if output_format == "txt" || output_format == "all" {
        if let Ok(content) = std::fs::read_to_string(&caption_file) {
            let mut text_lines: Vec<&str> = Vec::new();
            for raw in content.lines() {
                let line = raw.trim();
                if line.is_empty() {
                    continue;
                }
                if line.starts_with("WEBVTT")
                    || line.starts_with("Kind:")
                    || line.contains("-->")
                    || line.parse::<u64>().is_ok()
                    || line.starts_with("NOTE")
                {
                    continue;
                }
                text_lines.push(line);
            }
            let txt_output = format!("{}.txt", output_path);
            let _ = std::fs::write(&txt_output, text_lines.join("\n"));
        }
    }

    let final_name = format!("{}.{}", output_path, sub_format);
    if output_format != "all" && output_format != sub_format {
        let _ = std::fs::remove_file(&caption_file);
    } else if caption_file != final_name {
        let _ = std::fs::rename(&caption_file, &final_name);
    }

    true
}
