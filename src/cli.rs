//! clap-based CLI entry point for trans.

use crate::cache::CacheManager;
use crate::config::{
    get_config_path, load_config, set_config_value, Config, SETTABLE_KEYS,
};
use crate::diarizer::{get_hf_token, has_pyannote, run_diarization};
use crate::downloader::{download_audio, extract_native_captions, get_video_info};
use crate::formatter::write_output;
use crate::transcriber::{extract_audio_from_video, get_file_duration, TranscriptionEngine};
use crate::utils::{
    assign_speakers_to_segments, get_video_id, is_audio_file, is_local_file, sanitize_filename,
    OUTPUT_FORMATS, WHISPER_MODELS,
};
use chrono::Local;
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc;
use std::thread;

#[derive(Parser, Debug)]
#[command(
    name = "trans",
    about = "Transcribe YouTube, TikTok, Twitch videos and local audio/video files.",
    disable_help_subcommand = true,
    arg_required_else_help = false
)]
struct Cli {
    #[arg(short = 'V', long = "version", global = false)]
    version: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Transcribe video/audio URLs or local files to text.
    Transcribe(TranscribeArgs),
    /// Manage the transcript cache.
    #[command(subcommand)]
    Cache(CacheCmd),
    /// Manage persistent configuration.
    #[command(subcommand)]
    Config(ConfigCmd),
}

#[derive(Args, Debug)]
struct TranscribeArgs {
    /// Video/audio URL(s) or local file path(s).
    #[arg(required = true)]
    inputs: Vec<String>,

    /// Output base path (no extension). Single input only.
    #[arg(short = 'o', long = "output")]
    output: Option<String>,

    /// Directory for output files.
    #[arg(long = "output-dir")]
    output_dir: Option<PathBuf>,

    /// Whisper model: tiny, base, small, medium, large.
    #[arg(short = 'm', long = "model")]
    model: Option<String>,

    /// Language code (e.g. en, es). Auto-detect if unset.
    #[arg(short = 'l', long = "language")]
    language: Option<String>,

    /// Output format: txt, srt, vtt, json, all.
    #[arg(short = 'f', long = "format")]
    format: Option<String>,

    /// Copy transcript to clipboard.
    #[arg(short = 'c', long = "clipboard")]
    clipboard: Option<bool>,

    /// Keep downloaded audio file.
    #[arg(short = 'k', long = "keep-audio", default_value_t = false)]
    keep_audio: bool,

    /// Add timestamp to output filename.
    #[arg(short = 't', long = "timestamp", default_value_t = false)]
    timestamp: bool,

    /// Minimal output (errors only).
    #[arg(short = 'q', long = "quiet")]
    quiet: Option<bool>,

    /// Path to cookies.txt for authenticated downloads.
    #[arg(long = "cookies")]
    cookies: Option<PathBuf>,

    /// Skip cache lookup and force fresh transcription.
    #[arg(long = "no-cache", default_value_t = false)]
    no_cache: bool,

    /// Skip native captions, always use Whisper.
    #[arg(long = "force-whisper", default_value_t = false)]
    force_whisper: bool,

    /// Enable speaker diarization (requires pyannote-audio).
    #[arg(short = 'd', long = "diarize", default_value_t = false)]
    diarize: bool,

    /// Number of speakers (helps diarization accuracy).
    #[arg(long = "num-speakers")]
    num_speakers: Option<u32>,

    /// Translate non-English audio to English.
    #[arg(long = "translate", default_value_t = false)]
    translate: bool,
}

#[derive(Subcommand, Debug)]
enum CacheCmd {
    /// Delete all cached transcripts.
    Clear,
    /// Show cache statistics.
    Stats,
}

#[derive(Subcommand, Debug)]
enum ConfigCmd {
    /// Show current configuration.
    Show,
    /// Set a persistent configuration value.
    Set {
        /// Config key.
        key: String,
        /// Value to set.
        value: String,
    },
}

pub fn run() -> ExitCode {
    let cli = Cli::parse();

    if cli.version {
        println!("trans {}", crate::VERSION);
        return ExitCode::SUCCESS;
    }

    match cli.command {
        None => {
            // Mirror Python: print help if no subcommand.
            // clap will already exit with help on missing required, but we want
            // friendly behavior on bare invocation.
            print_help();
            ExitCode::SUCCESS
        }
        Some(Command::Transcribe(args)) => transcribe_cmd(args),
        Some(Command::Cache(c)) => match c {
            CacheCmd::Clear => cache_clear(),
            CacheCmd::Stats => cache_stats(),
        },
        Some(Command::Config(c)) => match c {
            ConfigCmd::Show => config_show(),
            ConfigCmd::Set { key, value } => config_set(&key, &value),
        },
    }
}

fn print_help() {
    use clap::CommandFactory;
    let mut cmd = Cli::command();
    let _ = cmd.print_help();
    println!();
}

fn copy_to_clipboard(text: &str, quiet: bool) {
    match arboard::Clipboard::new() {
        Ok(mut cb) => match cb.set_text(text.to_string()) {
            Ok(_) => {
                if !quiet {
                    println!("📋 Copied to clipboard");
                }
            }
            Err(e) => {
                if !quiet {
                    println!("⚠️  Clipboard copy failed: {}", e);
                }
            }
        },
        Err(e) => {
            if !quiet {
                println!("⚠️  Clipboard unavailable: {}", e);
            }
        }
    }
}

fn output_base(
    title: &str,
    output: Option<&str>,
    output_dir: Option<&Path>,
    timestamp: bool,
    config: &Config,
) -> String {
    if let Some(o) = output {
        return o.to_string();
    }
    let mut safe = sanitize_filename(title, 50);
    if timestamp {
        let ts = Local::now().format("%Y%m%d_%H%M%S").to_string();
        safe = format!("{}_{}", safe, ts);
    }
    if let Some(dir) = output_dir {
        return dir.join(safe).to_string_lossy().to_string();
    }
    if !config.output_dir.is_empty() {
        return PathBuf::from(&config.output_dir)
            .join(&safe)
            .to_string_lossy()
            .to_string();
    }
    safe
}

fn resolve_str(cli_val: Option<&str>, config_val: &str, default: &str) -> String {
    if let Some(v) = cli_val {
        return v.to_string();
    }
    if !config_val.is_empty() {
        return config_val.to_string();
    }
    default.to_string()
}

fn print_output_files(out_base: &str, fmt: &str, extras: &[&str]) {
    let formats: Vec<&str> = if fmt == "all" {
        extras.to_vec()
    } else {
        vec![fmt]
    };
    for ext in formats {
        let p = PathBuf::from(format!("{}.{}", out_base, ext));
        if p.exists() {
            println!("  → {}", p.display());
        }
    }
}

#[allow(clippy::too_many_arguments)]
struct UrlCtx<'a> {
    output: Option<&'a str>,
    output_dir: Option<&'a Path>,
    model: &'a str,
    language: Option<&'a str>,
    fmt: &'a str,
    clipboard: bool,
    keep_audio: bool,
    timestamp: bool,
    quiet: bool,
    cookies: Option<&'a Path>,
    no_cache: bool,
    force_whisper: bool,
    diarize: bool,
    num_speakers: Option<u32>,
    translate: bool,
    engine: &'a TranscriptionEngine,
    cache: &'a CacheManager,
    config: &'a Config,
}

fn process_url(url: &str, ctx: &UrlCtx) -> bool {
    let video_id = get_video_id(url);
    let cookies_str = ctx.cookies.map(|p| p.to_string_lossy().to_string());

    if !ctx.no_cache {
        let lookup_fmt = if ctx.fmt == "all" { "txt" } else { ctx.fmt };
        if let Ok(Some((transcript, title))) =
            ctx.cache.get(&video_id, lookup_fmt, ctx.config.cache.ttl_days)
        {
            if !ctx.quiet {
                println!("\n💾 Using cached transcript for: {}", title);
            }
            let out_b = output_base(&title, ctx.output, ctx.output_dir, ctx.timestamp, ctx.config);
            let out_fmt = if ctx.fmt == "all" { "txt" } else { ctx.fmt };
            let out_file = format!("{}.{}", out_b, out_fmt);
            if let Err(e) = std::fs::write(&out_file, &transcript) {
                eprintln!("✗ Failed to write {}: {}", out_file, e);
                return false;
            }
            if !ctx.quiet {
                println!("✓ Transcript written to {}", out_file);
            }
            if ctx.clipboard {
                copy_to_clipboard(&transcript, ctx.quiet);
            }
            return true;
        }
    }

    let info = match get_video_info(url, cookies_str.as_deref(), ctx.quiet) {
        Ok(i) => i,
        Err(e) => {
            if !ctx.quiet {
                eprintln!("✗ {}", e);
            }
            return false;
        }
    };
    let video_title = if info.title.is_empty() {
        "video".to_string()
    } else {
        info.title.clone()
    };
    let duration = info.duration;
    let out_b = output_base(
        &video_title,
        ctx.output,
        ctx.output_dir,
        ctx.timestamp,
        ctx.config,
    );

    if !ctx.quiet {
        println!("\n{}", "=".repeat(60));
        println!("📹 {}", video_title);
        if duration > 0 {
            let mins = duration / 60;
            let secs = duration % 60;
            println!("⏱️  Duration: {}:{:02}", mins, secs);
        }
        println!("{}\n", "=".repeat(60));
    }

    if !ctx.force_whisper && extract_native_captions(url, &out_b, ctx.fmt, ctx.quiet) {
        if !ctx.quiet {
            println!("\n✓ Transcription complete (native captions)");
            print_output_files(&out_b, ctx.fmt, &["txt", "vtt"]);
        }
        if !ctx.no_cache {
            let txt_path = PathBuf::from(format!("{}.txt", out_b));
            if txt_path.exists() {
                if let Ok(text) = std::fs::read_to_string(&txt_path) {
                    let _ = ctx
                        .cache
                        .put(&video_id, url, &video_title, &text, "txt", None);
                    if !ctx.quiet {
                        println!("💾 Cached for future use");
                    }
                }
            }
        }
        if ctx.clipboard {
            let txt_path = PathBuf::from(format!("{}.txt", out_b));
            if let Ok(text) = std::fs::read_to_string(&txt_path) {
                copy_to_clipboard(&text, ctx.quiet);
            }
        }
        return true;
    }

    let audio_file = format!("{}.audio.mp3", out_b);
    if !ctx.quiet {
        println!("→ Downloading audio...");
    }
    let final_audio = match download_audio(url, &audio_file, cookies_str.as_deref(), ctx.quiet) {
        Ok(p) => p,
        Err(e) => {
            if !ctx.quiet {
                eprintln!("✗ Error during download: {}", e);
            }
            let _ = std::fs::remove_file(&audio_file);
            return false;
        }
    };

    let (mut segments, info_dict) = match ctx.engine.transcribe(
        &final_audio,
        ctx.language,
        ctx.quiet,
        ctx.translate,
    ) {
        Ok(r) => r,
        Err(e) => {
            if !ctx.quiet {
                eprintln!("✗ Error during transcription: {}", e);
            }
            let _ = std::fs::remove_file(&audio_file);
            return false;
        }
    };

    if ctx.diarize {
        let token = get_hf_token().unwrap_or_default();
        match run_diarization(&final_audio, &token, ctx.num_speakers, ctx.quiet) {
            Ok(diar_segs) => {
                assign_speakers_to_segments(&mut segments, &diar_segs);
            }
            Err(e) => {
                if !ctx.quiet {
                    println!("  Warning: Diarization failed: {}", e);
                    println!("  Continuing without speaker labels...");
                }
            }
        }
    }

    let created = match write_output(&segments, &out_b, ctx.fmt, Some(&info_dict), ctx.diarize) {
        Ok(c) => c,
        Err(e) => {
            if !ctx.quiet {
                eprintln!("✗ Error writing output: {}", e);
            }
            return false;
        }
    };

    if !ctx.keep_audio && Path::new(&final_audio).exists() {
        let _ = std::fs::remove_file(&final_audio);
    } else if ctx.keep_audio && !ctx.quiet {
        println!("  Audio saved: {}", final_audio);
    }

    if !ctx.quiet {
        println!("\n✓ Transcription complete (Whisper)");
        for p in &created {
            if p.exists() {
                let size = p.metadata().map(|m| m.len()).unwrap_or(0);
                println!("  → {} ({} bytes)", p.display(), size);
            }
        }
    }

    if !ctx.no_cache {
        let txt_path = PathBuf::from(format!("{}.txt", out_b));
        if txt_path.exists() {
            if let Ok(text) = std::fs::read_to_string(&txt_path) {
                let _ = ctx.cache.put(
                    &video_id,
                    url,
                    &video_title,
                    &text,
                    "txt",
                    Some(ctx.model),
                );
                if !ctx.quiet {
                    println!("💾 Cached for future use");
                }
            }
        }
    }

    if ctx.clipboard {
        let txt_path = PathBuf::from(format!("{}.txt", out_b));
        if let Ok(text) = std::fs::read_to_string(&txt_path) {
            copy_to_clipboard(&text, ctx.quiet);
        }
    }

    true
}

#[allow(clippy::too_many_arguments)]
struct LocalCtx<'a> {
    output: Option<&'a str>,
    output_dir: Option<&'a Path>,
    language: Option<&'a str>,
    fmt: &'a str,
    clipboard: bool,
    timestamp: bool,
    quiet: bool,
    diarize: bool,
    num_speakers: Option<u32>,
    translate: bool,
    engine: &'a TranscriptionEngine,
    config: &'a Config,
}

fn process_local(filepath: &str, ctx: &LocalCtx) -> bool {
    let fp = PathBuf::from(filepath);
    if !fp.exists() {
        println!("✗ File not found: {}", fp.display());
        return false;
    }

    let title = fp
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio")
        .to_string();
    let out_b = output_base(&title, ctx.output, ctx.output_dir, ctx.timestamp, ctx.config);

    let duration = get_file_duration(filepath);

    if !ctx.quiet {
        println!("\n{}", "=".repeat(60));
        println!(
            "📁 {}",
            fp.file_name().and_then(|s| s.to_str()).unwrap_or(filepath)
        );
        if duration > 0.0 {
            let total = duration as u64;
            let hours = total / 3600;
            let mins = (total % 3600) / 60;
            let secs = total % 60;
            if hours > 0 {
                println!("⏱️  Duration: {}:{:02}:{:02}", hours, mins, secs);
            } else {
                println!("⏱️  Duration: {}:{:02}", mins, secs);
            }
        }
        println!("{}\n", "=".repeat(60));
    }

    let mut audio_file = filepath.to_string();
    let mut temp_audio: Option<String> = None;
    if !is_audio_file(filepath) {
        let temp = format!("{}.temp_audio.mp3", out_b);
        if !extract_audio_from_video(filepath, &temp, ctx.quiet) {
            return false;
        }
        audio_file = temp.clone();
        temp_audio = Some(temp);
    }

    let result = (|| -> Option<bool> {
        let (mut segments, info_dict) = match ctx
            .engine
            .transcribe(&audio_file, ctx.language, ctx.quiet, ctx.translate)
        {
            Ok(r) => r,
            Err(e) => {
                if !ctx.quiet {
                    eprintln!("✗ Error during transcription: {}", e);
                }
                return Some(false);
            }
        };

        if ctx.diarize {
            let token = get_hf_token().unwrap_or_default();
            match run_diarization(&audio_file, &token, ctx.num_speakers, ctx.quiet) {
                Ok(diar_segs) => {
                    assign_speakers_to_segments(&mut segments, &diar_segs);
                }
                Err(e) => {
                    if !ctx.quiet {
                        println!("  Warning: Diarization failed: {}", e);
                    }
                }
            }
        }

        let created = match write_output(&segments, &out_b, ctx.fmt, Some(&info_dict), ctx.diarize)
        {
            Ok(c) => c,
            Err(e) => {
                if !ctx.quiet {
                    eprintln!("✗ Error writing output: {}", e);
                }
                return Some(false);
            }
        };

        if !ctx.quiet {
            println!("\n✓ Transcription complete");
            for p in &created {
                if p.exists() {
                    let size = p.metadata().map(|m| m.len()).unwrap_or(0);
                    println!("  → {} ({} bytes)", p.display(), size);
                }
            }
        }

        if ctx.clipboard {
            let txt_path = PathBuf::from(format!("{}.txt", out_b));
            if let Ok(text) = std::fs::read_to_string(&txt_path) {
                copy_to_clipboard(&text, ctx.quiet);
            }
        }
        Some(true)
    })()
    .unwrap_or(false);

    if let Some(t) = temp_audio {
        let _ = std::fs::remove_file(t);
    }

    result
}

fn transcribe_cmd(args: TranscribeArgs) -> ExitCode {
    let cfg = load_config();

    let eff_model = resolve_str(args.model.as_deref(), &cfg.model, "base");
    let eff_format = resolve_str(args.format.as_deref(), &cfg.format, "txt");
    let eff_language_string = resolve_str(args.language.as_deref(), &cfg.language, "");
    let eff_language: Option<&str> = if eff_language_string.is_empty() {
        None
    } else {
        Some(&eff_language_string)
    };
    let eff_clipboard = args.clipboard.unwrap_or(cfg.clipboard);
    let eff_quiet = args.quiet.unwrap_or(cfg.quiet);

    if !WHISPER_MODELS.contains(&eff_model.as_str()) {
        println!(
            "✗ Invalid model '{}'. Choose from: {}",
            eff_model,
            WHISPER_MODELS.join(", ")
        );
        return ExitCode::from(1);
    }
    if !OUTPUT_FORMATS.contains(&eff_format.as_str()) {
        println!(
            "✗ Invalid format '{}'. Choose from: {}",
            eff_format,
            OUTPUT_FORMATS.join(", ")
        );
        return ExitCode::from(1);
    }
    if args.output.is_some() && args.inputs.len() > 1 {
        println!("✗ -o/--output can only be used with a single input");
        return ExitCode::from(1);
    }

    if args.diarize {
        if !has_pyannote() {
            println!("✗ Speaker diarization requires pyannote-audio.");
            println!("  Install: pip install pyannote-audio");
            return ExitCode::from(1);
        }
        if get_hf_token().is_none() {
            println!("✗ Speaker diarization requires a HuggingFace token.");
            println!("  1. Create at https://huggingface.co/settings/tokens");
            println!(
                "  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1"
            );
            println!("  3. Set HF_TOKEN env var or run: huggingface-cli login");
            return ExitCode::from(1);
        }
    }

    let cache = CacheManager::new();
    let engine = TranscriptionEngine::new(eff_model.clone());

    let urls: Vec<String> = args
        .inputs
        .iter()
        .filter(|i| !is_local_file(i))
        .cloned()
        .collect();

    // Pre-download URLs in parallel (max 3 threads) when batching.
    let mut downloaded: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    if urls.len() > 1 && urls.iter().all(|u| !is_local_file(u)) {
        let (tx, rx) = mpsc::channel();
        let mut active = 0usize;
        let mut idx = 0usize;
        let max_workers = 3usize;
        let cookies_clone = args.cookies.clone();
        let output_dir_clone = args.output_dir.clone();
        let cfg_clone = cfg.clone();
        let timestamp = args.timestamp;
        let quiet = eff_quiet;

        let urls_vec = urls.clone();

        while idx < urls_vec.len() || active > 0 {
            while active < max_workers && idx < urls_vec.len() {
                let url = urls_vec[idx].clone();
                let tx = tx.clone();
                let cookies = cookies_clone.clone();
                let output_dir = output_dir_clone.clone();
                let cfg = cfg_clone.clone();
                idx += 1;
                active += 1;
                thread::spawn(move || {
                    let cookies_str = cookies.as_ref().map(|p| p.to_string_lossy().to_string());
                    let info = get_video_info(&url, cookies_str.as_deref(), quiet)
                        .unwrap_or_default();
                    let title = if info.title.is_empty() {
                        get_video_id(&url)
                    } else {
                        info.title
                    };
                    let out_b = output_base(
                        &title,
                        None,
                        output_dir.as_deref(),
                        timestamp,
                        &cfg,
                    );
                    let audio_path = format!("{}.audio.mp3", out_b);
                    let dl =
                        download_audio(&url, &audio_path, cookies_str.as_deref(), true).ok();
                    let _ = tx.send((url, dl));
                });
            }
            if let Ok((url, audio)) = rx.recv() {
                active -= 1;
                if let Some(a) = audio {
                    downloaded.insert(url, a);
                }
            }
        }
        // (downloaded is currently unused beyond the parallel side-effect of
        // having pre-fetched the audio onto disk; the per-URL flow re-derives
        // the same path and download_audio() will short-circuit if the file
        // already exists. This mirrors the Python implementation's behavior
        // where pre-downloaded files are simply available on disk.)
        let _ = downloaded;
    }

    let mut success = 0;
    let mut fail = 0;
    let total_inputs = args.inputs.len();

    for inp in &args.inputs {
        let ok = if is_local_file(inp) {
            let ctx = LocalCtx {
                output: args.output.as_deref(),
                output_dir: args.output_dir.as_deref(),
                language: eff_language,
                fmt: &eff_format,
                clipboard: eff_clipboard,
                timestamp: args.timestamp,
                quiet: eff_quiet,
                diarize: args.diarize,
                num_speakers: args.num_speakers,
                translate: args.translate,
                engine: &engine,
                config: &cfg,
            };
            process_local(inp, &ctx)
        } else {
            let ctx = UrlCtx {
                output: args.output.as_deref(),
                output_dir: args.output_dir.as_deref(),
                model: &eff_model,
                language: eff_language,
                fmt: &eff_format,
                clipboard: eff_clipboard,
                keep_audio: args.keep_audio,
                timestamp: args.timestamp,
                quiet: eff_quiet,
                cookies: args.cookies.as_deref(),
                no_cache: args.no_cache,
                force_whisper: args.force_whisper,
                diarize: args.diarize,
                num_speakers: args.num_speakers,
                translate: args.translate,
                engine: &engine,
                cache: &cache,
                config: &cfg,
            };
            process_url(inp, &ctx)
        };
        if ok {
            success += 1;
        } else {
            fail += 1;
        }
    }

    if total_inputs > 1 && !eff_quiet {
        println!("\n{}", "=".repeat(60));
        println!("Summary: {} succeeded, {} failed", success, fail);
        println!("{}", "=".repeat(60));
    }

    if fail > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn cache_clear() -> ExitCode {
    let cache = CacheManager::new();
    match cache.clear() {
        Ok(n) => {
            println!("✓ Cleared {} cached transcript(s)", n);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("✗ {}", e);
            ExitCode::from(1)
        }
    }
}

fn cache_stats() -> ExitCode {
    let cache = CacheManager::new();
    match cache.stats() {
        Ok(s) => {
            println!("Entries : {}", s.count);
            println!("Size    : {} MB", s.size_mb);
            println!("Oldest  : {}", s.oldest.unwrap_or_else(|| "n/a".to_string()));
            println!("Newest  : {}", s.newest.unwrap_or_else(|| "n/a".to_string()));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("✗ {}", e);
            ExitCode::from(1)
        }
    }
}

fn config_show() -> ExitCode {
    let cfg = load_config();
    let path = get_config_path();
    println!("Config file: {}", path.display());
    println!();
    println!("model       = {}", cfg.model);
    println!("format      = {}", cfg.format);
    println!(
        "language    = {}",
        if cfg.language.is_empty() {
            "(auto)".to_string()
        } else {
            cfg.language.clone()
        }
    );
    println!(
        "output_dir  = {}",
        if cfg.output_dir.is_empty() {
            "(cwd)".to_string()
        } else {
            cfg.output_dir.clone()
        }
    );
    println!("clipboard   = {}", cfg.clipboard);
    println!("quiet       = {}", cfg.quiet);
    println!("keep_audio  = {}", cfg.keep_audio);
    println!("cache.ttl_days          = {}", cfg.cache.ttl_days);
    println!(
        "diarization.hf_token    = {}",
        if cfg.diarization.hf_token.is_empty() {
            "(not set)"
        } else {
            "(set)"
        }
    );
    ExitCode::SUCCESS
}

fn config_set(key: &str, value: &str) -> ExitCode {
    match set_config_value(key, value) {
        Ok(_) => {
            println!("✓ Set {} = {}", key, value);
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("✗ {}", e);
            // Show valid keys hint matching Python's flow.
            let _ = SETTABLE_KEYS;
            ExitCode::from(1)
        }
    }
}
