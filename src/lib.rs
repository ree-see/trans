//! trans — video/audio transcription CLI.

pub mod cache;
pub mod cli;
pub mod config;
pub mod diarizer;
pub mod downloader;
pub mod formatter;
pub mod transcriber;
pub mod utils;

pub const VERSION: &str = "0.5.0";
