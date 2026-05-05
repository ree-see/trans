//! TOML-based persistent configuration.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub fn get_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("trans")
        .join("config.toml")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default = "default_ttl_days")]
    pub ttl_days: i64,
}

fn default_ttl_days() -> i64 {
    30
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { ttl_days: 30 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiarizationConfig {
    #[serde(default)]
    pub hf_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub output_dir: String,
    #[serde(default)]
    pub clipboard: bool,
    #[serde(default)]
    pub quiet: bool,
    #[serde(default)]
    pub keep_audio: bool,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub diarization: DiarizationConfig,
}

fn default_model() -> String {
    "base".to_string()
}

fn default_format() -> String {
    "txt".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: default_model(),
            format: default_format(),
            language: String::new(),
            output_dir: String::new(),
            clipboard: false,
            quiet: false,
            keep_audio: false,
            cache: CacheConfig::default(),
            diarization: DiarizationConfig::default(),
        }
    }
}

/// Internal TOML file structure (matches Python's nested layout).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TomlFile {
    #[serde(default)]
    defaults: TomlDefaults,
    #[serde(default)]
    cache: CacheConfig,
    #[serde(default)]
    diarization: DiarizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TomlDefaults {
    #[serde(default = "default_model")]
    model: String,
    #[serde(default = "default_format")]
    format: String,
    #[serde(default)]
    language: String,
    #[serde(default)]
    output_dir: String,
    #[serde(default)]
    clipboard: bool,
    #[serde(default)]
    quiet: bool,
    #[serde(default)]
    keep_audio: bool,
}

impl Default for TomlDefaults {
    fn default() -> Self {
        Self {
            model: default_model(),
            format: default_format(),
            language: String::new(),
            output_dir: String::new(),
            clipboard: false,
            quiet: false,
            keep_audio: false,
        }
    }
}

pub fn load_config_from(path: &Path) -> Config {
    if !path.exists() {
        return Config::default();
    }
    let text = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Config::default(),
    };
    let parsed: TomlFile = match toml::from_str(&text) {
        Ok(t) => t,
        Err(_) => return Config::default(),
    };
    Config {
        model: parsed.defaults.model,
        format: parsed.defaults.format,
        language: parsed.defaults.language,
        output_dir: parsed.defaults.output_dir,
        clipboard: parsed.defaults.clipboard,
        quiet: parsed.defaults.quiet,
        keep_audio: parsed.defaults.keep_audio,
        cache: parsed.cache,
        diarization: parsed.diarization,
    }
}

pub fn load_config() -> Config {
    load_config_from(&get_config_path())
}

pub fn save_config_to(config: &Config, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lines = vec![
        "[defaults]".to_string(),
        format!("model = \"{}\"", config.model),
        format!("format = \"{}\"", config.format),
        format!("language = \"{}\"", config.language),
        format!("output_dir = \"{}\"", config.output_dir),
        format!("clipboard = {}", config.clipboard),
        format!("quiet = {}", config.quiet),
        format!("keep_audio = {}", config.keep_audio),
        String::new(),
        "[cache]".to_string(),
        format!("ttl_days = {}", config.cache.ttl_days),
        String::new(),
        "[diarization]".to_string(),
        format!("hf_token = \"{}\"", config.diarization.hf_token),
        String::new(),
    ];
    std::fs::write(path, lines.join("\n"))?;
    Ok(())
}

pub fn save_config(config: &Config) -> Result<()> {
    save_config_to(config, &get_config_path())
}

pub const SETTABLE_KEYS: &[&str] = &[
    "model",
    "format",
    "language",
    "output_dir",
    "clipboard",
    "quiet",
    "keep_audio",
    "cache.ttl_days",
    "diarization.hf_token",
];

pub fn set_config_value_at(key: &str, value: &str, path: &Path) -> Result<Config> {
    if !SETTABLE_KEYS.contains(&key) {
        return Err(anyhow!(
            "Unknown config key: {}. Valid keys: {}",
            key,
            SETTABLE_KEYS.join(", ")
        ));
    }
    let mut config = load_config_from(path);
    let bool_keys = ["clipboard", "quiet", "keep_audio"];
    let int_keys = ["cache.ttl_days"];

    let parse_bool = |v: &str| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");

    if bool_keys.contains(&key) {
        let b = parse_bool(value);
        match key {
            "clipboard" => config.clipboard = b,
            "quiet" => config.quiet = b,
            "keep_audio" => config.keep_audio = b,
            _ => unreachable!(),
        }
    } else if int_keys.contains(&key) {
        let n: i64 = value
            .parse()
            .map_err(|e| anyhow!("invalid integer for {}: {}", key, e))?;
        if key == "cache.ttl_days" {
            config.cache.ttl_days = n;
        }
    } else {
        match key {
            "model" => config.model = value.to_string(),
            "format" => config.format = value.to_string(),
            "language" => config.language = value.to_string(),
            "output_dir" => config.output_dir = value.to_string(),
            "diarization.hf_token" => config.diarization.hf_token = value.to_string(),
            _ => unreachable!(),
        }
    }

    save_config_to(&config, path)?;
    Ok(config)
}

pub fn set_config_value(key: &str, value: &str) -> Result<Config> {
    set_config_value_at(key, value, &get_config_path())
}
