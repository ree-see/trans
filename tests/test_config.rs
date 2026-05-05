//! Config TOML round-trip tests using a temp config path.

use tempfile::tempdir;
use trans::config::{load_config_from, save_config_to, set_config_value_at, Config};

#[test]
fn defaults_when_no_file() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("none.toml");
    let cfg = load_config_from(&path);
    assert_eq!(cfg.model, "base");
    assert_eq!(cfg.format, "txt");
    assert_eq!(cfg.cache.ttl_days, 30);
    assert!(!cfg.clipboard);
}

#[test]
fn round_trip_save_load() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("cfg.toml");
    let cfg = Config {
        model: "small".to_string(),
        format: "srt".to_string(),
        language: "en".to_string(),
        clipboard: true,
        cache: trans::config::CacheConfig { ttl_days: 7 },
        diarization: trans::config::DiarizationConfig {
            hf_token: "abc".to_string(),
        },
        ..Config::default()
    };
    save_config_to(&cfg, &path).unwrap();

    let loaded = load_config_from(&path);
    assert_eq!(loaded.model, "small");
    assert_eq!(loaded.format, "srt");
    assert_eq!(loaded.language, "en");
    assert!(loaded.clipboard);
    assert_eq!(loaded.cache.ttl_days, 7);
    assert_eq!(loaded.diarization.hf_token, "abc");
}

#[test]
fn set_value_string() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("cfg.toml");
    set_config_value_at("model", "medium", &path).unwrap();
    let loaded = load_config_from(&path);
    assert_eq!(loaded.model, "medium");
}

#[test]
fn set_value_bool() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("cfg.toml");
    set_config_value_at("clipboard", "true", &path).unwrap();
    assert!(load_config_from(&path).clipboard);
    set_config_value_at("clipboard", "false", &path).unwrap();
    assert!(!load_config_from(&path).clipboard);
}

#[test]
fn set_value_int() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("cfg.toml");
    set_config_value_at("cache.ttl_days", "14", &path).unwrap();
    assert_eq!(load_config_from(&path).cache.ttl_days, 14);
}

#[test]
fn set_value_unknown_key_errors() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("cfg.toml");
    let err = set_config_value_at("not_a_key", "x", &path);
    assert!(err.is_err());
}
