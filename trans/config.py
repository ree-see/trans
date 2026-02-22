"""TOML-based persistent configuration."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_config_dir

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


def get_config_path() -> Path:
    return Path(user_config_dir("trans")) / "config.toml"


@dataclass
class CacheConfig:
    ttl_days: int = 30


@dataclass
class DiarizationConfig:
    hf_token: str = ""


@dataclass
class Config:
    model: str = "base"
    format: str = "txt"
    language: str = ""
    output_dir: str = ""
    clipboard: bool = False
    quiet: bool = False
    keep_audio: bool = False
    cache: CacheConfig = field(default_factory=CacheConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file, falling back to defaults."""
    config_path = path or get_config_path()
    if not config_path.exists():
        return Config()

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return Config()

    defaults = data.get("defaults", {})
    cache_data = data.get("cache", {})
    diarization_data = data.get("diarization", {})

    return Config(
        model=defaults.get("model", "base"),
        format=defaults.get("format", "txt"),
        language=defaults.get("language", ""),
        output_dir=defaults.get("output_dir", ""),
        clipboard=defaults.get("clipboard", False),
        quiet=defaults.get("quiet", False),
        keep_audio=defaults.get("keep_audio", False),
        cache=CacheConfig(ttl_days=cache_data.get("ttl_days", 30)),
        diarization=DiarizationConfig(hf_token=diarization_data.get("hf_token", "")),
    )


def save_config(config: Config, path: Path | None = None) -> None:
    """Write config to TOML file."""
    config_path = path or get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "[defaults]",
        f'model = "{config.model}"',
        f'format = "{config.format}"',
        f'language = "{config.language}"',
        f'output_dir = "{config.output_dir}"',
        f'clipboard = {str(config.clipboard).lower()}',
        f'quiet = {str(config.quiet).lower()}',
        f'keep_audio = {str(config.keep_audio).lower()}',
        "",
        "[cache]",
        f"ttl_days = {config.cache.ttl_days}",
        "",
        "[diarization]",
        f'hf_token = "{config.diarization.hf_token}"',
        "",
    ]
    config_path.write_text("\n".join(lines))


# Valid keys for `trans config set`
_CONFIG_KEYS: dict[str, str] = {
    "model": "defaults.model",
    "format": "defaults.format",
    "language": "defaults.language",
    "output_dir": "defaults.output_dir",
    "clipboard": "defaults.clipboard",
    "quiet": "defaults.quiet",
    "keep_audio": "defaults.keep_audio",
    "cache.ttl_days": "cache.ttl_days",
    "diarization.hf_token": "diarization.hf_token",
}

SETTABLE_KEYS = list(_CONFIG_KEYS.keys())


def set_config_value(key: str, value: str, path: Path | None = None) -> Config:
    """Set a single config value by dotted key name."""
    if key not in _CONFIG_KEYS:
        raise ValueError(f"Unknown config key: {key}. Valid keys: {SETTABLE_KEYS}")

    config = load_config(path)
    bool_keys = {"clipboard", "quiet", "keep_audio"}
    int_keys = {"cache.ttl_days"}

    if key in bool_keys:
        typed_value = value.lower() in ("true", "1", "yes")
    elif key in int_keys:
        typed_value = int(value)
    else:
        typed_value = value

    if "." not in key:
        setattr(config, key, typed_value)
    elif key.startswith("cache."):
        setattr(config.cache, key.split(".", 1)[1], typed_value)
    elif key.startswith("diarization."):
        setattr(config.diarization, key.split(".", 1)[1], typed_value)

    save_config(config, path)
    return config
