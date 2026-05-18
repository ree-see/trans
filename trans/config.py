"""TOML-based persistent configuration."""

from __future__ import annotations

import os
import sys
import tempfile
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
    device: str = "cpu"
    compute_type: str = "int8"
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
        device=defaults.get("device", "cpu"),
        compute_type=defaults.get("compute_type", "int8"),
        cache=CacheConfig(ttl_days=cache_data.get("ttl_days", 30)),
        diarization=DiarizationConfig(hf_token=diarization_data.get("hf_token", "")),
    )


def save_config(config: Config, path: Path | None = None) -> None:
    """Write config to TOML file atomically with owner-only (0o600) perms.

    The atomic mkstemp + chmod + os.replace pattern closes two windows:
    1. The default-umask race where a `write_text` + `chmod` sequence leaves
       the file briefly at 0o644, exposing the HF token to sibling user
       processes.
    2. Symlink-following on the data write path itself (mkstemp creates a
       fresh inode; os.replace swaps it in atomically without dereferencing
       a symlinked destination).

    Out of scope: the parent directory is not validated. If
    `config_path.parent` is itself a symlink (or under one), `mkdir` will
    happily traverse it. The 0o600 file mode still keeps content private to
    the owner via the regular path; defending against a hostile-parent-dir
    swap is left to the OS / user.

    On Windows POSIX mode bits are advisory; ACLs follow the parent dir.
    """
    config_path = path or get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "[defaults]",
        f'model = "{config.model}"',
        f'format = "{config.format}"',
        f'language = "{config.language}"',
        f'output_dir = "{config.output_dir}"',
        f"clipboard = {str(config.clipboard).lower()}",
        f"quiet = {str(config.quiet).lower()}",
        f"keep_audio = {str(config.keep_audio).lower()}",
        f'device = "{config.device}"',
        f'compute_type = "{config.compute_type}"',
        "",
        "[cache]",
        f"ttl_days = {config.cache.ttl_days}",
        "",
        "[diarization]",
        f'hf_token = "{config.diarization.hf_token}"',
        "",
    ]
    body = "\n".join(lines)

    fd, tmp_str = tempfile.mkstemp(prefix=".config.toml.", dir=config_path.parent)
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, config_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# Valid keys for `trans config set`
_CONFIG_KEYS: dict[str, str] = {
    "model": "defaults.model",
    "format": "defaults.format",
    "language": "defaults.language",
    "output_dir": "defaults.output_dir",
    "clipboard": "defaults.clipboard",
    "quiet": "defaults.quiet",
    "keep_audio": "defaults.keep_audio",
    "device": "defaults.device",
    "compute_type": "defaults.compute_type",
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
