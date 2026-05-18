"""Speaker diarization via pyannote-audio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import assign_speakers_to_segments

try:
    from pyannote.audio import Pipeline as DiarizationPipeline

    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False


def get_hf_token(config_token: str = "") -> str | None:
    """Resolve the HuggingFace token from env, config, or cache file.

    Precedence (first non-empty wins):
      1. HF_TOKEN env var
      2. HUGGING_FACE_HUB_TOKEN env var
      3. config_token (caller passes cfg.diarization.hf_token)
      4. ~/.cache/huggingface/token (written by huggingface-cli login)
    """
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or config_token
        or _read_hf_cache_token()
    )
    return token or None


def _read_hf_cache_token() -> str | None:
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    try:
        return token_path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def run_diarization(
    audio_file: str,
    hf_token: str,
    num_speakers: int | None = None,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """
    Run speaker diarization using pyannote-audio.

    Returns a list of {'start', 'end', 'speaker'} dicts.
    """
    if not HAS_PYANNOTE:
        raise ImportError("pyannote-audio not installed. Run: pip install pyannote-audio")

    if not hf_token:
        raise ValueError(
            "HuggingFace token required for speaker diarization.\n"
            "1. Create a token at https://huggingface.co/settings/tokens\n"
            "2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Set HF_TOKEN environment variable or run: huggingface-cli login"
        )

    if not quiet:
        print("  Loading diarization model...")

    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    if not quiet:
        print("  Running speaker diarization...")

    diarization_args = {}
    if num_speakers:
        diarization_args["num_speakers"] = num_speakers

    diarization = pipeline(audio_file, **diarization_args)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    if not quiet:
        unique_speakers = len(set(s["speaker"] for s in segments))
        print(f"  Detected {unique_speakers} speaker(s)")

    return segments
