"""Whisper transcription engine with model reuse across batch runs."""

import subprocess
import sys
from pathlib import Path

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False


def get_file_duration(audio_file: str) -> float:
    """Return duration in seconds via ffprobe, or 0 on failure."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_file),
            ],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        pass
    return 0.0


def extract_audio_from_video(video_path: str, output_audio: str, quiet: bool = False) -> bool:
    """Extract audio track from a video file using ffmpeg."""
    if not quiet:
        print("→ Extracting audio from video...")

    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'libmp3lame',
        '-q:a', '2',
        str(output_audio),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            stderr = e.stderr.decode() if e.stderr else str(e)
            print(f"✗ Failed to extract audio: {stderr}")
        return False


class TranscriptionEngine:
    """Lazy-loading Whisper model, reused across multiple transcriptions."""

    def __init__(self, model_name: str = 'base'):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not HAS_FASTER_WHISPER:
                raise ImportError(
                    "faster-whisper is not installed. Run: pip install faster-whisper"
                )
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def transcribe(
        self,
        audio_file: str,
        language: str | None = None,
        quiet: bool = False,
    ) -> tuple[list[dict], dict]:
        """
        Transcribe audio file.

        Returns (segments, info) where segments is a list of
        {'start', 'end', 'text'} dicts and info contains language metadata.
        """
        if not quiet:
            print(f"  Loading {self.model_name} model...")

        total_duration = get_file_duration(audio_file)

        if not quiet:
            print("  Transcribing...")

        segments_gen, info = self.model.transcribe(
            audio_file,
            language=language or None,
            beam_size=5,
            vad_filter=False,
        )

        segments_list: list[dict] = []
        last_percent = 0

        for segment in segments_gen:
            seg_data = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
            }
            segments_list.append(seg_data)

            if not quiet and total_duration > 0:
                percent = min(100, int((segment.end / total_duration) * 100))
                if percent != last_percent:
                    sys.stdout.write(f"\r  Progress: {percent}%")
                    sys.stdout.flush()
                    last_percent = percent

        if not quiet and total_duration > 0:
            sys.stdout.write('\r  Progress: 100%\n')
            sys.stdout.flush()

        if not quiet and not segments_list:
            print("  Warning: No speech detected in audio")

        info_dict = {
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration,
        }
        return segments_list, info_dict
