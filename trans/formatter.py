"""Output file writing for transcript segments."""

import json
from pathlib import Path

from .utils import format_speaker_label, format_timestamp_srt, format_timestamp_vtt


def write_output(
    segments: list[dict],
    output_base: str,
    fmt: str,
    info: dict | None = None,
    diarized: bool = False,
) -> list[Path]:
    """
    Write transcript segments to one or more output files.

    Args:
        segments: List of {'start', 'end', 'text'} dicts (may include 'speaker').
        output_base: Base path without extension.
        fmt: One of 'txt', 'srt', 'vtt', 'json', 'all'.
        info: Optional dict with 'language', 'language_probability', 'duration'.
        diarized: Whether segments contain speaker labels.

    Returns:
        List of Path objects for created files.
    """
    has_speakers = diarized and segments and 'speaker' in segments[0]
    created: list[Path] = []

    if fmt in ('txt', 'all'):
        path = Path(f"{output_base}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            if has_speakers:
                current_speaker = None
                for seg in segments:
                    speaker = format_speaker_label(seg.get('speaker', 'UNKNOWN'))
                    if speaker != current_speaker:
                        if current_speaker is not None:
                            f.write('\n')
                        f.write(f"[{speaker}]\n")
                        current_speaker = speaker
                    f.write(seg['text'] + '\n')
            else:
                for seg in segments:
                    f.write(seg['text'] + '\n')
        created.append(path)

    if fmt in ('srt', 'all'):
        path = Path(f"{output_base}.srt")
        with open(path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start_time = format_timestamp_srt(seg['start'])
                end_time = format_timestamp_srt(seg['end'])
                text = seg['text']
                if has_speakers:
                    speaker = format_speaker_label(seg.get('speaker', 'UNKNOWN'))
                    text = f"[{speaker}] {text}"
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
        created.append(path)

    if fmt in ('vtt', 'all'):
        path = Path(f"{output_base}.vtt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                start_time = format_timestamp_vtt(seg['start'])
                end_time = format_timestamp_vtt(seg['end'])
                text = seg['text']
                if has_speakers:
                    speaker = format_speaker_label(seg.get('speaker', 'UNKNOWN'))
                    f.write(f"{start_time} --> {end_time}\n<v {speaker}>{text}\n\n")
                else:
                    f.write(f"{start_time} --> {end_time}\n{text}\n\n")
        created.append(path)

    if fmt in ('json', 'all'):
        path = Path(f"{output_base}.json")
        with open(path, 'w', encoding='utf-8') as f:
            output_data: dict = {
                'diarization': has_speakers,
                'segments': segments,
            }
            if info:
                output_data['language'] = info.get('language', '')
                output_data['language_probability'] = info.get('language_probability', 0)
                output_data['duration'] = info.get('duration', 0)
            if has_speakers:
                speakers = set(seg.get('speaker') for seg in segments)
                output_data['speakers'] = [
                    format_speaker_label(s) for s in sorted(speakers)
                ]
            json.dump(output_data, f, indent=2)
        created.append(path)

    return created
