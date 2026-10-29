"""
Exportador de transcripciones a formatos TXT, SRT, VTT y JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text.strip(),
        }


def format_timestamp(seconds: float, delimiter: str = ".", include_hours: bool = False) -> str:
    """Convierte segundos a formato de tiempo mm:ss.mmm o hh:mm:ss.mmm."""
    if seconds < 0:
        seconds = 0.0
    total_seconds = int(seconds)
    millis = int(round((seconds - total_seconds) * 1000))
    if millis >= 1000:
        millis = 999
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{delimiter}{millis:03d}"
    return f"{minutes:02d}:{secs:02d}{delimiter}{millis:03d}"


def format_srt_time(seconds: float) -> str:
    """Formato estándar para SRT: 00:00:00,000."""
    return format_timestamp(seconds, delimiter=",", include_hours=True)


def format_vtt_time(seconds: float) -> str:
    """Formato estándar para WebVTT: 00:00:00.000."""
    return format_timestamp(seconds, delimiter=".", include_hours=True)


def segments_to_txt(segments: List[TranscriptSegment], include_timestamps: bool = True) -> str:
    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if include_timestamps:
            start_str = format_timestamp(seg.start)
            end_str = format_timestamp(seg.end)
            lines.append(f"[{start_str} --> {end_str}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def segments_to_srt(segments: List[TranscriptSegment]) -> str:
    blocks = []
    idx = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        blocks.append(
            f"{idx}\n"
            f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n"
            f"{text}\n"
        )
        idx += 1
    return "\n".join(blocks)


def segments_to_vtt(segments: List[TranscriptSegment]) -> str:
    lines = ["WEBVTT\n"]
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append(
            f"{format_vtt_time(seg.start)} --> {format_vtt_time(seg.end)}\n"
            f"{text}\n"
        )
    return "\n".join(lines)


def segments_to_json(segments: List[TranscriptSegment], metadata: Optional[Dict[str, Any]] = None) -> str:
    data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            **(metadata or {}),
        },
        "segments": [seg.to_dict() for seg in segments],
        "full_text": " ".join(seg.text.strip() for seg in segments if seg.text.strip()),
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def save_transcription(
    segments: List[TranscriptSegment],
    output_dir: Path | str,
    base_filename: str,
    format_type: str = "txt",
    include_timestamps: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = format_type.lower().strip()
    if fmt == "txt":
        content = segments_to_txt(segments, include_timestamps=include_timestamps)
        extension = ".txt"
    elif fmt == "srt":
        content = segments_to_srt(segments)
        extension = ".srt"
    elif fmt == "vtt":
        content = segments_to_vtt(segments)
        extension = ".vtt"
    elif fmt == "json":
        content = segments_to_json(segments, metadata=metadata)
        extension = ".json"
    else:
        raise ValueError(f"Formato de exportación desconocido: {format_type}")

    file_path = out_dir / f"{base_filename}{extension}"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


def cache_last_session(
    segments: List[TranscriptSegment],
    base_filename: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Guarda automáticamente una copia temporal en caché de la última transcripción procesada."""
    from vozatexto.config import LAST_TRANSCRIPT_CACHE_PATH
    LAST_TRANSCRIPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "base_filename": base_filename,
        "metadata": {
            "cached_at": datetime.now().isoformat(),
            **(metadata or {}),
        },
        "segments": [seg.to_dict() for seg in segments],
        "full_text": " ".join(seg.text.strip() for seg in segments if seg.text.strip()),
    }
    with open(LAST_TRANSCRIPT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return LAST_TRANSCRIPT_CACHE_PATH


def load_cached_session() -> Optional[Dict[str, Any]]:
    """Carga la última transcripción guardada en la caché temporal."""
    from vozatexto.config import LAST_TRANSCRIPT_CACHE_PATH
    if not LAST_TRANSCRIPT_CACHE_PATH.exists():
        return None
    try:
        with open(LAST_TRANSCRIPT_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = [
            TranscriptSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in data.get("segments", [])
        ]
        return {
            "base_filename": data.get("base_filename", "ultima_transcripcion"),
            "metadata": data.get("metadata", {}),
            "segments": segments,
            "full_text": data.get("full_text", ""),
        }
    except Exception:
        return None
