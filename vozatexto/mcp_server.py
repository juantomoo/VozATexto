"""
Servidor MCP (Model Context Protocol) para VozATexto.
Permite transcribir archivos de audio/video, consultar última transcripción,
gestionar glosarios y aplicar correcciones desde agentes de IA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from vozatexto import __version__
from vozatexto.audio import list_input_devices
from vozatexto.config import AppConfig
from vozatexto.corrections import CorrectionsManager
from vozatexto.engine import SpeechEngine
from vozatexto.exporter import (
    TranscriptSegment,
    cache_last_session,
    load_cached_session,
    save_transcription,
    segments_to_txt,
)

server = MCPServer("vozatexto")
_config = AppConfig.load()
_engine = SpeechEngine(_config)


@server.tool()
def transcribe_file(
    audio_path: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
    timestamps: bool = True,
    export_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Transcribe un archivo de audio o video local (mp3, wav, m4a, ogg, flac, mp4, etc.)
    utilizando faster-whisper con Silero VAD, mitigación de alucinaciones y glosario de usuario.
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        return {"error": f"El archivo de audio '{audio_path}' no existe en la ruta indicada."}

    chosen_model = model or _config.model_file
    chosen_lang = language or _config.default_language
    formats = export_formats or ["txt", "srt", "json"]

    segments, metadata = _engine.transcribe_file(
        audio_path=path,
        model_name=chosen_model,
        language=chosen_lang,
    )

    base_name = f"{path.stem}_transcripcion"
    saved_files: Dict[str, str] = {}

    if segments:
        # Auto-guardado en caché temporal
        cache_last_session(
            segments=segments,
            base_filename=base_name,
            metadata={
                "source_file": str(path),
                **metadata,
            },
        )

        for fmt in formats:
            try:
                saved = save_transcription(
                    segments=segments,
                    output_dir=_config.output_dir,
                    base_filename=base_name,
                    format_type=fmt,
                    include_timestamps=timestamps,
                    metadata={"source_file": str(path), **metadata},
                )
                saved_files[fmt] = str(saved)
            except Exception as e:
                saved_files[fmt] = f"Error al guardar: {e}"

    full_text = " ".join(s.text for s in segments)
    formatted_txt = segments_to_txt(segments, include_timestamps=timestamps)

    return {
        "status": "success",
        "source_file": str(path),
        "total_segments": len(segments),
        "metadata": metadata,
        "saved_files": saved_files,
        "formatted_text": formatted_txt,
        "full_text": full_text,
        "segments": [s.to_dict() for s in segments],
    }


@server.tool()
def get_last_transcription() -> Dict[str, Any]:
    """
    Recupera la última transcripción almacenada en la memoria temporal de VozATexto,
    incluyendo segmentos, texto completo y metadatos.
    """
    cached = load_cached_session()
    if not cached or not cached.get("segments"):
        return {"status": "empty", "message": "No hay transcripciones previas en la memoria temporal."}

    return {
        "status": "success",
        "base_filename": cached.get("base_filename"),
        "metadata": cached.get("metadata", {}),
        "total_segments": len(cached.get("segments", [])),
        "full_text": cached.get("full_text", ""),
        "segments": [s.to_dict() for s in cached.get("segments", [])],
    }


@server.tool()
def apply_corrections(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Aplica las reglas fonéticas y el glosario de usuario (correcciones.json)
    sobre una transcripción existente (.txt, .srt, .vtt, .json) o sobre la última sesión en memoria.
    """
    mgr = CorrectionsManager()

    if file_path:
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            return {"error": f"Archivo no encontrado: {file_path}"}
        content = p.read_text(encoding="utf-8")
        corregido, cambios = mgr.correct_text(content)
        out_p = p.with_name(f"{p.stem}_corregido{p.suffix}")
        out_p.write_text(corregido, encoding="utf-8")
        return {
            "status": "success",
            "source_file": str(p),
            "output_file": str(out_p),
            "changes_applied": cambios,
        }

    # Aplicar a la última sesión en memoria
    cached = load_cached_session()
    if not cached or not cached.get("segments"):
        return {"status": "empty", "message": "No hay transcripción en la memoria temporal."}

    segments = cached["segments"]
    corregidos, cambios = mgr.correct_segments(segments)
    base_name = f"{cached['base_filename']}_corregido"

    saved_files = {}
    for fmt in ["txt", "srt", "vtt", "json"]:
        saved = save_transcription(
            segments=corregidos,
            output_dir=_config.output_dir,
            base_filename=base_name,
            format_type=fmt,
            include_timestamps=_config.show_timestamps,
        )
        saved_files[fmt] = str(saved)

    cache_last_session(corregidos, base_name, {**cached.get("metadata", {}), "correcciones": cambios})

    return {
        "status": "success",
        "changes_applied": cambios,
        "saved_files": saved_files,
        "total_segments": len(corregidos),
    }


@server.tool()
def manage_vocabulary(
    action: str,
    term: Optional[str] = None,
    replacement_from: Optional[str] = None,
    replacement_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gestiona el glosario y reglas de corrección (correcciones.json).
    Acciones:
    - 'list': Ver vocabulario guía y reemplazos fonéticos actuales.
    - 'add_term': Agregar un nombre propio o topónimo al vocabulario guía de Whisper.
    - 'add_replacement': Agregar una regla de sustitución (replacement_from -> replacement_to).
    """
    mgr = CorrectionsManager()

    if action == "list":
        return {
            "rules_file": str(mgr.file_path),
            "vocabulario_guia": mgr.vocabulario_guia,
            "reemplazos": mgr.reemplazos,
        }

    elif action == "add_term":
        if not term:
            return {"error": "Debes especificar el parámetro 'term' para agregar al vocabulario."}
        mgr.add_vocab_term(term)
        return {
            "status": "success",
            "added_term": term,
            "vocabulario_guia": mgr.vocabulario_guia,
        }

    elif action == "add_replacement":
        if not replacement_from or not replacement_to:
            return {"error": "Debes especificar 'replacement_from' y 'replacement_to'."}
        mgr.add_replacement(replacement_from, replacement_to)
        return {
            "status": "success",
            "added_replacement": {replacement_from: replacement_to},
            "reemplazos": mgr.reemplazos,
        }

    return {"error": f"Acción desconocida '{action}'. Usa 'list', 'add_term' o 'add_replacement'."}


@server.tool()
def list_audio_devices() -> Dict[str, Any]:
    """
    Devuelve la lista de dispositivos de captura de audio (micrófonos)
    detectados en el sistema y la configuración actual de hardware.
    """
    devices = list_input_devices()
    return {
        "status": "success",
        "compute_device": _config.device,
        "compute_type": _config.compute_type,
        "cpu_threads": _config.cpu_threads,
        "default_language": _config.default_language,
        "active_mic_index": _config.mic_device_index,
        "devices": devices,
    }


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
