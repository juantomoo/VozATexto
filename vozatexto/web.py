"""
Servidor Web FastAPI para la Interfaz Gráfica de VozATexto.
Proporciona endpoints REST y WebSocket para transcripción, gestión de glosarios y streaming.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from vozatexto import __version__
from vozatexto.audio import list_input_devices
from vozatexto.config import (
    AVAILABLE_LANGUAGES,
    AVAILABLE_MODELS,
    MODEL_DESCRIPTIONS,
    AppConfig,
)
from vozatexto.corrections import CorrectionsManager
from vozatexto.engine import SpeechEngine
from vozatexto.exporter import (
    TranscriptSegment,
    cache_last_session,
    load_cached_session,
    save_transcription,
    segments_to_json,
    segments_to_srt,
    segments_to_txt,
    segments_to_vtt,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="VozATexto Web UI", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_config = AppConfig.load()
_engine = SpeechEngine(_config)


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Error: Archivo index.html no encontrado.</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    devices = list_input_devices()
    return {
        "version": __version__,
        "device": _config.device,
        "compute_type": _config.compute_type,
        "cpu_threads": _config.cpu_threads,
        "output_dir": _config.output_dir,
        "default_language": _config.default_language,
        "model_file": _config.model_file,
        "model_realtime": _config.model_realtime,
        "show_timestamps": _config.show_timestamps,
        "available_models": AVAILABLE_MODELS,
        "model_descriptions": MODEL_DESCRIPTIONS,
        "available_languages": AVAILABLE_LANGUAGES,
        "input_devices": devices,
    }


@app.post("/api/transcribe-file")
async def transcribe_file_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form("auto"),
    model: Optional[str] = Form(None),
    timestamps: bool = Form(True),
):
    chosen_model = model or _config.model_file
    chosen_lang = language or _config.default_language

    # Guardar archivo subido en temporal
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        segments, metadata = _engine.transcribe_file(
            audio_path=tmp_path,
            model_name=chosen_model,
            language=chosen_lang,
        )

        orig_stem = Path(file.filename or "transcripcion").stem
        base_name = f"{orig_stem}_transcripcion"

        # Guardar en disco en los formatos estándar
        saved_files = {}
        if segments:
            cache_last_session(
                segments=segments,
                base_filename=base_name,
                metadata={
                    "source_file": file.filename,
                    **metadata,
                },
            )
            for fmt in ["txt", "srt", "vtt", "json"]:
                p = save_transcription(
                    segments=segments,
                    output_dir=_config.output_dir,
                    base_filename=base_name,
                    format_type=fmt,
                    include_timestamps=timestamps,
                    metadata={"source_file": file.filename, **metadata},
                )
                saved_files[fmt] = str(p)

        return {
            "status": "success",
            "filename": file.filename,
            "base_name": base_name,
            "metadata": metadata,
            "segments": [s.to_dict() for s in segments],
            "full_text": " ".join(s.text for s in segments),
            "formatted_txt": segments_to_txt(segments, include_timestamps=timestamps),
            "saved_files": saved_files,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@app.post("/api/transcribe-chunk")
async def transcribe_chunk_endpoint(
    chunk: UploadFile = File(...),
    language: Optional[str] = Form("auto"),
    model: Optional[str] = Form(None),
    offset: float = Form(0.0),
):
    """Transcribe un bloque de audio recibido desde el navegador (formato webm/wav/ogg)."""
    chosen_model = model or _config.model_realtime
    suffix = Path(chunk.filename or "chunk.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(chunk.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        segments, metadata = _engine.transcribe_file(
            audio_path=tmp_path,
            model_name=chosen_model,
            language=language,
        )
        # Ajustar offset temporal relativo
        for s in segments:
            s.start += offset
            s.end += offset

        return {
            "status": "success",
            "segments": [s.to_dict() for s in segments],
            "text": " ".join(s.text for s in segments),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "segments": []}
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@app.get("/api/last-transcription")
async def get_last_transcription_endpoint():
    cached = load_cached_session()
    if not cached or not cached.get("segments"):
        return {"status": "empty", "message": "No hay transcripciones previas en la memoria temporal."}

    segments = cached["segments"]
    return {
        "status": "success",
        "base_filename": cached.get("base_filename"),
        "metadata": cached.get("metadata", {}),
        "total_segments": len(segments),
        "full_text": cached.get("full_text", ""),
        "segments": [s.to_dict() for s in segments],
        "formatted_txt": segments_to_txt(segments, include_timestamps=_config.show_timestamps),
    }


@app.get("/api/vocabulary")
async def get_vocabulary():
    mgr = CorrectionsManager()
    mgr.load()
    return {
        "vocabulario_guia": mgr.vocabulario_guia,
        "reemplazos": mgr.reemplazos,
        "file_path": str(mgr.file_path),
    }


@app.post("/api/vocabulary/term")
async def add_term(term: str = Form(...)):
    mgr = CorrectionsManager()
    mgr.add_vocab_term(term)
    return {"status": "success", "vocabulario_guia": mgr.vocabulario_guia}


@app.delete("/api/vocabulary/term")
async def remove_term(term: str = Form(...)):
    mgr = CorrectionsManager()
    if term in mgr.vocabulario_guia:
        mgr.vocabulario_guia.remove(term)
        mgr.save()
    return {"status": "success", "vocabulario_guia": mgr.vocabulario_guia}


@app.post("/api/vocabulary/replacement")
async def add_replacement(from_text: str = Form(...), to_text: str = Form(...)):
    mgr = CorrectionsManager()
    mgr.add_replacement(from_text, to_text)
    return {"status": "success", "reemplazos": mgr.reemplazos}


@app.delete("/api/vocabulary/replacement")
async def remove_replacement(from_text: str = Form(...)):
    mgr = CorrectionsManager()
    if from_text in mgr.reemplazos:
        del mgr.reemplazos[from_text]
        mgr.save()
    return {"status": "success", "reemplazos": mgr.reemplazos}


@app.post("/api/apply-corrections")
async def apply_corrections_endpoint(data: Dict[str, Any]):
    """Aplica el glosario actual sobre una lista de segmentos enviada por el frontend."""
    mgr = CorrectionsManager()
    raw_segments = [
        TranscriptSegment(start=s["start"], end=s["end"], text=s["text"])
        for s in data.get("segments", [])
    ]
    corregidos, cambios = mgr.correct_segments(raw_segments)
    base_name = data.get("base_name", "transcripcion_corregida")

    saved_files = {}
    if data.get("save_to_disk", True) and corregidos:
        for fmt in ["txt", "srt", "vtt", "json"]:
            p = save_transcription(
                segments=corregidos,
                output_dir=_config.output_dir,
                base_filename=f"{base_name}_corregido",
                format_type=fmt,
                include_timestamps=data.get("timestamps", True),
            )
            saved_files[fmt] = str(p)

    return {
        "status": "success",
        "changes_applied": cambios,
        "segments": [s.to_dict() for s in corregidos],
        "full_text": " ".join(s.text for s in corregidos),
        "saved_files": saved_files,
    }


def start_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Inicia el servidor web FastAPI y abre automáticamente la interfaz en el navegador."""
    url = f"http://{host}:{port}"

    def _open():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if open_browser:
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n[VozATexto] Servidor Web UI iniciado en: {url}")
    print("[VozATexto] Presiona Ctrl+C para detener el servidor.\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
