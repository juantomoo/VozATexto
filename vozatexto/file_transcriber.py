"""
Lógica para la transcripción de archivos grabados de audio o video.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vozatexto.config import AppConfig
from vozatexto.engine import SpeechEngine
from vozatexto.exporter import (
    TranscriptSegment,
    cache_last_session,
    format_timestamp,
    save_transcription,
)


def format_file_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def process_audio_file(
    file_path: Union[str, Path],
    engine: SpeechEngine,
    config: AppConfig,
    console: Console,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    show_timestamps: Optional[bool] = None,
    save_formats: Optional[List[str]] = None,
) -> List[TranscriptSegment]:
    """Transcribe un archivo de audio/video y muestra resultados en consola."""
    path = Path(file_path).resolve()
    if not path.is_file():
        console.print(f"[bold red]❌ Error:[/bold red] El archivo '{path}' no existe.")
        return []

    chosen_model = model_name or config.model_file
    chosen_lang = language if language is not None else config.default_language
    use_timestamps = show_timestamps if show_timestamps is not None else config.show_timestamps

    file_size_str = format_file_size(path.stat().st_size)

    # Panel informativo inicial
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="cyan bold")
    info_table.add_column()
    info_table.add_row("Archivo:", str(path.name))
    info_table.add_row("Ruta:", str(path.parent))
    info_table.add_row("Tamaño:", file_size_str)
    info_table.add_row("Modelo:", f"{chosen_model} ({config.device} / {config.compute_type})")
    info_table.add_row("Idioma:", "Detección automática" if chosen_lang == "auto" else chosen_lang.upper())
    info_table.add_row("Marcas de tiempo:", "Activadas" if use_timestamps else "Desactivadas")

    console.print(
        Panel(
            info_table,
            title="[bold green]📁 Procesando Archivo[/bold green]",
            border_style="green",
        )
    )

    start_time = time.time()
    with console.status(
        f"[bold yellow]Cargando modelo '{chosen_model}' y analizando audio con Silero VAD...[/bold yellow]",
        spinner="dots",
    ):
        try:
            segments, metadata = engine.transcribe_file(
                audio_path=path,
                model_name=chosen_model,
                language=chosen_lang,
            )
        except Exception as e:
            console.print(f"[bold red]❌ Error durante la transcripción:[/bold red] {e}")
            return []

    elapsed = time.time() - start_time
    detected_lang = metadata.get("language", "desconocido")
    lang_prob = metadata.get("language_probability", 1.0)
    audio_duration = metadata.get("duration", 0.0)

    console.print()
    console.print(
        f"[bold green]✔ Transcripción completada en {elapsed:.2f}s[/bold green] "
        f"(Duración audio: {format_timestamp(audio_duration)} | Idioma: [cyan]{detected_lang.upper()}[/cyan] [dim]{lang_prob*100:.1f}%[/dim])"
    )
    console.print()

    # Mostrar transcripción en pantalla
    console.rule("[bold]Resultado de la Transcripción[/bold]")
    if not segments:
        console.print("[dim italic]No se detectó voz o contenido hablado en el archivo.[/dim italic]")
        return []

    for seg in segments:
        if use_timestamps:
            ts_str = f"[dim cyan][{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}][/dim cyan]"
            console.print(f"{ts_str} {seg.text}")
        else:
            console.print(seg.text)

    # Auto-guardado de seguridad en caché temporal (para nunca perder el trabajo)
    base_name = f"{path.stem}_transcripcion"
    if segments:
        cache_last_session(
            segments=segments,
            base_filename=base_name,
            metadata={
                "source_file": str(path),
                "model": chosen_model,
                "language": detected_lang,
                "elapsed_seconds": round(elapsed, 2),
            },
        )

    # Exportación opcional
    if save_formats:
        for fmt in save_formats:
            out_file = save_transcription(
                segments=segments,
                output_dir=config.output_dir,
                base_filename=base_name,
                format_type=fmt,
                include_timestamps=use_timestamps,
                metadata={
                    "source_file": str(path),
                    "model": chosen_model,
                    "language": detected_lang,
                    "elapsed_seconds": round(elapsed, 2),
                },
            )
            console.print(f"[green]💾 Guardado ({fmt.upper()}):[/green] [underline]{out_file}[/underline]")

    return segments
