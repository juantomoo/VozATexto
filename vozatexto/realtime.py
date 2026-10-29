"""
Módulo de transcripción en tiempo real desde el micrófono con segmentación inteligente y VAD.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vozatexto.audio import AudioCaptureStream, compute_rms
from vozatexto.config import AppConfig
from vozatexto.engine import SpeechEngine
from vozatexto.exporter import (
    TranscriptSegment,
    cache_last_session,
    format_timestamp,
    save_transcription,
)


def run_realtime_transcription(
    engine: SpeechEngine,
    config: AppConfig,
    console: Console,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    show_timestamps: Optional[bool] = None,
    device_index: Optional[int] = None,
    save_formats: Optional[List[str]] = None,
) -> List[TranscriptSegment]:
    """Ejecuta una sesión de transcripción interactiva en tiempo real por micrófono."""
    chosen_model = model_name or config.model_realtime
    chosen_lang = language if language is not None else config.default_language
    use_timestamps = show_timestamps if show_timestamps is not None else config.show_timestamps
    mic_idx = device_index if device_index is not None else config.mic_device_index

    # Panel informativo inicial
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="cyan bold")
    info_table.add_column()
    info_table.add_row("Modo:", "Tiempo Real (Micrófono)")
    info_table.add_row("Modelo:", f"{chosen_model} ({config.device} / {config.compute_type})")
    info_table.add_row("Idioma:", "Detección automática" if chosen_lang == "auto" else chosen_lang.upper())
    info_table.add_row("Marcas de tiempo:", "Activadas" if use_timestamps else "Desactivadas")
    if mic_idx is not None:
        info_table.add_row("Dispositivo Mic:", f"ID #{mic_idx}")

    console.print(
        Panel(
            info_table,
            title="[bold green]🎙️ Transcripción en Tiempo Real[/bold green]",
            border_style="green",
        )
    )

    # Precargar modelo antes de iniciar la captura para evitar lag inicial
    with console.status(f"[bold yellow]Inicializando modelo '{chosen_model}' en memoria...[/bold yellow]", spinner="dots"):
        engine.load_model(chosen_model)

    console.print()
    console.print("[bold red]● GRABANDO[/bold red] - Habla al micrófono de forma natural.")
    console.print("[dim]Presiona [bold white]Ctrl+C[/bold white] en cualquier momento para finalizar y guardar.[/dim]")
    console.print()
    console.rule("[bold]Transcripción en Vivo[/bold]")

    stream = AudioCaptureStream(device_index=mic_idx, block_duration_ms=100)
    try:
        stream.start()
    except Exception as e:
        console.print(f"[bold red]❌ Error al abrir el micrófono:[/bold red] {e}")
        return []

    all_segments: List[TranscriptSegment] = []
    accumulated_audio: List[np.ndarray] = []
    
    # Parámetros del detector de pausas de voz
    has_speech = False
    silence_frames = 0
    silence_threshold_frames = 8   # 8 * 100ms = 0.8s de silencio para emitir segmento
    max_frames_per_utterance = 60  # 6.0s máximo para forzar emisión y evitar retraso
    min_speech_frames = 6          # 0.6s mínimo de audio antes de procesar
    energy_threshold = 0.015       # Umbral base de RMS

    session_start_time = time.time()
    utterance_start_offset = 0.0

    try:
        while stream.is_running:
            chunk = stream.read_chunk(timeout=0.2)
            if chunk is None:
                continue

            current_time = time.time() - session_start_time
            rms = compute_rms(chunk)
            accumulated_audio.append(chunk)

            if rms > energy_threshold:
                if not has_speech:
                    has_speech = True
                    utterance_start_offset = max(0.0, current_time - 0.2)
                silence_frames = 0
            else:
                if has_speech:
                    silence_frames += 1

            # Condición para procesar el bloque de voz acumulado
            should_transcribe = False
            if has_speech and silence_frames >= silence_threshold_frames and len(accumulated_audio) >= min_speech_frames:
                should_transcribe = True
            elif len(accumulated_audio) >= max_frames_per_utterance and has_speech:
                should_transcribe = True

            if should_transcribe and accumulated_audio:
                audio_np = np.concatenate(accumulated_audio)
                # Transcripción del bloque con VAD
                segs = engine.transcribe_audio_chunk(
                    audio_data=audio_np,
                    model_name=chosen_model,
                    language=chosen_lang,
                    offset_seconds=utterance_start_offset,
                )

                for seg in segs:
                    all_segments.append(seg)
                    if use_timestamps:
                        ts = f"[dim cyan][{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}][/dim cyan]"
                        console.print(f"{ts} {seg.text}")
                    else:
                        console.print(seg.text)

                # Reset de estado para la siguiente frase
                accumulated_audio = []
                has_speech = False
                silence_frames = 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Deteniendo grabación por solicitud del usuario...[/yellow]")
    finally:
        stream.stop()

        # Procesar remanente si existía voz al momento de interrumpir
        if accumulated_audio and has_speech:
            audio_np = np.concatenate(accumulated_audio)
            segs = engine.transcribe_audio_chunk(
                audio_data=audio_np,
                model_name=chosen_model,
                language=chosen_lang,
                offset_seconds=utterance_start_offset,
            )
            for seg in segs:
                all_segments.append(seg)
                if use_timestamps:
                    ts = f"[dim cyan][{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}][/dim cyan]"
                    console.print(f"{ts} {seg.text}")
                else:
                    console.print(seg.text)

    total_duration = time.time() - session_start_time
    console.rule()
    console.print(
        f"[bold green]✔ Sesión finalizada.[/bold green] "
        f"Duración total: [cyan]{format_timestamp(total_duration)}[/cyan] | Segmentos: [cyan]{len(all_segments)}[/cyan]"
    )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"grabacion_en_vivo_{timestamp_str}"

    # Auto-guardado de seguridad en caché temporal
    if all_segments:
        cache_last_session(
            segments=all_segments,
            base_filename=base_name,
            metadata={
                "mode": "realtime",
                "model": chosen_model,
                "language": chosen_lang,
                "total_duration": round(total_duration, 2),
            },
        )

    if save_formats and all_segments:
        for fmt in save_formats:
            out_file = save_transcription(
                segments=all_segments,
                output_dir=config.output_dir,
                base_filename=base_name,
                format_type=fmt,
                include_timestamps=use_timestamps,
                metadata={
                    "mode": "realtime",
                    "model": chosen_model,
                    "language": chosen_lang,
                    "total_duration": round(total_duration, 2),
                },
            )
            console.print(f"[green]💾 Guardado ({fmt.upper()}):[/green] [underline]{out_file}[/underline]")

    return all_segments
