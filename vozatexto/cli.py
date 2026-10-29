"""
Interfaz de línea de comandos (CLI) interactiva para VozATexto.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
from vozatexto.exporter import load_cached_session, save_transcription
from vozatexto.file_transcriber import process_audio_file
from vozatexto.realtime import run_realtime_transcription

console = Console()


def render_banner() -> None:
    banner_text = (
        f"[bold cyan]VozATexto[/bold cyan] [dim]v{__version__}[/dim]\n"
        "[italic white]Transcriptor de voz a texto interactivo y multilingüe[/italic white]\n"
        "[dim]Motor: faster-whisper (CTranslate2) + Silero VAD[/dim]"
    )
    console.print(Panel(banner_text, border_style="cyan", expand=False))


def show_system_status(config: AppConfig) -> None:
    table = Table(title="Estado del Sistema y Hardware", border_style="dim")
    table.add_column("Propiedad", style="cyan bold")
    table.add_column("Valor", style="white")

    table.add_row("Dispositivo de cómputo:", f"{config.device.upper()} ({config.compute_type})")
    table.add_row("Hilos de CPU asignados:", str(config.cpu_threads))
    table.add_row("Idioma predeterminado:", AVAILABLE_LANGUAGES.get(config.default_language, config.default_language))
    table.add_row("Modelo en tiempo real:", config.model_realtime)
    table.add_row("Modelo para archivos:", config.model_file)
    table.add_row("Marcas de tiempo por defecto:", "Sí" if config.show_timestamps else "No")
    table.add_row("Carpeta de exportación:", config.output_dir)

    mic_name = "Predeterminado del sistema"
    if config.mic_device_index is not None:
        devices = list_input_devices()
        for d in devices:
            if d["index"] == config.mic_device_index:
                mic_name = d["label"]
                break
    table.add_row("Micrófono seleccionado:", mic_name)

    console.print(table)
    console.print()


def interactive_settings_menu(config: AppConfig) -> None:
    while True:
        choice = questionary.select(
            "⚙️ Configuración de VozATexto:",
            choices=[
                "1. Cambiar idioma predeterminado",
                "2. Cambiar modelo para tiempo real",
                "3. Cambiar modelo para archivos",
                "4. Alternar marcas de tiempo por defecto",
                "5. Seleccionar micrófono de entrada",
                "6. Cambiar carpeta de exportación",
                "Volver al Menú Principal",
            ],
        ).ask()

        if not choice or "Volver" in choice:
            break

        if "1. Cambiar idioma" in choice:
            lang_choices = [
                questionary.Choice(title=f"{code} - {label}", value=code)
                for code, label in AVAILABLE_LANGUAGES.items()
            ]
            new_lang = questionary.select(
                "Selecciona el idioma predeterminado:",
                choices=lang_choices,
                default=config.default_language,
            ).ask()
            if new_lang:
                config.default_language = new_lang
                config.save()
                console.print(f"[green]✔ Idioma actualizado a: {new_lang}[/green]")

        elif "2. Cambiar modelo para tiempo real" in choice:
            model_choices = [
                questionary.Choice(
                    title=f"{m} ({MODEL_DESCRIPTIONS.get(m, '')})",
                    value=m,
                )
                for m in AVAILABLE_MODELS
            ]
            new_model = questionary.select(
                "Selecciona modelo para tiempo real (micrófono):",
                choices=model_choices,
                default=config.model_realtime,
            ).ask()
            if new_model:
                config.model_realtime = new_model
                config.save()
                console.print(f"[green]✔ Modelo de tiempo real actualizado a: {new_model}[/green]")

        elif "3. Cambiar modelo para archivos" in choice:
            model_choices = [
                questionary.Choice(
                    title=f"{m} ({MODEL_DESCRIPTIONS.get(m, '')})",
                    value=m,
                )
                for m in AVAILABLE_MODELS
            ]
            new_model = questionary.select(
                "Selecciona modelo para archivos de audio/video:",
                choices=model_choices,
                default=config.model_file,
            ).ask()
            if new_model:
                config.model_file = new_model
                config.save()
                console.print(f"[green]✔ Modelo de archivo actualizado a: {new_model}[/green]")

        elif "4. Alternar marcas de tiempo" in choice:
            current = config.show_timestamps
            new_val = questionary.confirm(
                "¿Deseas activar las marcas de tiempo por defecto?",
                default=current,
            ).ask()
            config.show_timestamps = bool(new_val)
            config.save()
            estado = "Activadas" if config.show_timestamps else "Desactivadas"
            console.print(f"[green]✔ Marcas de tiempo: {estado}[/green]")

        elif "5. Seleccionar micrófono" in choice:
            devices = list_input_devices()
            if not devices:
                console.print("[red]No se detectaron micrófonos disponibles.[/red]")
                continue
            choices = [
                questionary.Choice(title="Dispositivo predeterminado del sistema", value=None)
            ] + [
                questionary.Choice(title=d["label"], value=d["index"])
                for d in devices
            ]
            selected_dev = questionary.select(
                "Selecciona el micrófono a usar:",
                choices=choices,
            ).ask()
            config.mic_device_index = selected_dev
            config.save()
            console.print("[green]✔ Configuración de micrófono guardada.[/green]")

        elif "6. Cambiar carpeta" in choice:
            new_dir = questionary.text(
                "Ingresa la ruta absoluta o relativa para guardar transcripciones:",
                default=config.output_dir,
            ).ask()
            if new_dir:
                path = Path(new_dir).resolve()
                path.mkdir(parents=True, exist_ok=True)
                config.output_dir = str(path)
                config.save()
                console.print(f"[green]✔ Carpeta actualizada: {config.output_dir}[/green]")


def prompt_save_segments(
    segments: list,
    config: AppConfig,
    base_name: str,
    default_timestamps: bool,
) -> None:
    if not segments:
        return

    saved_formats = set()

    while True:
        choice = questionary.select(
            "¿En qué formato deseas guardar la transcripción?",
            choices=[
                "💾 1. Guardar en TODOS los formatos (TXT, SRT, VTT, JSON) [Recomendado]",
                "📄 2. Guardar como TXT (Texto plano / marcas de tiempo)",
                "⏱️  3. Guardar como SRT (Subtítulos para video / YouTube / VLC)",
                "📊 4. Guardar como JSON (Estructurado con metadatos completos)",
                "🌐 5. Guardar como VTT (WebVTT para web)",
                "☑️  6. Selección manual con casillas (Espacio para marcar, Enter para confirmar)",
                "✔  Terminar y Volver al Menú Principal",
            ],
        ).ask()

        if not choice or "Terminar" in choice:
            break

        formats_to_save = []
        if "TODOS los formatos" in choice:
            formats_to_save = ["txt", "srt", "vtt", "json"]
        elif "TXT" in choice:
            formats_to_save = ["txt"]
        elif "SRT" in choice:
            formats_to_save = ["srt"]
        elif "JSON" in choice:
            formats_to_save = ["json"]
        elif "VTT" in choice:
            formats_to_save = ["vtt"]
        elif "Selección manual" in choice:
            cb_result = questionary.checkbox(
                "Marca los formatos deseados [Usa BARRA ESPACIADORA para marcar/desmarcar, ENTER para confirmar]:",
                choices=[
                    questionary.Choice("txt (Texto plano)", value="txt", checked=("txt" not in saved_formats)),
                    questionary.Choice("srt (Subtítulos)", value="srt", checked=("srt" not in saved_formats)),
                    questionary.Choice("vtt (WebVTT)", value="vtt", checked=("vtt" not in saved_formats)),
                    questionary.Choice("json (Datos JSON)", value="json", checked=("json" not in saved_formats)),
                ],
            ).ask()
            if cb_result:
                formats_to_save = cb_result

        for fmt in formats_to_save:
            saved_file = save_transcription(
                segments=segments,
                output_dir=config.output_dir,
                base_filename=base_name,
                format_type=fmt,
                include_timestamps=default_timestamps,
            )
            saved_formats.add(fmt)
            console.print(f"[green]✔ Guardado ({fmt.upper()}):[/green] [underline]{saved_file}[/underline]")

        # Si ya guardó en todos los formatos, terminar
        if "TODOS los formatos" in choice or len(saved_formats) >= 4:
            break

        # Preguntar si desea guardar en otro formato adicional
        more = questionary.confirm("¿Deseas guardar también en algún otro formato?", default=True).ask()
        if not more:
            break


def interactive_realtime_flow(engine: SpeechEngine, config: AppConfig) -> None:
    # Preguntar ajustes de sesión o usar defaults
    lang = questionary.select(
        "Idioma para la sesión en tiempo real:",
        choices=[
            questionary.Choice(title=f"{label}", value=code)
            for code, label in AVAILABLE_LANGUAGES.items()
        ],
        default=config.default_language,
    ).ask()
    if lang is None:
        return

    model = questionary.select(
        "Modelo para tiempo real:",
        choices=[
            questionary.Choice(title=f"{m} ({MODEL_DESCRIPTIONS.get(m, '')})", value=m)
            for m in ["base", "small", "large-v3-turbo", "tiny"]
        ],
        default=config.model_realtime if config.model_realtime in ["base", "small", "large-v3-turbo", "tiny"] else "small",
    ).ask()
    if model is None:
        return

    use_ts = questionary.confirm("¿Incluir marcas de tiempo [mm:ss]?", default=config.show_timestamps).ask()
    if use_ts is None:
        return

    segments = run_realtime_transcription(
        engine=engine,
        config=config,
        console=console,
        model_name=model,
        language=lang,
        show_timestamps=use_ts,
        device_index=config.mic_device_index,
    )

    if segments:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_save_segments(
            segments=segments,
            config=config,
            base_name=f"grabacion_mic_{now_str}",
            default_timestamps=use_ts,
        )


def interactive_file_flow(engine: SpeechEngine, config: AppConfig) -> None:
    file_path_str = questionary.text(
        "Ruta del archivo de audio o video a transcribir (e.g. archivo.mp3, audio.wav):"
    ).ask()

    if not file_path_str:
        return

    # Limpiar posibles comillas de arrastrar archivo a la terminal
    clean_path = file_path_str.strip().strip("'\"")
    path = Path(clean_path).expanduser().resolve()

    if not path.is_file():
        console.print(f"[bold red]❌ El archivo no existe:[/bold red] {path}")
        return

    lang = questionary.select(
        "Idioma del archivo:",
        choices=[
            questionary.Choice(title=label, value=code)
            for code, label in AVAILABLE_LANGUAGES.items()
        ],
        default=config.default_language,
    ).ask()
    if lang is None:
        return

    model = questionary.select(
        "Modelo a utilizar:",
        choices=[
            questionary.Choice(title=f"{m} ({MODEL_DESCRIPTIONS.get(m, '')})", value=m)
            for m in AVAILABLE_MODELS
        ],
        default=config.model_file,
    ).ask()
    if model is None:
        return

    use_ts = questionary.confirm("¿Mostrar con marcas de tiempo?", default=config.show_timestamps).ask()
    if use_ts is None:
        return

    segments = process_audio_file(
        file_path=path,
        engine=engine,
        config=config,
        console=console,
        model_name=model,
        language=lang,
        show_timestamps=use_ts,
    )

    if segments:
        prompt_save_segments(
            segments=segments,
            config=config,
            base_name=f"{path.stem}_transcripcion",
            default_timestamps=use_ts,
        )


def interactive_reexport_flow(config: AppConfig) -> None:
    cached = load_cached_session()
    if not cached or not cached.get("segments"):
        console.print("[yellow]⚠️  No hay ninguna transcripción en la memoria temporal reciente.[/yellow]")
        return

    base_name = cached["base_filename"]
    segments = cached["segments"]
    metadata = cached.get("metadata", {})
    text_preview = cached.get("full_text", "")[:120] + "..." if len(cached.get("full_text", "")) > 120 else cached.get("full_text", "")

    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="cyan bold")
    info_table.add_column()
    info_table.add_row("Archivo base:", base_name)
    info_table.add_row("Segmentos en memoria:", str(len(segments)))
    if "cached_at" in metadata:
        info_table.add_row("Fecha / Hora:", metadata["cached_at"])
    info_table.add_row("Vista previa:", f"[dim]{text_preview}[/dim]")

    console.print(
        Panel(
            info_table,
            title="[bold green]💾 Última Transcripción en Memoria[/bold green]",
            border_style="green",
        )
    )

    prompt_save_segments(
        segments=segments,
        config=config,
        base_name=base_name,
        default_timestamps=config.show_timestamps,
    )


def interactive_corrections_menu(config: AppConfig) -> None:
    mgr = CorrectionsManager()

    while True:
        choice = questionary.select(
            "✏️  Glosario y Reglas de Corrección de Usuario:",
            choices=[
                "1. Ver reglas actuales y vocabulario guía",
                "2. Agregar nuevo reemplazo fonético (ej. 'Swayna' ➔ 'Sueños del Bosque')",
                "3. Agregar palabra o nombre propio al vocabulario guía",
                "4. Aplicar correcciones a una transcripción existente (.txt, .srt, .json o última sesión)",
                "Volver al Menú Principal",
            ],
        ).ask()

        if not choice or "Volver" in choice:
            break

        if "1. Ver reglas actuales" in choice:
            mgr.load()
            table = Table(title=f"Glosario y Reglas ({mgr.file_path.name})", border_style="cyan")
            table.add_column("Tipo", style="cyan bold")
            table.add_column("Detalle", style="white")

            vocab_str = ", ".join(mgr.vocabulario_guia) if mgr.vocabulario_guia else "(vacío)"
            table.add_row("Vocabulario Guía (Whisper)", vocab_str)

            if mgr.reemplazos:
                reemplazos_lines = [f"'{err}' ➔ '{corr}'" for err, corr in mgr.reemplazos.items()]
                table.add_row("Reemplazos Fonéticos", "\n".join(reemplazos_lines))
            else:
                table.add_row("Reemplazos Fonéticos", "(sin reemplazos)")

            table.add_row("Ruta del archivo", str(mgr.file_path))
            table.add_row("Seguridad Git", "Protegido por .gitignore (nunca se sube al repo público)")

            console.print(table)
            console.print()

        elif "2. Agregar nuevo reemplazo" in choice:
            err = questionary.text("Palabra o frase incorrecta (como la escuchó Whisper):").ask()
            if not err:
                continue
            corr = questionary.text(f"Palabra o frase correcta para sustituir a '{err}':").ask()
            if not corr:
                continue
            mgr.add_replacement(err, corr)
            console.print(f"[bold green]✔ Regla guardada:[/bold green] '{err}' ➔ '{corr}'")

        elif "3. Agregar palabra o nombre propio" in choice:
            term = questionary.text("Nuevo nombre propio, topónimo o término clave para guiar a Whisper:").ask()
            if term:
                mgr.add_vocab_term(term)
                console.print(f"[bold green]✔ Término agregado al vocabulario guía:[/bold green] '{term}'")

        elif "4. Aplicar correcciones a una transcripción existente" in choice:
            sub = questionary.select(
                "¿A qué transcripción deseas aplicar las correcciones?",
                choices=[
                    "1. A la última transcripción en memoria",
                    "2. Seleccionar un archivo existente (.txt, .srt, .vtt, .json)",
                    "Cancelar",
                ],
            ).ask()

            if not sub or "Cancelar" in sub:
                continue

            if "1. A la última transcripción" in sub:
                cached = load_cached_session()
                if not cached or not cached.get("segments"):
                    console.print("[yellow]⚠️  No hay ninguna transcripción en la memoria temporal reciente.[/yellow]")
                    continue
                segments = cached["segments"]
                corregidos, cambios = mgr.correct_segments(segments)
                console.print(f"[bold green]✔ Se aplicaron {cambios} correcciones sobre {len(segments)} segmentos.[/bold green]")
                prompt_save_segments(
                    segments=corregidos,
                    config=config,
                    base_name=f"{cached['base_filename']}_corregido",
                    default_timestamps=config.show_timestamps,
                )

            elif "2. Seleccionar un archivo" in sub:
                f_path = questionary.text("Ruta del archivo (.txt, .srt, .vtt o .json) a corregir:").ask()
                if not f_path:
                    continue
                p = Path(f_path.strip().strip("'\"")).expanduser().resolve()
                if not p.is_file():
                    console.print(f"[red]❌ Archivo no encontrado: {p}[/red]")
                    continue
                content = p.read_text(encoding="utf-8")
                corregido, cambios = mgr.correct_text(content)
                if cambios == 0:
                    console.print("[dim]No se encontraron términos para corregir según el glosario actual.[/dim]")
                    continue
                out_p = p.with_name(f"{p.stem}_corregido{p.suffix}")
                out_p.write_text(corregido, encoding="utf-8")
                console.print(f"[bold green]✔ Se realizaron {cambios} cambios.[/bold green] Guardado en: [underline]{out_p}[/underline]")


def interactive_main_loop() -> None:
    config = AppConfig.load()
    engine = SpeechEngine(config)

    render_banner()

    while True:
        action = questionary.select(
            "¿Qué deseas hacer?",
            choices=[
                "🌐 1. Abrir Interfaz Gráfica Web (Navegador)",
                "🎙️  2. Transcribir en Tiempo Real (Micrófono)",
                "📁 3. Transcribir Archivo Grabado (Audio / Video)",
                "💾 4. Re-exportar / Convertir Última Transcripción a otros formatos",
                "✏️  5. Glosario y Correcciones de Usuario (Nombres propios, Lugares)",
                "⚙️  6. Configuración y Preferencias",
                "📊 7. Ver Estado del Hardware y Sistema",
                "🚪 8. Salir",
            ],
        ).ask()

        if not action or "8. Salir" in action:
            console.print("[dim cyan]Hasta luego.[/dim cyan]")
            break

        if "1. Abrir Interfaz Gráfica" in action:
            from vozatexto.web import start_server
            console.print("[green]Iniciando Interfaz Gráfica Web...[/green]")
            start_server()
            break
        elif "2. Transcribir en Tiempo Real" in action:
            interactive_realtime_flow(engine, config)
        elif "3. Transcribir Archivo Grabado" in action:
            interactive_file_flow(engine, config)
        elif "4. Re-exportar" in action:
            interactive_reexport_flow(config)
        elif "5. Glosario y Correcciones" in action:
            interactive_corrections_menu(config)
        elif "6. Configuración y Preferencias" in action:
            interactive_settings_menu(config)
        elif "7. Ver Estado" in action:
            show_system_status(config)

        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VozATexto: Transcripción CLI de voz a texto interactiva con faster-whisper y Silero VAD."
    )
    parser.add_argument("file", nargs="?", help="Ruta directa de un archivo de audio o video para transcribir")
    parser.add_argument("-r", "--realtime", action="store_true", help="Iniciar inmediatamente en modo micrófono en tiempo real")
    parser.add_argument("-m", "--model", choices=AVAILABLE_MODELS, help="Modelo de Whisper a usar (e.g. large-v3-turbo, small, base)")
    parser.add_argument("-l", "--lang", choices=["es", "en", "auto"], help="Idioma (es, en, auto)")
    parser.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=None, help="Activar o desactivar marcas de tiempo")
    parser.add_argument("-o", "--output-dir", help="Directorio donde guardar los resultados")
    parser.add_argument("--format", nargs="+", choices=["txt", "srt", "vtt", "json"], help="Formatos de guardado automático")
    parser.add_argument("--mic", type=int, help="Índice del dispositivo de micrófono a usar")
    parser.add_argument("--ui", "--web", action="store_true", help="Iniciar la interfaz gráfica web en el navegador")
    parser.add_argument("--status", action="store_true", help="Mostrar diagnóstico de hardware y configuración")

    args = parser.parse_args()
    config = AppConfig.load()

    if args.ui:
        from vozatexto.web import start_server
        start_server()
        return

    if args.output_dir:
        config.output_dir = str(Path(args.output_dir).resolve())

    if args.mic is not None:
        config.mic_device_index = args.mic

    engine = SpeechEngine(config)

    if args.status:
        render_banner()
        show_system_status(config)
        return

    # Si se pasó un archivo por argumento CLI
    if args.file:
        render_banner()
        fmt_list = args.format or ["txt"]
        process_audio_file(
            file_path=args.file,
            engine=engine,
            config=config,
            console=console,
            model_name=args.model,
            language=args.lang,
            show_timestamps=args.timestamps,
            save_formats=fmt_list,
        )
        return

    # Si se pasó la bandera de tiempo real directo
    if args.realtime:
        render_banner()
        fmt_list = args.format or ["txt"]
        run_realtime_transcription(
            engine=engine,
            config=config,
            console=console,
            model_name=args.model,
            language=args.lang,
            show_timestamps=args.timestamps,
            device_index=config.mic_device_index,
            save_formats=fmt_list,
        )
        return

    # Modo interactivo completo por defecto
    try:
        interactive_main_loop()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Operación cancelada. Saliendo...[/dim]")


if __name__ == "__main__":
    main()
