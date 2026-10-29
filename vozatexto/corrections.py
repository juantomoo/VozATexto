"""
Gestión de correcciones de usuario, glosario de nombres propios y vocabulario guía.
Permite corregir transcripciones nuevas y existentes preservando la rigurosidad acústica.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vozatexto.config import APP_DIR, USER_CONFIG_PATH
from vozatexto.exporter import TranscriptSegment


def get_corrections_file_path() -> Path:
    """Busca el archivo correcciones.json local o en la configuración de usuario."""
    local_path = APP_DIR / "correcciones.json"
    if local_path.is_file():
        return local_path
    user_path = USER_CONFIG_PATH.parent / "correcciones.json"
    return user_path


class CorrectionsManager:
    """Carga y aplica reglas de corrección fonética y vocabulario guía."""

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or get_corrections_file_path()
        self.vocabulario_guia: List[str] = []
        self.reemplazos: Dict[str, str] = {}
        self.load()

    def load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vocabulario_guia = data.get("vocabulario_guia", [])
            self.reemplazos = data.get("reemplazos", {})
        except Exception:
            pass

    def save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "_comentario": "Glosario y reglas de corrección de VozATexto (protegido por .gitignore).",
            "vocabulario_guia": self.vocabulario_guia,
            "reemplazos": self.reemplazos,
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_initial_prompt(self) -> Optional[str]:
        """
        Genera un prompt guía para Whisper antes de la inferencia.
        Condiciona al modelo a preferir la ortografía correcta de nombres propios.
        """
        if not self.vocabulario_guia:
            return None
        return ", ".join(self.vocabulario_guia) + "."

    def correct_text(self, text: str) -> Tuple[str, int]:
        """
        Aplica los reemplazos configurados sobre un texto.
        Devuelve (texto_corregido, cantidad_de_cambios).
        """
        if not text or not self.reemplazos:
            return text, 0

        modificado = text
        total_cambios = 0

        # Ordenar reemplazos por longitud descendente para evitar colisiones en subfrases
        claves_ordenadas = sorted(self.reemplazos.items(), key=lambda item: len(item[0]), reverse=True)

        for err, corr in claves_ordenadas:
            if not err:
                continue
            # Si el término es una sola palabra alfanumérica, usar límites de palabra (\b)
            # para no sustituir partes internas de otras palabras (e.g. evitar 'Elisa' en 'Elisabeth')
            if re.match(r"^\w+$", err):
                pattern = re.compile(rf"\b{re.escape(err)}\b", re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(err), re.IGNORECASE)

            nuevos_cambios = len(pattern.findall(modificado))
            if nuevos_cambios > 0:
                total_cambios += nuevos_cambios
                modificado = pattern.sub(corr, modificado)

        return modificado, total_cambios

    def correct_segments(self, segments: List[TranscriptSegment]) -> Tuple[List[TranscriptSegment], int]:
        """Aplica correcciones a una lista de segmentos manteniendo intactas las marcas de tiempo."""
        corregidos = []
        total_cambios = 0
        for seg in segments:
            nuevo_texto, cambios = self.correct_text(seg.text)
            total_cambios += cambios
            corregidos.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=nuevo_texto,
                )
            )
        return corregidos, total_cambios

    def add_replacement(self, erroneo: str, correcto: str) -> None:
        self.reemplazos[erroneo.strip()] = correcto.strip()
        if correcto.strip() not in self.vocabulario_guia:
            self.vocabulario_guia.append(correcto.strip())
        self.save()

    def add_vocab_term(self, termino: str) -> None:
        term = termino.strip()
        if term and term not in self.vocabulario_guia:
            self.vocabulario_guia.append(term)
            self.save()
