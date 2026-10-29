"""
Configuración global, detección de hardware y preferencias de VozATexto.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

try:
    import ctranslate2
    _CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0
except Exception:
    _CUDA_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = APP_DIR / "transcripciones"
USER_CONFIG_PATH = Path.home() / ".config" / "vozatexto" / "settings.json"
LAST_TRANSCRIPT_CACHE_PATH = Path.home() / ".config" / "vozatexto" / "ultima_transcripcion.json"

AVAILABLE_MODELS = [
    "large-v3-turbo",
    "small",
    "base",
    "medium",
    "large-v3",
    "tiny",
]

MODEL_DESCRIPTIONS = {
    "large-v3-turbo": "Recomendado para archivos: alta precisión, optimizado y veloz (int8)",
    "small": "Recomendado para tiempo real en CPU: balance óptimo de velocidad y precisión",
    "base": "Ultra rápido para tiempo real en CPU: latencia mínima",
    "medium": "Muy preciso pero más exigente en CPU",
    "large-v3": "Máxima precisión absoluta (más lento en CPU)",
    "tiny": "Muy ligero, útil para pruebas de concepto",
}

AVAILABLE_LANGUAGES = {
    "es": "Español (es)",
    "en": "Inglés (en)",
    "auto": "Detección automática",
}


@dataclass
class AppConfig:
    device: Literal["cpu", "cuda"] = "cuda" if _CUDA_AVAILABLE else "cpu"
    compute_type: str = "float16" if _CUDA_AVAILABLE else "int8"
    cpu_threads: int = max(1, min(os.cpu_count() or 4, 6))
    default_language: str = "es"
    model_realtime: str = "small"
    model_file: str = "large-v3-turbo"
    show_timestamps: bool = True
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    vad_min_silence_ms: int = 500
    beam_size_realtime: int = 1
    beam_size_file: int = 3
    mic_device_index: int | None = None

    def save(self, path: Path = USER_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path = USER_CONFIG_PATH) -> AppConfig:
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Validar y combinar con defaults
            valid_keys = cls.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)
        except Exception:
            return cls()
