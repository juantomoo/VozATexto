"""
Motor de inferencia basado en faster-whisper (CTranslate2) con Silero VAD y filtros anti-alucinación.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from faster_whisper import WhisperModel

from vozatexto.config import AppConfig
from vozatexto.corrections import CorrectionsManager
from vozatexto.exporter import TranscriptSegment

# Suprimir logs ruidosos de faster_whisper / ctranslate2 si no son errores
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


class SpeechEngine:
    """Envoltorio optimizado para modelos faster-whisper."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.load()
        self.current_model_name: Optional[str] = None
        self._model: Optional[WhisperModel] = None

    def load_model(self, model_name: str) -> WhisperModel:
        if self._model is not None and self.current_model_name == model_name:
            return self._model

        # Inicialización del modelo con CTranslate2
        self._model = WhisperModel(
            model_size_or_path=model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads,
            download_root=str(Path.home() / ".cache" / "huggingface" / "hub"),
        )
        self.current_model_name = model_name
        return self._model

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
    ) -> Tuple[List[TranscriptSegment], Dict[str, Any]]:
        """
        Transcribe un archivo de audio o video.
        Retorna (lista_de_segmentos, info_de_transcripción).
        """
        chosen_model = model_name or self.config.model_file
        model = self.load_model(chosen_model)

        lang = language if language and language != "auto" else None
        b_size = beam_size if beam_size is not None else self.config.beam_size_file
        silence_ms = min_silence_duration_ms or self.config.vad_min_silence_ms

        vad_params = dict(
            min_silence_duration_ms=silence_ms,
            threshold=0.5,
        )

        corrections = CorrectionsManager()
        initial_prompt = corrections.get_initial_prompt()

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=b_size,
            vad_filter=True,
            vad_parameters=vad_params,
            condition_on_previous_text=False,  # Mitigación estricta de alucinaciones
            temperature=0.0,                   # Greedy decoding estable
            initial_prompt=initial_prompt,     # Vocabulario guía de nombres propios
        )

        raw_segments = []
        for s in segments_iter:
            text = s.text.strip()
            if text:
                raw_segments.append(TranscriptSegment(start=s.start, end=s.end, text=text))

        # Aplicar reglas fonéticas y ortográficas del usuario preservando timestamps
        segments, num_corrections = corrections.correct_segments(raw_segments)

        metadata = {
            "model": chosen_model,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "corrections_applied": num_corrections,
        }
        return segments, metadata

    def transcribe_audio_chunk(
        self,
        audio_data: np.ndarray,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
        offset_seconds: float = 0.0,
    ) -> List[TranscriptSegment]:
        """
        Transcribe un bloque en memoria (numpy array float32 a 16kHz).
        Ideal para tiempo real / streaming.
        """
        if len(audio_data) == 0:
            return []

        chosen_model = model_name or self.config.model_realtime
        model = self.load_model(chosen_model)

        lang = language if language and language != "auto" else None
        b_size = beam_size if beam_size is not None else self.config.beam_size_realtime

        vad_params = dict(
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            threshold=0.5,
        )

        corrections = CorrectionsManager()
        initial_prompt = corrections.get_initial_prompt()

        segments_iter, _ = model.transcribe(
            audio_data,
            language=lang,
            beam_size=b_size,
            vad_filter=True,
            vad_parameters=vad_params,
            condition_on_previous_text=False,
            temperature=0.0,
            initial_prompt=initial_prompt,
        )

        raw_segments = []
        for s in segments_iter:
            text = s.text.strip()
            if text:
                raw_segments.append(
                    TranscriptSegment(
                        start=offset_seconds + s.start,
                        end=offset_seconds + s.end,
                        text=text,
                    )
                )

        segments, _ = corrections.correct_segments(raw_segments)
        return segments
