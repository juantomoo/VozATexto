"""
Manejo de dispositivos de audio y captura de micrófono mediante sounddevice.
"""

from __future__ import annotations

import queue
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Frecuencia nativa esperada por Whisper


def list_input_devices() -> List[Dict[str, Any]]:
    """Devuelve la lista de dispositivos de captura de audio disponibles."""
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    input_devs = []

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            is_default = (idx == default_input)
            label = f"[{idx}] {dev['name']}"
            if is_default:
                label += " (Predeterminado)"
            input_devs.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
                "is_default": is_default,
                "label": label,
            })
    return input_devs


class AudioCaptureStream:
    """Captura continua de audio desde el micrófono a 16kHz mono float32."""

    def __init__(self, device_index: Optional[int] = None, block_duration_ms: int = 100):
        self.device_index = device_index
        self.samplerate = SAMPLE_RATE
        self.block_size = int(self.samplerate * (block_duration_ms / 1000.0))
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self.is_running = False
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            # Descarte de advertencias menores de buffer
            pass
        # indata shape: (frames, 1), convertimos a 1D float32
        self.queue.put(indata.flatten().copy())

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            device=self.device_index,
            blocksize=self.block_size,
            callback=self._callback,
        )
        self._stream.start()

    def read_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def compute_rms(audio_chunk: np.ndarray) -> float:
    """Calcula el valor RMS (Root Mean Square) para medir la energía/volumen."""
    if len(audio_chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_chunk ** 2)))
