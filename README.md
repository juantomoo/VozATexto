# 🎙️ VozATexto — Transcriptor CLI de Voz a Texto

Aplicación de terminal (CLI) interactiva en Python para la transcripción de voz a texto tanto **en tiempo real (micrófono)** como **por archivos grabados de audio o video**, con soporte para **Español**, **Inglés** y **detección automática**, y control total sobre las **marcas de tiempo (timestamps)**.

Construida bajo los lineamientos de máxima eficiencia y calidad para hardware local (optimizado con cuantización `int8` y multithreading para CPU AMD Ryzen / x86_64, con soporte nativo de CUDA si se dispone de GPU dedicada).

---

## ✨ Características Principales

- **Motor `faster-whisper` (CTranslate2)**: Inferencia hasta 4x más rápida que Whisper vanilla con una fracción del consumo de memoria RAM/VRAM.
- **Filtro Anti-Alucinaciones**:
  - **Silero VAD integrado**: Descarta de forma preventiva silencios y ruidos no vocales, erradicando la causa #1 de alucinaciones.
  - **`condition_on_previous_text=False`**: Evita bucles repetitivos donde una palabra alucinada contamina los siguientes segmentos.
- **Modos de Operación**:
  1. **🎙️ Tiempo Real (Micrófono)**: Habla en vivo; la aplicación segmenta dinámicamente las pausas de voz y emite la transcripción fluida en tu consola.
  2. **📁 Archivos Grabados**: Procesa cualquier formato de audio o video (`mp3`, `wav`, `m4a`, `ogg`, `flac`, `mp4`, `mkv`, etc.).
- **Marcas de Tiempo (Timestamps)**:
  - Formato en pantalla: `[00:01.200 --> 00:04.500] Texto transcrito...`
  - Alternador sencillo para visualización con o sin marcas de tiempo.
- **Exportación Multi-formato**:
  - **TXT**: Texto corrido o con marcas de tiempo.
  - **SRT**: Subtítulos estándar listos para reproductores de video (VLC, YouTube, etc.).
  - **VTT**: Subtítulos WebVTT para la web.
  - **JSON**: Metadatos completos, duración, probabilidades y segmentos estructurados.
- **TUI Interactiva & CLI Unificada**: Menú visual con navegación por teclado (`questionary` + `rich`) y soporte directo para argumentos de línea de comandos.

---

## 🚀 Inicio Rápido

### 1. Ejecutar el Menú Interactivo
Desde la carpeta del proyecto, simplemente ejecuta:
```bash
./run.sh
```
o alternativamente con `uv`:
```bash
uv run vozatexto
```

Navega con las flechas del teclado `↑` / `↓` y presiona `Enter` para seleccionar.

---

## ✏️ Glosario y Reglas de Corrección de Usuario (`correcciones.json`)

Para preservar la **rigurosidad acústica** y a la vez permitir la corrección de **nombres propios, topónimos y términos técnicos** (ej. *Elise*, *Barichara*, *Sueños del Bosque*):

1. **Protección en Git:** El archivo `correcciones.json` está explícitamente excluido en `.gitignore` para garantizar que tus nombres personales y datos confidenciales nunca se suban a un repositorio público (se incluye únicamente `correcciones.example.json` como plantilla).
2. **Doble Mecanismo Inteligente:**
   - **Vocabulario Guía (`initial_prompt`):** Se le pasan los términos clave a Whisper antes de procesar el audio, condicionándolo a preferir la ortografía correcta desde el decodificador acústico.
   - **Reemplazos Fonéticos:** Reglas deterministas que corrigen cualquier interpretación fonética errónea (ej. `"Swayna Stolboski"` ➔ `"Sueños del Bosque"`, `"Columbia"` ➔ `"Colombia"`, `"Elisa"` ➔ `"Elise"`).
3. **Gestión desde el Menú:**
   Puedes gestionar tus reglas directamente seleccionando en el menú:
   `✏️  4. Glosario y Correcciones de Usuario (Nombres propios, Lugares)`
   o aplicando las correcciones a cualquier transcripción existente (.txt, .srt, .vtt, .json).

---

## ⌨️ Uso Directo por Línea de Comandos (CLI)

Puedes saltarte el menú interactivo y usar comandos directos:

### Transcribir un archivo grabado:
```bash
./run.sh audio.mp3
```
Con opciones personalizadas:
```bash
# Con modelo turbo, en español y exportando a SRT y TXT
./run.sh grabacion.m4a -m large-v3-turbo -l es --format srt txt

# Sin marcas de tiempo en el texto de salida
./run.sh entrevista.wav --no-timestamps
```

### Iniciar directamente en tiempo real (micrófono):
```bash
# Iniciar grabación en vivo con modelo rápido (base) y guardado automático
./run.sh --realtime -m base -l es --format txt srt
```

### Ver diagnóstico del hardware y audio:
```bash
./run.sh --status
```

---

## 🧠 Modelos Disponibles y Rendimiento

| Modelo | Recomendado para | Velocidad en CPU | Precisión |
|---|---|---|---|
| `base` | Tiempo real / Micrófono | Ultra rápida | Buena |
| `small` | **Tiempo real predeterminado** | Rápida y fluida | Muy buena |
| `large-v3-turbo` | **Archivos predeterminado** | Rápida (optimizada) | Excelente |
| `large-v3` | Máxima precisión para archivos | Moderada | Máxima |
| `medium` | Archivos multilingües equilibrados | Media | Alta |
| `tiny` | Pruebas de concepto ultraligeras | Instantánea | Básica |

*Nota: Los modelos se descargan automáticamente de Hugging Face en la primera ejecución y se almacenan en la caché local (`~/.cache/huggingface/hub`).*

---

## 📂 Directorio de Salida
Todas las transcripciones guardadas se almacenan por defecto en:
```
/home/juan/Datos/Datos Juan/ProyectosSoftware/VozATexto/transcripciones/
```
(Puedes personalizar esta ruta en cualquier momento desde el menú de *Configuración y Preferencias*).
