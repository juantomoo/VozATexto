#!/usr/bin/env bash
# Lanzador directo de la Interfaz Gráfica Web de VozATexto
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
UV_BIN="${HOME}/.local/bin/uv"
if [ ! -x "$UV_BIN" ]; then
    UV_BIN="$(which uv || true)"
fi

exec "$UV_BIN" run python -m vozatexto.web "$@"
