#!/usr/bin/env bash
# Lanzador directo de la Interfaz Gráfica Web de VozATexto
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
exec uv run python -m vozatexto.web "$@"
