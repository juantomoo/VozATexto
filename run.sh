#!/usr/bin/env bash
# Lanzador de la aplicación CLI VozATexto con entorno uv
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Ejecutar aplicación mediante uv
exec uv run python -m vozatexto.cli "$@"
