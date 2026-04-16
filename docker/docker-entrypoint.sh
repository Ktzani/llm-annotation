#!/bin/bash
# =============================================================
# docker-entrypoint.sh
#
# Lê a variável GPU_ID (padrão: 0) e a repassa como
# CUDA_VISIBLE_DEVICES para que apenas aquela GPU fique
# visível para o processo Python.
#
# Uso:
#   docker run -e GPU_ID=2 ...     → usa a GPU 2
#   docker run -e GPU_ID=0,1 ...   → usa as GPUs 0 e 1
# =============================================================
set -e

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
echo "[entrypoint] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Permite passar comandos alternativos (ex.: bash, pytest)
if [ "$1" = "bash" ] || [ "$1" = "sh" ] || [ "$1" = "pytest" ]; then
    exec "$@"
fi

exec python -m uvicorn src.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --reload \
    --reload-dir src
