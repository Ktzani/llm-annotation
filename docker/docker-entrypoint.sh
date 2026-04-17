#!/bin/bash
# =============================================================
# docker-entrypoint.sh
#
# O CUDA_VISIBLE_DEVICES é definido pelo docker-compose.yml
# via variável GPU_IDS na subida do container.
# =============================================================
set -e
 
# Permite passar comandos alternativos (ex.: bash, pytest)
if [ "$1" = "bash" ] || [ "$1" = "sh" ] || [ "$1" = "pytest" ]; then
    exec "$@"
fi
 
exec python -m uvicorn src.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir src \
    --log-level info 