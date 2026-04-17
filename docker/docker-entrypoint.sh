#!/bin/bash
# =============================================================
# docker-entrypoint.sh
#
# O CUDA_VISIBLE_DEVICES é definido pelo docker-compose.yml
# via variável GPU_IDS na subida do container.
# =============================================================
set -e

# Gera CUDA_DEVICE_IDS automaticamente a partir do GPU_IDS
# GPU_IDS=3      → CUDA_VISIBLE_DEVICES=0
# GPU_IDS=1,2,3  → CUDA_VISIBLE_DEVICES=0,1,2
GPU_COUNT=$(echo "${GPU_IDS:-0}" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s',' 0 $((GPU_COUNT - 1)))

echo "[entrypoint] GPU_IDS=${GPU_IDS:-0}"
echo "[entrypoint] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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