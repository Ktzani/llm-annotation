#!/bin/bash
# =============================================================
# docker-entrypoint.sh
#
# Comportamento controlado pela variavel ENABLE_ANNOTATION:
#   ENABLE_ANNOTATION=1  → sobe Ollama + pull + preload dos modelos,
#                          depois sobe a API. Modo anotacao.
#   ENABLE_ANNOTATION=0  → pula tudo do Ollama e sobe so a API.
#                          Modo fine-tuning (GPU 100% livre).
#
# CUDA_VISIBLE_DEVICES é definido pelo docker-compose.yml via
# variável GPU_IDS na subida do container.
# =============================================================
set -e

ENABLE_ANNOTATION="${ENABLE_ANNOTATION:-0}"

# ---- Modelos puxados na primeira subida ---------------------
OLLAMA_MODELS_TO_PULL=(
    "qwen3:8b"
    "llama3.1:8b"
    "deepseek-r1:8b"
)

if [ "$ENABLE_ANNOTATION" = "1" ] || [ "$ENABLE_ANNOTATION" = "true" ]; then
    echo "[entrypoint] ENABLE_ANNOTATION=1 → subindo Ollama (modo anotacao)"

    # ---- Sobe o Ollama em background ------------------------
    export OLLAMA_HOST="0.0.0.0:11434"
    export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-5}"
    export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
    export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
    export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"
    export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

    echo "[entrypoint] Iniciando ollama serve em background..."
    # Prefixa toda saida do ollama com "[ollama]" para nao misturar
    # com os logs do uvicorn no `docker logs`. Process substitution
    # garante que $! seja o PID do ollama, nao do sed.
    # O grep filtra os access logs "[GIN] ... POST ..." (uma linha
    # por requisicao) mas preserva startup, load de modelos e erros.
    ollama serve \
        > >(grep --line-buffered -v '\[GIN\]' | sed -u 's/^/[ollama] /') \
        2> >(grep --line-buffered -v '\[GIN\]' | sed -u 's/^/[ollama] /' >&2) &
    OLLAMA_PID=$!

    # Se a API morrer, derruba o ollama junto (evita orfao)
    trap 'kill -TERM $OLLAMA_PID 2>/dev/null || true' EXIT

    # ---- Espera o ollama ficar pronto -----------------------
    echo "[entrypoint] Aguardando ollama responder em :11434..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "[entrypoint] Ollama pronto."
            break
        fi
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            echo "[entrypoint] ERRO: ollama serve morreu antes de subir." >&2
            exit 1
        fi
        sleep 1
    done

    # ---- Pull dos modelos (idempotente) ---------------------
    # Baixa os pesos para o volume /root/.ollama. Se ja estiverem
    # la, o pull retorna rapido.
    for model in "${OLLAMA_MODELS_TO_PULL[@]}"; do
        echo "[entrypoint] ollama pull $model"
        ollama pull "$model"
    done

    # ---- Pre-carrega os modelos na VRAM ---------------------
    # POST em /api/generate sem prompt faz o load do modelo em
    # memoria. Com OLLAMA_KEEP_ALIVE=24h, eles ficam residentes.
    for model in "${OLLAMA_MODELS_TO_PULL[@]}"; do
        echo "[entrypoint] preload $model na VRAM"
        curl -s http://localhost:11434/api/generate \
            -d "{\"model\": \"$model\"}" > /dev/null
    done
else
    echo "[entrypoint] ENABLE_ANNOTATION=0 → Ollama NAO sera iniciado (modo fine-tuning, GPU 100% livre)"
fi

# ---- Permite passar comandos alternativos (bash, pytest...) -
if [ "$1" = "bash" ] || [ "$1" = "sh" ] || [ "$1" = "pytest" ]; then
    exec "$@"
fi

# ---- Sobe a API ---------------------------------------------
echo "[entrypoint] Iniciando uvicorn na porta 8000..."
exec python -m uvicorn src.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir src \
    --log-level info
