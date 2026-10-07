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
# Overridavel na subida: OLLAMA_ANNOTATION_MODELS="qwen3:8b llama3.1:8b deepseek-r1:8b"
# (nao usar OLLAMA_MODELS: e a variavel nativa do Ollama para o diretorio
# onde os pesos sao armazenados)
if [ -n "$OLLAMA_ANNOTATION_MODELS" ]; then
    # shellcheck disable=SC2206
    OLLAMA_MODELS_TO_PULL=($OLLAMA_ANNOTATION_MODELS)
else
    OLLAMA_MODELS_TO_PULL=(
        "qwen3:8b"
        "llama3.1:8b"
        "deepseek-r1:8b"
    )
fi

if [ "$ENABLE_ANNOTATION" = "1" ] || [ "$ENABLE_ANNOTATION" = "true" ]; then
    echo "[entrypoint] ENABLE_ANNOTATION=1 → subindo Ollama (modo anotacao)"

    # ---- Sobe o Ollama em background ------------------------
    export OLLAMA_HOST="0.0.0.0:11434"
    export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-15}"
    export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
    export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
    export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-12288}"
    export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

    echo "[entrypoint] Iniciando ollama serve em background..."
    # Toda a saida do ollama (startup, load de modelos, access logs
    # [GIN] e os logs verbosos do runner: slot/srv print_timing, prompt
    # cache, etc.) é mantida FORA do `docker logs` — que assim mostra so
    # o uvicorn (a API).
    #
    # Por padrao a saida é DESCARTADA (/dev/null) para nao acumular em
    # disco num run longo (o runner cospe varias linhas por requisicao).
    # Para depurar, suba com OLLAMA_LOG=/root/.ollama/ollama.log que ele
    # grava em arquivo (no volume persistente). Inspecionar com:
    #   docker exec <container> tail -f /root/.ollama/ollama.log
    OLLAMA_LOG="${OLLAMA_LOG:-/dev/null}"
    if [ "$OLLAMA_LOG" = "/dev/null" ]; then
        echo "[entrypoint] Saida do ollama descartada (defina OLLAMA_LOG=<arquivo> para depurar)"
    else
        echo "[entrypoint] Logs do ollama em $OLLAMA_LOG (fora do docker logs)"
    fi
    ollama serve > "$OLLAMA_LOG" 2>&1 &
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

    # ---- Limita as threads de CPU por modelo ----------------
    # Em container, o llama-server enxerga todos os nucleos do HOST
    # (ex.: 255) e cria threads demais para a cota de CPU do container.
    # A cota estoura, o kernel pausa as threads e a GPU fica ociosa
    # esperando a CPU (visto no RunPod: CPU 97%, GPU 29%, anotacao 6x
    # mais lenta). Com o modelo 100% na GPU, poucas threads bastam.
    # Divide a cota real (cgroup) entre os modelos, reservando 2 vCPUs
    # para a API. Grava PARAMETER num_thread no Modelfile (mesmo nome e
    # mesmos pesos; nao altera as respostas).
    # Override: OLLAMA_NUM_THREAD=<n>; OLLAMA_NUM_THREAD=0 desativa.
    available_cpus() {
        local quota="" period=""
        if [ -r /sys/fs/cgroup/cpu.max ]; then
            read -r quota period < /sys/fs/cgroup/cpu.max
        elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
            quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
            period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        fi
        if [ -n "$quota" ] && [ "$quota" != "max" ] && [ "$quota" -gt 0 ]; then
            echo $(( quota / period ))
        else
            nproc
        fi
    }

    NUM_MODELS=${#OLLAMA_MODELS_TO_PULL[@]}
    AUTO_THREADS=$(( ($(available_cpus) - 2) / NUM_MODELS ))
    [ "$AUTO_THREADS" -lt 1 ] && AUTO_THREADS=1
    NUM_THREAD="${OLLAMA_NUM_THREAD:-$AUTO_THREADS}"

    if [ "$NUM_THREAD" != "0" ]; then
        echo "[entrypoint] num_thread=$NUM_THREAD por modelo ($(available_cpus) vCPUs / $NUM_MODELS modelos)"
        for model in "${OLLAMA_MODELS_TO_PULL[@]}"; do
            ollama show "$model" --modelfile | grep -v '^PARAMETER num_thread' > /tmp/Modelfile
            echo "PARAMETER num_thread $NUM_THREAD" >> /tmp/Modelfile
            ollama create "$model" -f /tmp/Modelfile > /dev/null
        done
    else
        echo "[entrypoint] OLLAMA_NUM_THREAD=0 → num_thread padrao do Ollama"
    fi

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
