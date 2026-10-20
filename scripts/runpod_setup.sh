#!/bin/bash
# =============================================================
# scripts/runpod_setup.sh — prepara e sobe o llm-annotation num pod RunPod
#
# O pod RunPod é um container sem privilégios: Docker NÃO roda lá dentro,
# então tudo roda nativo (pyenv + .venv + Ollama + uvicorn no tmux).
#
# Idempotente: refaz TODOS os passos do setup original em qualquer situação
# (pod novo, pod reiniciado ou já configurado). O que persiste fica em
# /workspace (volume); o que está fora dele (apt, ~/.bashrc, Ollama, Poetry,
# ~/.gitconfig, processos) é reinstalado a cada execução.
#
# Uso (como root, no pod):
#   bash /workspace/runpod_setup.sh                  # modo anotação (padrão)
#   MODO=finetuning bash /workspace/runpod_setup.sh  # API sem o Ollama na GPU
#   GPU=0  OLLAMA_NUM_THREAD=8  (opcionais)
#
# Guarde uma cópia em /workspace/runpod_setup.sh: num volume vazio o
# repositório (e este script) ainda não existe.
#
# Depois: exponha a porta 8000 em "Expose HTTP Ports" no painel do RunPod.
# =============================================================
set -e

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/llm-annotation
REPO_URL=https://github.com/Ktzani/llm-annotation.git
PYTHON_VERSION=3.11.9
POETRY_VERSION=2.2.1
MODELS=(qwen3:8b llama3.1:8b deepseek-r1:8b)
MODO="${MODO:-anotacao}"
GPU="${GPU:-0}"

step() { echo; echo "===== $* ====="; }

# Cota real de CPU do container (cgroup v2/v1); nproc enxerga todos os núcleos do host
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
CPUS=$(available_cpus)

# -------------------------------------------------------------
step "1. Pacotes do sistema (build do Python, tmux, Ollama, pipx)"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    curl git tmux pciutils zstd pipx

# -------------------------------------------------------------
step "2. Volume /workspace aceita chmod e symlink (exigido pelo git e pelo .venv)"
cd "$WORKSPACE"
if touch .t && chmod +x .t && ln -sf .t .l; then
    echo "volume OK"
    rm -f .t .l
else
    rm -f .t .l
    echo "ERRO: o volume não aceita chmod/symlink — troque o volume do pod" >&2
    exit 1
fi

# -------------------------------------------------------------
step "3. env.sh (variáveis do ambiente, no volume) + ~/.bashrc"
cat > "$WORKSPACE/env.sh" << 'EOF'
# Gerado por scripts/runpod_setup.sh
export PYENV_ROOT=/workspace/.pyenv
export PATH="$PYENV_ROOT/bin:/root/.local/bin:$PATH"
command -v pyenv > /dev/null && eval "$(pyenv init - bash)"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/workspace/llm-annotation/data/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
# Ollama: mesmos valores do docker-compose (NUM_PARALLEL >= max_concurrent_texts)
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_NUM_PARALLEL=15
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_CONTEXT_LENGTH=12288
export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_MODELS=/workspace/ollama-models
# Coluna run_env do checkpoint de anotação
export ANNOTATION_RUN_ENV=runpod
EOF
grep -q 'source /workspace/env.sh' ~/.bashrc 2>/dev/null || echo 'source /workspace/env.sh' >> ~/.bashrc

# -------------------------------------------------------------
step "4. pyenv + Python $PYTHON_VERSION (no volume)"
[ -d "$WORKSPACE/.pyenv" ] || git clone https://github.com/pyenv/pyenv.git "$WORKSPACE/.pyenv"
source "$WORKSPACE/env.sh"
pyenv install -s "$PYTHON_VERSION"

# -------------------------------------------------------------
step "5. Repositório (clone, safe.directory, .git, .python-version, .env)"
git config --global --get-all safe.directory 2>/dev/null | grep -qx "$REPO_DIR" \
    || git config --global --add safe.directory "$REPO_DIR"
[ -e "$REPO_DIR" ] || git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

# Clone antigo com --separate-git-dir: o .git é um arquivo apontando para a home (perdida)
if [ -f .git ]; then
    echo ".git apontava para fora do volume — reconectando ao GitHub (código e dados intactos)"
    mv .git .git.quebrado
    git init -q && git symbolic-ref HEAD refs/heads/main
    git remote add origin "$REPO_URL"
    git fetch origin main && git reset -q origin/main && git branch -u origin/main
fi
git status -sb | head -5

pyenv local "$PYTHON_VERSION"
if [ ! -f .env ]; then
    cp .env-example .env
    echo "AVISO: .env criado a partir do .env-example — preencha os tokens"
fi
mkdir -p data/results data/.cache

# -------------------------------------------------------------
step "6. Poetry $POETRY_VERSION + .venv no repositório"
if ! command -v poetry > /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 - --version "$POETRY_VERSION" \
        || pipx install "poetry==$POETRY_VERSION"
fi
poetry config virtualenvs.in-project true --local
# .venv quebrado (Python de origem sumiu) é recriado
.venv/bin/python --version > /dev/null 2>&1 || rm -rf .venv
poetry env use "$(pyenv which python)"
poetry install
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available())" || true

# -------------------------------------------------------------
step "7. Ollama + modelos (com num_thread limitado à cota de CPU)"
command -v ollama > /dev/null || curl -fsSL https://ollama.com/install.sh | sh
tmux kill-session -t ollama 2>/dev/null || true
tmux new-session -d -s ollama "source $WORKSPACE/env.sh; CUDA_VISIBLE_DEVICES=$GPU ollama serve; exec bash"
for _ in $(seq 1 60); do curl -sf localhost:11434/api/tags > /dev/null && break; sleep 1; done

# Sem limite, cada llama-server cria threads para todos os núcleos do host e estoura a cota
NUM_THREAD="${OLLAMA_NUM_THREAD:-$(( (CPUS - 2) / ${#MODELS[@]} ))}"
[ "$NUM_THREAD" -lt 1 ] && NUM_THREAD=1
echo "num_thread=$NUM_THREAD por modelo ($CPUS vCPUs / ${#MODELS[@]} modelos)"
for m in "${MODELS[@]}"; do
    ollama show "$m" > /dev/null 2>&1 || ollama pull "$m"
    ollama show "$m" --modelfile | grep -v '^PARAMETER num_thread' > /tmp/Modelfile
    echo "PARAMETER num_thread $NUM_THREAD" >> /tmp/Modelfile
    ollama create "$m" -f /tmp/Modelfile > /dev/null
done

if [ "$MODO" = "finetuning" ]; then
    echo "MODO=finetuning → Ollama derrubado (GPU livre para os folds)"
    tmux kill-session -t ollama 2>/dev/null || true
else
    for m in "${MODELS[@]}"; do
        curl -s localhost:11434/api/generate -d "{\"model\":\"$m\"}" > /dev/null
    done
    ollama ps
fi

# -------------------------------------------------------------
step "8. API (uvicorn no tmux 'api')"
# Threads do PyTorch por fold (5 folds em paralelo dividem a cota)
OMP_THREADS=$(( CPUS / 5 ))
[ "$OMP_THREADS" -lt 1 ] && OMP_THREADS=1
tmux kill-session -t api 2>/dev/null || true
tmux new-session -d -s api "cd $REPO_DIR && source $WORKSPACE/env.sh && OMP_NUM_THREADS=$OMP_THREADS TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload --reload-dir src; exec bash"
for _ in $(seq 1 120); do curl -sf localhost:8000/health > /dev/null && break; sleep 1; done
curl -s localhost:8000/health; echo

# -------------------------------------------------------------
step "Pronto"
echo "Logs:      tmux attach -t api   |   tmux attach -t ollama   (sair: Ctrl+B depois D)"
echo "Lembrete:  exponha a porta 8000 em 'Expose HTTP Ports' no painel do RunPod"
echo "Fine-tuning com o Ollama de pé? derrube antes: tmux kill-session -t ollama"
