#!/bin/bash
# =============================================================
# scripts/lab_setup.sh — prepara e sobe o llm-annotation numa máquina do
# laboratório (Linux com GPU NVIDIA, ex.: RTX 6000 Ada 48 GB), sem Docker
#
# Mesmo fluxo do scripts/runpod_setup.sh, adaptado para a máquina física:
#   - funciona SEM root: pyenv, Poetry e Ollama ficam em $BASE_DIR (sudo só
#     é usado, se disponível, para os pacotes do sistema)
#   - Ollama extraído em $BASE_DIR/ollama, sem serviço systemd; se já houver
#     um Ollama do sistema na porta 11434 (a que a aplicação usa), o script
#     para com instruções em vez de criar conflito
#   - VRAM de 48 GB: OLLAMA_NUM_PARALLEL=5 (= max_concurrent_classes da
#     anotação binária) com contexto 12288; ver o passo 7 se não couber
#
# Idempotente: refaz todos os passos; o que já existe é conferido e mantido.
#
# Uso:
#   bash lab_setup.sh                        # modo anotação (padrão)
#   MODO=finetuning bash lab_setup.sh        # API sem o Ollama na GPU
#   Opcionais: BASE_DIR=~/llm  BRANCH=experimentacao-kenia  GPU=0
#              OLLAMA_CONTEXT_LENGTH=12288  OLLAMA_NUM_PARALLEL=5  OLLAMA_NUM_THREAD=<n>
# =============================================================
set -e

BASE_DIR="${BASE_DIR:-$HOME/llm}"
REPO_DIR="$BASE_DIR/llm-annotation"
REPO_URL=https://github.com/Ktzani/llm-annotation.git
BRANCH="${BRANCH:-experimentacao-kenia}"
PYTHON_VERSION=3.11.9
POETRY_VERSION=2.2.1
MODELS=(qwen3:8b gemma4:12b deepseek-r1:8b)
NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-5}"
CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-12288}"
MODO="${MODO:-anotacao}"
GPU="${GPU:-0}"
OLLAMA_DIR="$BASE_DIR/ollama"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"

step() { echo; echo "===== $* ====="; }
warn() { echo "AVISO: $*" >&2; }
die()  { echo "ERRO: $*" >&2; exit 1; }

# Cota de CPU (cgroup, se houver) ou núcleos da máquina
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
step "0. Verificações: GPU e permissões"
command -v nvidia-smi > /dev/null || die "nvidia-smi não encontrado: driver NVIDIA ausente"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
if [ "$(id -u)" -eq 0 ]; then SUDO=""; HAS_ROOT=1
elif sudo -n true 2> /dev/null; then SUDO="sudo"; HAS_ROOT=1
else SUDO=""; HAS_ROOT=0; fi
echo "root/sudo: $([ $HAS_ROOT -eq 1 ] && echo sim || echo 'não (instalação só no usuário)') | CPUs: $CPUS | base: $BASE_DIR"

# -------------------------------------------------------------
step "1. Pacotes do sistema"
if [ $HAS_ROOT -eq 1 ]; then
    $SUDO apt-get update
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y \
        build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
        curl git tmux pciutils zstd pipx
else
    missing=""
    for cmd in git curl tmux zstd gcc make; do command -v $cmd > /dev/null || missing="$missing $cmd"; done
    [ -z "$missing" ] || die "faltam no sistema:$missing — peça ao administrador (apt-get install$missing) ou rode com sudo"
    warn "sem sudo: as bibliotecas de compilação do Python (libssl-dev, libffi-dev, ...) não podem ser conferidas; se o 'pyenv install' falhar, peça para instalá-las"
fi

# -------------------------------------------------------------
step "2. Pasta base $BASE_DIR (precisa aceitar chmod e symlink)"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"
if touch .t && chmod +x .t && ln -sf .t .l; then
    echo "pasta OK"; rm -f .t .l
else
    rm -f .t .l; die "$BASE_DIR não aceita chmod/symlink (disco de rede?) — use BASE_DIR em disco local"
fi

# -------------------------------------------------------------
step "3. env.sh + ~/.bashrc"
cat > "$BASE_DIR/env.sh" << EOF
# Gerado por scripts/lab_setup.sh
export PYENV_ROOT=$BASE_DIR/.pyenv
export PATH="\$PYENV_ROOT/bin:$OLLAMA_DIR/bin:\$HOME/.local/bin:\$PATH"
command -v pyenv > /dev/null && eval "\$(pyenv init - bash)"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=$REPO_DIR/data/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
# Ollama: NUM_PARALLEL >= max_concurrent_classes (binária) / max_concurrent_texts
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=$NUM_PARALLEL
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_CONTEXT_LENGTH=$CONTEXT_LENGTH
export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_MODELS=$BASE_DIR/ollama-models
# Coluna run_env do checkpoint de anotação
export ANNOTATION_RUN_ENV=lab
EOF
grep -q "source $BASE_DIR/env.sh" ~/.bashrc 2> /dev/null || echo "source $BASE_DIR/env.sh" >> ~/.bashrc

# -------------------------------------------------------------
step "4. pyenv + Python $PYTHON_VERSION"
[ -d "$BASE_DIR/.pyenv" ] || git clone https://github.com/pyenv/pyenv.git "$BASE_DIR/.pyenv"
source "$BASE_DIR/env.sh"
pyenv install -s "$PYTHON_VERSION"

# -------------------------------------------------------------
step "5. Repositório (branch $BRANCH), .python-version, .env"
if [ ! -e "$REPO_DIR" ]; then
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git status -sb | head -3
[ "$(git branch --show-current)" = "$BRANCH" ] \
    || warn "o repositório está na branch '$(git branch --show-current)', não em '$BRANCH' (não troquei para não mexer no seu trabalho)"
[ -f src/api/routes/binary_annotation.py ] \
    || warn "src/api/routes/binary_annotation.py não existe: a anotação binária ainda não está nesta branch (commit/push pendente?)"

pyenv local "$PYTHON_VERSION"
if [ ! -f .env ]; then
    cp .env-example .env
    warn ".env criado a partir do .env-example — preencha os tokens"
fi
mkdir -p data/results data/.cache data/datasets

# -------------------------------------------------------------
step "6. Poetry $POETRY_VERSION + .venv no repositório"
if ! command -v poetry > /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 - --version "$POETRY_VERSION" \
        || pipx install "poetry==$POETRY_VERSION"
fi
poetry config virtualenvs.in-project true --local
.venv/bin/python --version > /dev/null 2>&1 || rm -rf .venv
poetry env use "$(pyenv which python)"
poetry install
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available())" || true

# -------------------------------------------------------------
step "7. Ollama (em $OLLAMA_DIR, sem serviço do sistema) + modelos"
# A aplicação chama http://localhost:11434: a porta precisa ser deste Ollama
if ! tmux has-session -t ollama 2> /dev/null && curl -sf localhost:11434/api/tags > /dev/null; then
    if systemctl is-active --quiet ollama 2> /dev/null; then
        [ $HAS_ROOT -eq 1 ] || die "há um Ollama do sistema (systemd) na porta 11434 com outra configuração — peça para pararem: sudo systemctl disable --now ollama"
        warn "parando o Ollama do sistema (systemd) para usar a configuração deste script"
        $SUDO systemctl disable --now ollama
    else
        die "a porta 11434 já está em uso por outro processo: $(command -v ss > /dev/null && ss -ltnp 2> /dev/null | grep 11434)"
    fi
fi

# Instalação completa = binário + llama-server (o processo que roda os modelos)
if [ ! -x "$OLLAMA_BIN" ] || [ ! -x "$OLLAMA_DIR/lib/ollama/llama-server" ]; then
    tmux kill-session -t ollama 2> /dev/null || true
    rm -rf "$OLLAMA_DIR" && mkdir -p "$OLLAMA_DIR"
    # Baixa inteiro e confere antes de extrair: um download interrompido no pipe deixava a instalação pela metade
    curl -fL --progress-bar https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.$$.tar.zst
    zstd -tq /tmp/ollama.$$.tar.zst || die "download do Ollama corrompido — rode o script de novo"
    zstd -dc /tmp/ollama.$$.tar.zst | tar -xf - -C "$OLLAMA_DIR"
    rm -f /tmp/ollama.$$.tar.zst
    [ -x "$OLLAMA_DIR/lib/ollama/llama-server" ] || die "llama-server ausente em $OLLAMA_DIR/lib/ollama após a extração"
fi
"$OLLAMA_BIN" --version

tmux kill-session -t ollama 2> /dev/null || true
tmux new-session -d -s ollama "source $BASE_DIR/env.sh; CUDA_VISIBLE_DEVICES=$GPU $OLLAMA_BIN serve; exec bash"
for _ in $(seq 1 60); do curl -sf localhost:11434/api/tags > /dev/null && break; sleep 1; done
curl -sf localhost:11434/api/tags > /dev/null || die "o Ollama não subiu — veja: tmux attach -t ollama"

# Limita as threads de CPU de cada llama-server (sem isso, cada um usa todos os núcleos)
NUM_THREAD="${OLLAMA_NUM_THREAD:-$(( (CPUS - 2) / ${#MODELS[@]} ))}"
[ "$NUM_THREAD" -lt 1 ] && NUM_THREAD=1
echo "num_thread=$NUM_THREAD por modelo ($CPUS CPUs / ${#MODELS[@]} modelos)"
for m in "${MODELS[@]}"; do
    "$OLLAMA_BIN" show "$m" > /dev/null 2>&1 || "$OLLAMA_BIN" pull "$m"
    "$OLLAMA_BIN" show "$m" --modelfile | grep -v '^PARAMETER num_thread' > /tmp/Modelfile.$$
    echo "PARAMETER num_thread $NUM_THREAD" >> /tmp/Modelfile.$$
    "$OLLAMA_BIN" create "$m" -f /tmp/Modelfile.$$ > /dev/null
done
rm -f /tmp/Modelfile.$$

if [ "$MODO" = "finetuning" ]; then
    echo "MODO=finetuning → Ollama derrubado (GPU livre para os folds)"
    tmux kill-session -t ollama 2> /dev/null || true
else
    for m in "${MODELS[@]}"; do
        curl -s localhost:11434/api/generate -d "{\"model\":\"$m\"}" > /dev/null
    done
    "$OLLAMA_BIN" ps
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    # Modelo que não coube na VRAM roda parte na CPU (lento): reduza o contexto
    if "$OLLAMA_BIN" ps | tail -n +2 | grep -qv "100% GPU"; then
        warn "algum modelo não está 100% na GPU (VRAM insuficiente). Rode de novo com OLLAMA_CONTEXT_LENGTH=8192"
    fi
fi

# -------------------------------------------------------------
step "8. API (uvicorn no tmux 'api')"
OMP_THREADS=$(( CPUS / 5 ))
[ "$OMP_THREADS" -lt 1 ] && OMP_THREADS=1
tmux kill-session -t api 2> /dev/null || true
tmux new-session -d -s api "cd $REPO_DIR && source $BASE_DIR/env.sh && OMP_NUM_THREADS=$OMP_THREADS TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload --reload-dir src; exec bash"
for _ in $(seq 1 120); do curl -sf localhost:8000/health > /dev/null && break; sleep 1; done
curl -s localhost:8000/health; echo

# -------------------------------------------------------------
step "Pronto"
echo "Logs:   tmux attach -t api   |   tmux attach -t ollama   (sair: Ctrl+B depois D)"
