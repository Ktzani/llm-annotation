# ============================================================
# llm-annotation — Imagem Docker com suporte a GPU (CUDA 12.1)
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- variáveis de build ---
ARG PYTHON_VERSION=3.11.9
ARG POETRY_VERSION=2.2.1

# --- variáveis de runtime ---
# GPU padrão: 0. Sobrescreva na hora de rodar:
#   docker run -e GPU_ID=3 ...
ENV GPU_ID=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/data/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PATH="/root/.local/bin:$PATH"

# --- dependências do sistema ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        build-essential \
        libssl-dev \
        libffi-dev \
        # pciutils: necessário para o instalador do Ollama detectar a GPU
        pciutils \
        # zstd: o instalador do Ollama usa para extrair o pacote
        zstd \
        # necessários para compilar extensões Python
        python3-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION%.*} \
        python${PYTHON_VERSION%.*}-distutils \
        python${PYTHON_VERSION%.*}-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION%.*} \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION%.*} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION%.*} 1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Ollama ---
# Instala o binário do ollama. Os modelos NÃO são baixados aqui — eles são
# puxados no primeiro start pelo entrypoint e persistem no volume montado
# em /root/.ollama (ver docker-compose.yml).
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- Poetry ---
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}

# --- diretório de trabalho ---
WORKDIR /app

# --- copia somente os arquivos de dependência primeiro (cache de layers) ---
COPY pyproject.toml poetry.lock* ./

# --- instala dependências (sem o pacote em si ainda) ---
# O source "pytorch" com CUDA 12.1 já está declarado no pyproject.toml
RUN poetry install --no-root --no-directory

# --- copia o restante do projeto ---
COPY . .

# --- instala o pacote (src/) ---
RUN poetry install --only-root

# --- diretórios de dados (montados como volume em produção) ---
RUN mkdir -p /app/data/results /app/data/.cache

# --- script de entrypoint: aplica CUDA_VISIBLE_DEVICES antes de subir a API ---
RUN mv /app/docker/docker-entrypoint.sh /docker-entrypoint.sh \
    && chmod +x /docker-entrypoint.sh
 
EXPOSE 8000
 
ENTRYPOINT ["/docker-entrypoint.sh"]
