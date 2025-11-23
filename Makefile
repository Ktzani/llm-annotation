# Makefile para Sistema de Anotação com LLMs (Poetry)

.PHONY: help install install-dev clean test lint format notebook run-example

help:
	@echo "Comandos disponíveis:"
	@echo ""
	@echo "  📦 Instalação:"
	@echo "    make install       - Instala dependências com Poetry"
	@echo "    make install-dev   - Instala com dependências de desenvolvimento"
	@echo "    make setup-env     - Cria arquivo .env"
	@echo ""
	@echo "  🧹 Limpeza:"
	@echo "    make clean         - Remove arquivos temporários"
	@echo "    make clean-all     - Remove tudo (incluindo .venv)"
	@echo ""
	@echo "  🧪 Testes e Qualidade:"
	@echo "    make test          - Executa testes"
	@echo "    make test-cov      - Testes com coverage"
	@echo "    make lint          - Verifica código (flake8)"
	@echo "    make format        - Formata código (black + isort)"
	@echo "    make type-check    - Verifica tipos (mypy)"
	@echo ""
	@echo "  🚀 Execução:"
	@echo "    make notebook      - Inicia Jupyter Notebook"
	@echo "    make run-example   - Executa exemplo"
	@echo ""
	@echo "  📊 Poetry:"
	@echo "    make poetry-show   - Mostra dependências"
	@echo "    make poetry-update - Atualiza dependências"
	@echo "    make poetry-lock   - Atualiza poetry.lock"
	@echo ""
	@echo "  🔧 Git:"
	@echo "    make git-init      - Inicializa repositório Git"

# Instalação
install:
	@echo "📦 Instalando com Poetry..."
	poetry install

install-dev:
	@echo "📦 Instalando com dependências de desenvolvimento..."
	poetry install --with dev

install-all:
	@echo "📦 Instalando tudo (incluindo extras)..."
	poetry install --with dev --extras all

# Limpeza
clean:
	@echo "🧹 Limpando arquivos temporários..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.ipynb_checkpoints' -delete
	find . -type f -name '.DS_Store' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage

clean-all: clean
	@echo "🧹 Removendo ambiente virtual..."
	rm -rf .venv

# Testes
test:
	@echo "🧪 Executando testes..."
	poetry run pytest tests/ -v

test-cov:
	@echo "🧪 Executando testes com coverage..."
	poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "📊 Relatório em: htmlcov/index.html"

# Qualidade de código
lint:
	@echo "🔍 Verificando código com flake8..."
	poetry run flake8 src/ examples/ --max-line-length=100

format:
	@echo "✨ Formatando código..."
	poetry run black src/ examples/ --line-length=100
	poetry run isort src/ examples/

type-check:
	@echo "🔍 Verificando tipos com mypy..."
	poetry run mypy src/

check-all: format lint type-check
	@echo "✅ Todas as verificações completas!"

# Execução
notebook:
	@echo "📓 Iniciando Jupyter Notebook..."
	poetry run jupyter notebook notebooks/analise_consenso_llms.ipynb

run-example:
	@echo "🚀 Executando exemplo..."
	poetry run python examples/exemplo_uso.py

# Configuração
setup-env:
	@if [ ! -f config/.env ]; then \
		cp config/.env.example config/.env; \
		echo "✓ Arquivo .env criado! Edite config/.env com suas API keys"; \
	else \
		echo "⚠️  Arquivo .env já existe!"; \
	fi

# Poetry
poetry-show:
	@echo "📊 Mostrando dependências..."
	poetry show --tree

poetry-update:
	@echo "📦 Atualizando dependências..."
	poetry update

poetry-lock:
	@echo "🔒 Atualizando poetry.lock..."
	poetry lock --no-update

poetry-export:
	@echo "📤 Exportando requirements.txt..."
	poetry export -f requirements.txt --output requirements.txt --without-hashes

# Git
git-init:
	@echo "🔧 Inicializando Git..."
	git init
	git add .
	git commit -m "Initial commit: Sistema de Anotação com LLMs"
	@echo "✓ Repositório Git inicializado!"

# Build
build:
	@echo "📦 Criando distribuição..."
	poetry build

publish:
	@echo "📤 Publicando no PyPI..."
	poetry publish

# Shell interativo
shell:
	@echo "🐚 Iniciando shell Poetry..."
	poetry shell

# Adicionar dependência
add:
	@echo "Uso: make add pkg=nome-do-pacote"
	@echo "Exemplo: make add pkg=requests"
ifdef pkg
	poetry add $(pkg)
else
	@echo "❌ Erro: especifique pkg=nome-do-pacote"
endif

add-dev:
	@echo "Uso: make add-dev pkg=nome-do-pacote"
ifdef pkg
	poetry add --group dev $(pkg)
else
	@echo "❌ Erro: especifique pkg=nome-do-pacote"
endif
