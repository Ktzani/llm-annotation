# Sistema de Anotação Automática com Múltiplas LLMs

Sistema completo para reduzir custos humanos na anotação de datasets através do uso de múltiplas LLMs e análise de consenso.

**Gerenciamento de dependências:** Poetry 🎯

---

## 🚀 Início Rápido com Poetry

### 1. Pré-requisitos

```bash
# Instalar Poetry (se ainda não tiver)
curl -sSL https://install.python-poetry.org | python3 -

# Verificar instalação
poetry --version
```

### 2. Instalar Dependências

```bash
# Instalar apenas dependências de produção
poetry install

# OU instalar com dependências de desenvolvimento
poetry install --with dev

# OU instalar tudo (incluindo extras)
poetry install --with dev --extras all
```

### 3. Configurar API Keys

```bash
# Criar arquivo .env
make setup-env

# Editar com suas chaves
nano config/.env
```

### 4. Executar

```bash
# Opção A: Jupyter Notebook (RECOMENDADO)
make notebook

# Opção B: Script de exemplo
make run-example

# Opção C: Shell interativo
poetry shell
python examples/exemplo_uso.py
```

---

## 📁 Estrutura do Projeto

```
llm_annotation_system/
│
├── pyproject.toml              # ⭐ Configuração Poetry
├── poetry.lock                 # Lock de dependências
├── Makefile                    # Comandos úteis
├── README.md                   # Este arquivo
│
├── src/llm_annotation_system/  # Código-fonte
│   ├── __init__.py
│   ├── llm_annotator.py
│   ├── consensus_analyzer.py
│   ├── visualizer.py
│   └── config.py
│
├── notebooks/                  # Jupyter Notebooks
│   └── analise_consenso_llms.ipynb
│
├── examples/                   # Exemplos
│   └── exemplo_uso.py
│
├── docs/                       # Documentação
├── config/                     # Configurações
├── data/                       # Dados
├── results/                    # Resultados
├── cache/                      # Cache
└── tests/                      # Testes
```

---

## 💻 Comandos Poetry Úteis

### Gerenciamento de Dependências

```bash
# Adicionar dependência
poetry add nome-pacote
# ou
make add pkg=nome-pacote

# Adicionar dependência de desenvolvimento
poetry add --group dev nome-pacote
# ou
make add-dev pkg=nome-pacote

# Remover dependência
poetry remove nome-pacote

# Atualizar dependências
poetry update
# ou
make poetry-update

# Mostrar dependências
poetry show --tree
# ou
make poetry-show
```

### Ambiente Virtual

```bash
# Ativar shell com ambiente virtual
poetry shell

# Executar comando no ambiente
poetry run python script.py

# Desativar shell
exit
```

### Exportar para requirements.txt

```bash
# Se precisar de requirements.txt tradicional
poetry export -f requirements.txt --output requirements.txt
# ou
make poetry-export
```

---

## 🎯 Uso como Pacote

### Instalar em modo desenvolvimento

```bash
# Poetry instala automaticamente em modo editável
poetry install
```

### Usar em outro projeto

```bash
# Em outro projeto Poetry
poetry add git+https://github.com/Ktzani/llm-annotation.git

# Ou após publicar no PyPI
poetry add llm-annotation
```

### Importar no código

```python
from llm_annotation_system import LLMAnnotator, ConsensusAnalyzer

# Usar normalmente
annotator = LLMAnnotator(models, categories, api_keys)
df = annotator.annotate_dataset(texts)
```

---

## 🧪 Testes e Qualidade

```bash
# Executar testes
make test

# Testes com coverage
make test-cov

# Verificar código (flake8)
make lint

# Formatar código (black + isort)
make format

# Verificar tipos (mypy)
make type-check

# Tudo de uma vez
make check-all
```

---

## 📦 Publicar Pacote

```bash
# Build
poetry build

# Publicar no PyPI
poetry publish

# Ou testar no Test PyPI primeiro
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry publish -r testpypi
```

---

## 🎯 Funcionalidades

✅ Anotação com múltiplas LLMs (GPT-4, Claude 3, Gemini, etc.)  
✅ Validação de consenso interno (múltiplas repetições)  
✅ Análise estatística completa (Cohen's Kappa, Fleiss', etc.)  
✅ Teste de variações de parâmetros  
✅ Estratégias de resolução de conflitos  
✅ Visualizações e dashboard interativo  
✅ Sistema de cache para economizar API calls  

---

## 📊 Configuração do pyproject.toml

### Dependências Principais

```toml
[tool.poetry.dependencies]
python = "^3.9"
pandas = "^2.0.0"
numpy = "^1.24.0"
openai = "^1.0.0"
anthropic = "^0.18.0"
# ... outras
```

### Dependências de Desenvolvimento

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.7.0"
flake8 = "^6.1.0"
# ... outras
```

### Extras Opcionais

```toml
[tool.poetry.extras]
cohere = ["cohere"]
all = ["cohere"]
```

Instalar extras:
```bash
poetry install --extras cohere
poetry install --extras all
```

---

## 🔧 Configuração de Ferramentas

O `pyproject.toml` já inclui configurações para:

- **Black**: Formatação de código
- **isort**: Organização de imports
- **mypy**: Verificação de tipos
- **pytest**: Testes e coverage

Tudo está pré-configurado e pronto para uso!

---

## 🆚 Poetry vs pip/requirements.txt

| Característica | Poetry | pip |
|----------------|--------|-----|
| Gerenciamento | ✅ Completo | ⚠️ Básico |
| Lock de versões | ✅ poetry.lock | ❌ Não |
| Ambientes virtuais | ✅ Automático | ⚠️ Manual |
| Publicação PyPI | ✅ Integrado | ⚠️ Manual |
| Resolução de deps | ✅ Inteligente | ⚠️ Simples |

---

## 💡 Dicas

### 1. Sempre use poetry.lock

```bash
# Committar no Git
git add poetry.lock

# Instalar versões exatas
poetry install
```

### 2. Atualizar dependências

```bash
# Atualizar tudo
poetry update

# Atualizar pacote específico
poetry update openai
```

### 3. Verificar vulnerabilidades

```bash
# Auditar dependências
poetry show --outdated
```

### 4. Scripts customizados

Adicione em `pyproject.toml`:

```toml
[tool.poetry.scripts]
meu-comando = "modulo:funcao"
```

Depois use:

```bash
poetry run meu-comando
```

---

## 📖 Documentação

- **Poetry**: https://python-poetry.org/docs/
- **Projeto**: `docs/`
- **Notebook**: `notebooks/analise_consenso_llms.ipynb`

---

## 🤝 Contribuindo

```bash
# 1. Fork o repositório
# 2. Criar branch
git checkout -b feature/nova-funcionalidade

# 3. Instalar dependências de dev
poetry install --with dev

# 4. Fazer mudanças e testar
make check-all

# 5. Commit e push
git commit -am "Adiciona nova funcionalidade"
git push origin feature/nova-funcionalidade

# 6. Abrir Pull Request
```

---

## 📄 Licença

MIT License - Veja [LICENSE](LICENSE)

---

## 📞 Contato

**Autor**: Gabriel Catizani  
**Email**: gabrielcatizani01@gmail.com

---

⭐ **Desenvolvido com Poetry!** 

Para comandos úteis, execute: `make help`
