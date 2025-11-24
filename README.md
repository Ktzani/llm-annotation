# Sistema de Anotação Automática com Múltiplas LLMs

Sistema completo para reduzir custos humanos na anotação de datasets através do uso de múltiplas LLMs e análise de consenso.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Metodologia](#metodologia)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Guia Detalhado](#guia-detalhado)
- [Resultados e Métricas](#resultados-e-métricas)
- [FAQ](#faq)

## 🎯 Visão Geral

Este projeto implementa uma metodologia para anotação automática de datasets usando múltiplas LLMs (Large Language Models) com análise de consenso. O objetivo é reduzir significativamente o custo e tempo necessários para anotação manual, mantendo alta qualidade nas classificações.

### Características Principais

✅ **Múltiplas LLMs**: Suporte para GPT-4, GPT-3.5, Claude 3, Gemini, e Cohere  
✅ **Consenso Robusto**: Cada LLM anota múltiplas vezes para validação interna  
✅ **Análise Estatística**: Métricas completas de concordância (Cohen's Kappa, Fleiss' Kappa, etc.)  
✅ **Visualizações**: Gráficos e dashboards interativos  
✅ **HuggingFace**: Integração completa com datasets do HuggingFace  
✅ **Flexível**: Suporte para diferentes estratégias de resolução de conflitos  
✅ **Cache**: Sistema de cache para economizar chamadas de API  

## 📊 Metodologia

A metodologia implementada segue os seguintes passos:

### 1. Anotação Múltipla
- 5 LLMs diferentes anotam cada instância do dataset
- Cada LLM faz múltiplas anotações (padrão: 3x) da mesma instância
- Validação de consenso interno para cada LLM

### 2. Análise de Consenso
- Cálculo de tabela de consenso entre LLMs
- Estatísticas por instância (porcentagem de acordo)
- Identificação de casos problemáticos (empates, discordâncias)

### 3. Validação de Parâmetros (LLM Hacking)
- Teste de diferentes configurações (temperatura, top_p, etc.)
- Avaliação do impacto de variações de parâmetros
- Identificação de configurações mais estáveis

### 4. Estratégias de Resolução
Para casos sem consenso claro:
- **Voto majoritário**: Escolhe a classe mais votada
- **Threshold**: Aceita apenas se consenso ≥ X%
- **Flag for review**: Marca para revisão humana
- **Remove**: Remove instâncias problemáticas
- **Weighted vote**: Voto ponderado por confiabilidade do modelo

## 🚀 Instalação

### Pré-requisitos

- Python 3.9+
- Poetry (gerenciador de dependências)
- API keys para as LLMs que deseja usar

### Instalação de Dependências

```bash
# Instalar dependências com Poetry
poetry install
```

### Configurar API Keys

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sua-chave-aqui
ANTHROPIC_API_KEY=sua-chave-aqui
GOOGLE_API_KEY=sua-chave-aqui
```

## 🏃 Uso Rápido

### Opção 1: Notebook Jupyter (RECOMENDADO)

```bash
poetry run jupyter notebook src/notebooks/analise_consenso_llms.ipynb
```

Este notebook contém:
- Setup completo
- Exemplos de uso
- Análises detalhadas
- Visualizações
- Interpretação de resultados

### Opção 2: Script Principal

```bash
# Exemplo básico
poetry run python src/main.py

# Com datasets HuggingFace
poetry run python src/main_huggingface.py --modo basico
```

## 🤗 Datasets do HuggingFace

Este projeto tem suporte completo para datasets do HuggingFace!

### Início Rápido com HuggingFace

1. **Descobrir estrutura do dataset:**
```bash
poetry run python src/main_huggingface.py --modo descobrir --dataset waashk/seu-dataset
```

2. **Configurar dataset:**
Edite `src/config/dataset_config.py` com as informações do seu dataset

3. **Executar anotação:**
```bash
poetry run python src/main_huggingface.py --modo basico
```

### Documentação HuggingFace

Ver guias completos:
- [GUIA_DATASETS.md](docs/GUIA_DATASETS.md) - Guia completo

### Arquivos Importantes

- `src/config/dataset_config.py` - Configuração de datasets HuggingFace
- `src/main_huggingface.py` - Script principal com HuggingFace
- `docs/GUIA_DATASETS.md` - Guia completo de uso

## 📁 Estrutura do Projeto

```
├── pyproject.toml                    # Configuração Poetry
├── .env                              # API keys (criar)
├── Makefile                          # Comandos úteis
│
├── src/
│   ├── config/                       # Configurações
│   │   ├── prompts.py               # Templates de prompts
│   │   ├── llm_configs.py           # Configuração de modelos
│   │   ├── experiment.py            # Parâmetros do experimento
│   │   ├── evaluation.py            # Métricas de avaliação
│   │   ├── conflict_resolution.py   # Estratégias de resolução
│   │   └── dataset_config.py        # ⭐ Datasets HuggingFace
│   │
│   ├── llm_annotation_system/        # Código principal
│   │   ├── llm_annotator.py         # Anotador principal
│   │   ├── consensus_analyzer.py    # Análise de consenso
│   │   └── visualizer.py            # Visualizações
│   │
│   ├── notebooks/                    # Jupyter Notebooks
│   │   └── analise_consenso_llms.ipynb  # ⭐ Notebook principal
│   │
│   ├── main.py                       # Script exemplo básico
│   └── main_huggingface.py           # ⭐ Script com HuggingFace
│
├── data/                             # Dados
│   └── .cache/                       # Cache de datasets
│
├── results/                          # Resultados gerados
│   ├── figures/                      # Visualizações
│   ├── reports/                      # Relatórios CSV
│   └── final/                        # Resultados finais
│
└── docs/                             # Documentação
    ├── INSTRUCOES.md
    ├── RESUMO_EXECUTIVO.md
    └── GUIA_DATASETS.md          # ⭐ Guia HuggingFace
    
```

## 📖 Guia Detalhado

### Customizar Prompts

Edite `src/config/prompts.py`:

```python
BASE_ANNOTATION_PROMPT = """You are an expert...
{text}
{categories}
"""
```

### Adicionar Novos Modelos

Em `src/config/llm_configs.py`:

```python
LLM_CONFIGS["seu-modelo"] = {
    "provider": "openai",
    "model_name": "nome-do-modelo",
    "default_params": {"temperature": 0.0}
}
```

### Configurar Datasets HuggingFace

Em `src/config/dataset_config.py`:

```python
HUGGINGFACE_DATASETS = {
    "meu_dataset": {
        "path": "waashk/nome-dataset",
        "text_column": "text",
        "label_column": "label",  # opcional
        "categories": None,  # extrair automaticamente
        "combine_splits": ["train", "test"],  # dataset completo
        "sample_size": 100,  # começar pequeno
    }
}
```

### Testar Variações de Parâmetros

```python
df = annotator.annotate_dataset(
    texts=texts,
    test_param_variations=True  # Testa diferentes parâmetros
)
```

## 📊 Resultados e Métricas

### Interpretação

**Cohen's Kappa**:
- `> 0.80`: Excelente ✅
- `0.60 - 0.80`: Bom ✅
- `0.40 - 0.60`: Moderado ⚠️
- `< 0.40`: Fraco ❌

**Consenso Score**:
- `≥ 80%`: Alto - aceitar ✅
- `60-80%`: Médio - revisar amostra ⚠️
- `< 60%`: Baixo - revisão obrigatória ❌

### Arquivos Gerados

1. **dataset_anotado_final.csv**: Dataset com anotações finais
2. **annotations_complete.csv**: Todas anotações detalhadas
3. **high_confidence_annotations.csv**: Consenso ≥ 80%
4. **needs_human_review.csv**: Casos problemáticos
5. **experiment_summary.json**: Estatísticas completas
6. **interactive_dashboard.html**: Dashboard interativo

## 🔧 Comandos Make

```bash
make help              # Ver todos os comandos
make install           # Instalar dependências
make notebook          # Abrir Jupyter Notebook
make clean             # Limpar arquivos temporários
make format            # Formatar código
make lint              # Verificar código
```

## 💡 FAQ

**Q: Quantos modelos usar?**  
A: Recomendamos 5 modelos de diferentes provedores para consenso robusto.

**Q: Quantas repetições?**  
A: 3 repetições é um bom balanço entre confiabilidade e custo.

**Q: Como reduzir custos?**  
A: Use cache, amostras pequenas (`sample_size`), e modelos mais baratos inicialmente.

**Q: O que fazer com casos sem consenso?**  
A: Depende do caso - revisão humana é mais confiável, voto majoritário é mais rápido.

**Q: Como usar meus datasets do HuggingFace?**  
A: Veja [GUIA_HUGGINGFACE.md](docs/GUIA_HUGGINGFACE.md) para instruções completas.

**Q: Posso usar o dataset completo sem dividir train/test?**  
A: Sim! Use `combine_splits: ["train", "test"]` em `dataset_config.py`.

## 📚 Documentação Adicional

- [INSTRUCOES.md](docs/INSTRUCOES.md) - Guia rápido geral
- [RESUMO_EXECUTIVO.md](docs/RESUMO_EXECUTIVO.md) - Resumo para orientador
- [GUIA_DATASETS.md](docs/GUIA_DATASETS.md) - Guia completo HuggingFace

---
