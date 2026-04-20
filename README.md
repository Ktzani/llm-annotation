# Sistema de Anotação Automática com Múltiplas LLMs

Sistema completo para anotação automática de datasets usando múltiplas LLMs open-source com análise de consenso, seguido de fine-tuning supervisionado com RoBERTa.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Metodologia](#metodologia)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Modelos Suportados](#modelos-suportados)
- [Datasets Suportados](#datasets-suportados)
- [Fine-Tuning](#fine-tuning)
- [API REST](#api-rest)
- [Resultados e Métricas](#resultados-e-métricas)
- [FAQ](#faq)

## 🎯 Visão Geral

Este projeto implementa uma metodologia para anotação automática de datasets usando múltiplas LLMs open-source com análise de consenso. O objetivo é reduzir significativamente o custo e tempo necessários para anotação manual, mantendo alta qualidade nas classificações. Os dados anotados são então usados para fine-tuning de um modelo RoBERTa via cross-validation.

### Características Principais

- **LLMs Open-Source**: Suporte para Llama, Mistral, Gemma, Phi-3, DeepSeek, Qwen, via Ollama, HuggingFace API e Groq
- **Consenso Robusto**: Cada LLM anota múltiplas vezes para validação interna
- **Análise Estatística**: Métricas completas de concordância (Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha)
- **Fine-Tuning**: Pipeline completo de fine-tuning RoBERTa com cross-validation
- **Prevenção de Data Leakage**: Remoção automática de instâncias repetidas entre treino e teste
- **Rastreamento de Instâncias Inválidas**: Log de instâncias com label=-1 para depuração
- **HuggingFace**: Integração com 30+ datasets do HuggingFace (namespace waashk)
- **Cache**: Sistema de cache para economizar chamadas de API
- **API REST**: Servidor FastAPI para gerenciamento de experimentos
- **GPU**: Suporte a CUDA para fine-tuning acelerado

## 📊 Metodologia

### 1. Anotação Múltipla
- Múltiplas LLMs anotam cada instância do dataset
- Cada LLM faz múltiplas anotações (padrão: 3x) da mesma instância
- Validação de consenso interno para cada LLM

### 2. Análise de Consenso
- Cálculo de tabela de consenso entre LLMs
- Estatísticas por instância (porcentagem de acordo)
- Identificação de casos problemáticos (empates, discordâncias)

### 3. Validação de Parâmetros (LLM Hacking)
- Teste de diferentes configurações (temperatura, num_predict, etc.)
- Avaliação do impacto de variações de parâmetros
- Identificação de configurações mais estáveis

### 4. Estratégias de Resolução
Para casos sem consenso claro:
- **Voto majoritário**: Escolhe a classe mais votada
- **Threshold**: Aceita apenas se consenso ≥ X%
- **Flag for review**: Marca para revisão humana
- **Remove outliers**: Remove instâncias da minoria abaixo do threshold
- **Weighted vote**: Voto ponderado por confiabilidade do modelo
- **Unanimous only**: Aceita apenas com 100% de acordo

### 5. Fine-Tuning (RoBERTa)
- Filtragem de instâncias inválidas (label=-1) e problemáticas
- Prevenção de data leakage: remove instâncias do treino que aparecem no teste
- Fine-tuning RoBERTa-base com cross-validation de 5 folds
- Métricas por fold e métricas agregadas

## 🚀 Instalação

### Pré-requisitos

- Python 3.9+
- Poetry (gerenciador de dependências)
- Ollama (para modelos locais) **ou** HuggingFace API token **ou** Groq API key

### Instalação de Dependências

```bash
poetry install
```

### Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Para HuggingFace API
HF_TOKEN=seu-token-aqui

# Para Groq API
GROQ_API_KEY=sua-chave-aqui

# Para fine-tuning com GPU (opcional)
CUDA_VISIBLE_DEVICES=0
```

### Configurar Ollama (modelos locais)

```bash
# Instalar Ollama
# https://ollama.com/download

# Baixar um modelo
ollama pull llama3.1:8b
ollama pull mistral:7b
```

## 🏃 Uso Rápido

### Executar Anotação

```bash
# Configurar experimento em um JSON (ver src/config/)
poetry run python run_annotation.py
```

### Executar Fine-Tuning

```bash
poetry run python run_fine_tunning.py
```

### Iniciar API REST

```bash
poetry run uvicorn src.api.server:app --reload
```

### Notebook Jupyter

```bash
poetry run jupyter notebook src/notebooks/analise_consenso_llms.ipynb
```

## 🤖 Modelos Suportados

Todos os modelos são open-source, sem custo de API proprietária:

| Família | Modelos | Provider |
|---------|---------|----------|
| **Meta LLaMA** | Llama2-7b, Llama3-8b, Llama3-70b, Llama3.1-8b, Llama3.1-70b | Ollama, HF |
| **Mistral** | Mistral-7b, Mixtral-8x7b, Mistral Nemo-12b | Ollama, HF |
| **Google Gemma** | Gemma-7b, Gemma2-9b, Gemma2-27b, Gemma3-4b | Ollama, HF |
| **Microsoft Phi-3** | Phi3-mini, Phi3.5-mini, Phi3.5-medium | Ollama, HF |
| **DeepSeek** | DeepSeek-R1-8b, R1-14b, R1-Distill-Llama-8B, DeepSeek-V3 | Ollama, Groq |
| **Qwen** | Qwen2-7b, Qwen2.5-7b, Qwen2.5-32b, Qwen3-8b | Ollama, HF |
| **BigScience** | BLOOMZ | HF |

**Providers:**
- **Ollama**: Execução local (100% privado, sem custo de API)
- **HuggingFace Inference API**: Nuvem
- **Groq**: Muito rápido (300+ tokens/s), com tier gratuito

Configuração dos modelos em `src/config/llms.py`.

## 🤗 Datasets Suportados

30+ datasets configurados em `src/config/datasets_collected.py` (namespace `waashk` no HuggingFace):

| Tipo | Datasets |
|------|----------|
| **Notícias** | AG News, Reuters-90 |
| **Sentimento** | MPQA, Yelp 2013/2015, Movie Reviews, SST1/SST2, Vader Movie |
| **Científico** | ACM, DBLP, WOS-11967, WOS-5736, OHSUMED |
| **Web** | WebKB, 20 Newsgroups, Twitter, TREC |
| **Médico** | Medline |
| **Outros** | Books, Pang Movie |

Para adicionar um dataset:

```python
# src/config/datasets_collected.py
DATASETS = {
    "meu_dataset": {
        "path": "waashk/nome-dataset",
        "text_column": "text",
        "label_column": "label",
        "categories": None,             # extrair automaticamente
        "combine_splits": ["train", "test"],
        "sample_size": 100,
    }
}
```

## 🎛️ Fine-Tuning

O pipeline de fine-tuning treina um modelo RoBERTa com os rótulos gerados por consenso das LLMs.

### Execução

```bash
poetry run python run_fine_tunning.py
```

### O pipeline faz automaticamente:
1. Carrega as anotações de consenso geradas
2. Remove instâncias com label inválido (label=-1) — com log
3. Remove instâncias problemáticas marcadas no consenso
4. Alinha splits HuggingFace com os dados anotados via `text_id`
5. **Remove instâncias do teste que aparecem no treino** (prevenção de data leakage)
6. Treina RoBERTa-base com cross-validation de 5 folds
7. Calcula métricas por fold e agrega resultados

### Prevenção de Data Leakage

O sistema detecta e remove automaticamente instâncias duplicadas entre treino e teste, emitindo warning em vez de erro, para garantir avaliação justa:

```
WARNING: 42 instâncias do teste aparecem no treino e foram removidas (data leakage)
```

## 🌐 API REST

O projeto inclui um servidor FastAPI para gerenciar experimentos via HTTP.

```bash
poetry run uvicorn src.api.server:app --reload
# Docs: http://localhost:8000/docs
```

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| POST | `/experiments` | Criar experimento de anotação |
| GET | `/experiments/{id}` | Status do experimento |
| POST | `/fine-tuning` | Iniciar job de fine-tuning |
| GET | `/fine-tuning/{id}` | Status do fine-tuning |
| GET | `/datasets` | Listar datasets disponíveis |
| GET | `/health` | Health check |

## 📁 Estrutura do Projeto

```
├── pyproject.toml                      # Configuração Poetry
├── .env                                # Variáveis de ambiente (criar)
├── run_annotation.py                   # ⭐ Entry point: anotação
├── run_fine_tunning.py                 # ⭐ Entry point: fine-tuning
│
├── src/
│   ├── config/                         # Configurações
│   │   ├── prompts.py                 # Templates de prompts
│   │   ├── llms.py                    # ⭐ Registry de modelos open-source
│   │   ├── datasets_collected.py      # ⭐ 30+ datasets HuggingFace
│   │   ├── conflict_resolution.py     # Estratégias de resolução
│   │   └── evaluation_metrics.py      # Métricas de avaliação
│   │
│   ├── llm_annotation_system/          # Pipeline de anotação
│   │   ├── annotation/
│   │   │   ├── llm_annotator.py       # Orquestrador principal
│   │   │   └── annotation_engine.py   # Motor de anotação
│   │   ├── core/
│   │   │   ├── llm_provider.py        # Inicialização de modelos
│   │   │   ├── response_processor.py  # Extração de labels
│   │   │   └── cache_manager.py       # Cache de chamadas
│   │   └── consensus/
│   │       ├── consensus_calculator.py # Cálculo de consenso
│   │       └── consensus_metrics.py   # Métricas estatísticas
│   │
│   ├── fine_tune_system/               # Pipeline de fine-tuning
│   │   ├── fine_tune/
│   │   │   └── supervised_fine_tuner.py # Fine-tuning RoBERTa
│   │   ├── training/
│   │   │   ├── cross_validator.py     # Cross-validation 5-fold
│   │   │   ├── splits_aligner.py      # ⭐ Alinhamento + anti-leakage
│   │   │   └── trainer_builder.py     # Configuração do HF Trainer
│   │   └── pipeline.py                # Orquestração do fine-tuning
│   │
│   ├── api/                            # API REST (FastAPI)
│   │   ├── server.py                  # Aplicação FastAPI
│   │   ├── routes/                    # Endpoints
│   │   └── services/                  # Lógica de negócio
│   │
│   ├── utils/                          # Utilitários
│   │   ├── data_loader.py             # Carregamento HuggingFace
│   │   └── get_text_id_from_text.py   # Geração de IDs únicos
│   │
│   └── notebooks/
│       └── analise_consenso_llms.ipynb # ⭐ Notebook de análise
│
├── data/
│   └── .cache/                         # Cache de datasets
│
├── results/                            # Resultados gerados
│   ├── figures/                        # Visualizações
│   ├── reports/                        # Relatórios CSV
│   └── final/                          # Resultados finais
│
└── docs/                               # Documentação
    ├── INSTRUCOES.md
    ├── RESUMO_EXECUTIVO.md
    └── GUIA_DATASETS.md
```

## 📊 Resultados e Métricas

### Interpretação do Consenso

**Cohen's Kappa**:
- `> 0.80`: Excelente
- `0.60 - 0.80`: Bom
- `0.40 - 0.60`: Moderado
- `< 0.40`: Fraco

**Consenso Score**:
- `≥ 80%`: Alto — aceitar
- `60-80%`: Médio — revisar amostra
- `< 60%`: Baixo — revisão obrigatória

### Arquivos Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `dataset_anotado_final.csv` | Dataset com anotações finais de consenso |
| `annotations_complete.csv` | Todas as anotações detalhadas por modelo |
| `high_confidence_annotations.csv` | Instâncias com consenso ≥ 80% |
| `needs_human_review.csv` | Casos problemáticos para revisão |
| `experiment_summary.json` | Estatísticas completas do experimento |
| `interactive_dashboard.html` | Dashboard interativo (Plotly) |

## 💡 FAQ

**Q: Preciso de API keys pagas?**
A: Não. O sistema usa apenas modelos open-source via Ollama (local e gratuito), HuggingFace API (gratuito com limites) ou Groq (tier gratuito generoso).

**Q: Quantos modelos usar?**
A: Recomendamos 3-5 modelos de arquiteturas diferentes para consenso robusto.

**Q: Quantas repetições?**
A: 3 repetições é um bom balanço entre confiabilidade e tempo de execução.

**Q: O que é data leakage e como o sistema lida?**
A: O sistema detecta automaticamente instâncias que aparecem tanto no treino quanto no teste e as remove do conjunto de avaliação, emitindo um warning no log.

**Q: O que significa label=-1?**
A: A LLM retornou uma resposta fora das categorias configuradas. Essas instâncias são logadas e removidas antes do fine-tuning.

**Q: Como usar meus datasets do HuggingFace?**
A: Adicione a configuração em `src/config/datasets_collected.py`. Veja [GUIA_DATASETS.md](docs/GUIA_DATASETS.md) para instruções completas.

## 📚 Documentação Adicional

- [INSTRUCOES.md](docs/INSTRUCOES.md) - Guia completo de uso
- [RESUMO_EXECUTIVO.md](docs/RESUMO_EXECUTIVO.md) - Resumo para orientador
- [GUIA_DATASETS.md](docs/GUIA_DATASETS.md) - Guia de datasets HuggingFace

---
