# Instruções de Uso do Sistema

## O Que Você Tem

Um **sistema completo** para anotação automática com LLMs open-source e fine-tuning supervisionado. O código está organizado em dois pipelines principais: anotação com consenso e fine-tuning RoBERTa.

---

## Estrutura do Projeto

### Configurações (`src/config/`)

| Arquivo | Conteúdo |
|---------|----------|
| `prompts.py` | Templates de prompts (zero-shot, few-shot, chain-of-thought) |
| `llms.py` | Registry de 25+ modelos open-source com parâmetros |
| `datasets_collected.py` | 30+ datasets HuggingFace pré-configurados |
| `conflict_resolution.py` | Estratégias de resolução de conflitos |
| `evaluation_metrics.py` | Nomes das métricas de avaliação |

### Pipeline de Anotação (`src/llm_annotation_system/`)

| Módulo | Função |
|--------|--------|
| `annotation/llm_annotator.py` | Orquestrador — coordena modelos, cache, engine |
| `annotation/annotation_engine.py` | Executa anotações com cache e repetições |
| `core/llm_provider.py` | Inicializa modelos (Ollama, HF, Groq) |
| `core/response_processor.py` | Extrai label e confiança da resposta da LLM |
| `core/cache_manager.py` | Cache de chamadas para economizar tempo |
| `consensus/consensus_calculator.py` | Calcula consenso e aplica estratégia de resolução |
| `consensus/consensus_metrics.py` | Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha |

### Pipeline de Fine-Tuning (`src/fine_tune_system/`)

| Módulo | Função |
|--------|--------|
| `fine_tune/supervised_fine_tuner.py` | Fine-tuning RoBERTa-base |
| `training/cross_validator.py` | Cross-validation 5-fold |
| `training/splits_aligner.py` | Alinha splits + remove data leakage |
| `training/trainer_builder.py` | Configura HuggingFace Trainer |
| `pipeline.py` | Orquestrador do fine-tuning |

### Scripts de Entrada

| Script | Função |
|--------|--------|
| `run_annotation.py` | Executa pipeline de anotação |
| `run_fine_tunning.py` | Executa pipeline de fine-tuning |
| `src/api/server.py` | Servidor REST (FastAPI) |

---

## Como Começar

### Passo 1: Instalar Dependências

```bash
poetry install
```

### Passo 2: Configurar Variáveis de Ambiente

Crie `.env` na raiz:

```env
# Para HuggingFace API
HF_TOKEN=seu-token-aqui

# Para Groq API
GROQ_API_KEY=sua-chave-aqui

# Para GPU (fine-tuning)
CUDA_VISIBLE_DEVICES=0
```

### Passo 3: Instalar Ollama (modelos locais)

```bash
# https://ollama.com/download
ollama pull llama3.1:8b   # ou qualquer modelo de src/config/llms.py
```

### Passo 4: Executar Anotação

```bash
poetry run python run_annotation.py
```

### Passo 5: Executar Fine-Tuning

```bash
poetry run python run_fine_tunning.py
```

---

## O Que o Sistema Faz

### 1. Anotação Automática

- Múltiplas LLMs open-source anotam cada texto
- Cada LLM anota múltiplas vezes (validação interna)
- Sistema de cache (não repete chamadas)
- Extração de label + confiança da resposta

### 2. Análise de Consenso

- Calcula consenso entre todas as LLMs
- Identifica casos problemáticos (empates, discordâncias)
- Métricas: Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha
- Estratégias configuráveis de resolução de conflitos

### 3. Validação com Ground Truth (Opcional)

- Se dataset tem labels, valida automaticamente
- Calcula accuracy, precision, recall, F1
- Gera classification report

### 4. Fine-Tuning RoBERTa

- Remove instâncias com label=-1 (logadas para depuração)
- Remove instâncias problemáticas do consenso
- **Previne data leakage**: remove do teste instâncias presentes no treino
- Cross-validation 5-fold com checkpoints por fold

### 5. Outputs Gerados

**CSVs:**
- `dataset_anotado_final.csv` — Dataset final anotado
- `annotations_complete.csv` — Todas as anotações detalhadas
- `high_confidence_annotations.csv` — Consenso ≥ 80%
- `needs_human_review.csv` — Casos problemáticos

**Visualizações:**
- `agreement_heatmap.png` — Heatmap de concordância
- `consensus_distribution.png` — Distribuição de consenso
- `interactive_dashboard.html` — Dashboard interativo

**Resumos:**
- `experiment_summary.json` — Estatísticas completas

---

## Modelos Disponíveis

Todos open-source, sem custo de API proprietária:

```python
# src/config/llms.py — exemplos
"llama3.1:8b"       # Meta, via Ollama
"mistral:7b"        # Mistral AI, via Ollama
"gemma2:9b"         # Google, via Ollama
"phi3.5:mini"       # Microsoft, via Ollama
"deepseek-r1:8b"    # DeepSeek, via Ollama
"qwen2.5:7b"        # Alibaba, via Ollama
```

**Providers:**
- **Ollama**: Local, gratuito, privado
- **HuggingFace Inference API**: Nuvem, gratuito com limites
- **Groq**: Nuvem, muito rápido, tier gratuito

---

## Usar Datasets do HuggingFace

Configure em `src/config/datasets_collected.py`:

```python
DATASETS = {
    "meu_dataset": {
        "path": "waashk/nome-do-dataset",
        "text_column": "text",
        "label_column": "label",          # opcional (para validação)
        "categories": None,               # extrair automaticamente
        "combine_splits": ["train", "test"],
        "sample_size": 100,               # começar pequeno
    }
}
```

Ver [GUIA_DATASETS.md](GUIA_DATASETS.md) para mais detalhes.

---

## Customizações

### Adicionar Novo Modelo

Em `src/config/llms.py`:

```python
LLM_CONFIGS["novo-modelo"] = {
    "provider": "ollama",   # ou "huggingface", "groq"
    "model_name": "nome:tag",
    "default_params": {
        "temperature": 0.0,
        "num_predict": 50,
    },
    "alternative_params": [
        {"temperature": 0.3},
        {"temperature": 0.5},
    ]
}
```

### Customizar Prompts

Em `src/config/prompts.py`:

```python
BASE_ANNOTATION_PROMPT = """
Classifique o texto abaixo.

**Text:**
{text}

**Categories:**
{categories}

CLASSIFICATION:
"""
```

---

## Dicas

### Para Reduzir Tempo de Execução

1. Usar modelos menores (7b-8b) para triagem inicial
2. Ativar cache (`cache_manager`) — evita re-anotação de textos já processados
3. Usar Groq para execução mais rápida (300+ tokens/s)

### Para Melhorar Qualidade

1. Ajustar prompts em `src/config/prompts.py`
2. Testar `few_shot` e `chain_of_thought` prompts
3. Analisar `needs_human_review.csv` para entender discordâncias

### Para Datasets Grandes

1. Usar `sample_size` para testes antes de escalar
2. Processar em batches
3. Cache local em `data/.cache/huggingface/`

---

## Checklist

Antes de executar:

- [ ] `poetry install` executado
- [ ] Ollama instalado + pelo menos 1 modelo baixado
- [ ] `.env` criado com as keys necessárias
- [ ] Dataset configurado em `datasets_collected.py`
- [ ] Testado com `sample_size: 100`
- [ ] Prompts revisados para o domínio

---


OLLAMA_NUM_PARALLEL=5  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_KEEP_ALIVE=24h ollama serve

curl http://localhost:11434/api/generate -d '{"model": "qwen3:8b"}'
curl http://localhost:11434/api/generate -d '{"model": "llama3.1:8b"}'
curl http://localhost:11434/api/generate -d '{"model": "deepseek-r1:8b"}'


brev copy -r ./data lbd-8a100-server:/home/nvidia/workspace/catizani/llm-annotation

brev copy ./data/results/results.zip lbd-8a100-server:/home/nvidia/workspace/catizani/llm-annotation/data

docker build -f docker/Dockerfile -t llm-annotation:latest .

GPU_IDS=<GPUs> CUDA_DEVICE_IDS=<0, 1, ... (colocar a partir da quantidade de GPUs)> PORT=8000 docker compose -f docker/docker-compose.yml up -d