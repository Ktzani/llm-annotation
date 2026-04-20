# Resumo Executivo — Anotação Automática com LLMs

## Visão Geral do Projeto

**Objetivo**: Reduzir custos humanos na anotação de datasets usando múltiplas LLMs open-source com análise de consenso, seguido de fine-tuning supervisionado para classificação de texto.

**Data de atualização**: Abril 2026

---

## Metodologia Implementada

### 1. Anotação Multi-LLM

- Múltiplas LLMs open-source anotam cada instância do dataset
- **Modelos suportados**: Llama (2/3/3.1), Mistral, Mixtral, Gemma (2/3), Phi-3, DeepSeek, Qwen, BLOOMZ
- **Providers**: Ollama (local), HuggingFace API, Groq
- **Anotação redundante**: Cada LLM anota 3x a mesma instância (validação interna)

### 2. Análise de Consenso

Implementação de múltiplas métricas:
- **Cohen's Kappa**: Concordância par a par
- **Fleiss' Kappa**: Concordância geral entre múltiplos anotadores
- **Krippendorff's Alpha**: Concordância robusta
- **Entropia**: Medida de incerteza nas classificações

### 3. Validação de Parâmetros (LLM Hacking)

Testa sistematicamente variações nos parâmetros das LLMs:
- Temperature (0.0, 0.3, 0.5)
- num_predict / max_new_tokens
- Cada modelo tem 3 configurações alternativas pré-definidas

### 4. Estratégias de Resolução de Conflitos

Quando não há consenso claro:
1. **Voto majoritário**: Escolhe classe mais votada
2. **Threshold-based**: Aceita apenas se consenso ≥ X%
3. **Flag for review**: Marca para revisão humana
4. **Remove outliers**: Remove classes da minoria
5. **Weighted voting**: Voto ponderado por confiança
6. **Unanimous only**: Aceita apenas com 100% de acordo

### 5. Fine-Tuning RoBERTa

Pipeline supervisionado com os rótulos de consenso:
- Filtragem de instâncias inválidas (label=-1) com logging
- **Prevenção de data leakage**: remove do teste instâncias que aparecem no treino
- Fine-tuning RoBERTa-base com cross-validation de 5 folds
- Suporte a GPU via CUDA

---

## Estrutura do Sistema

### Módulos Principais

#### Pipeline de Anotação (`src/llm_annotation_system/`)

1. **annotation/llm_annotator.py**
   - Orquestração de múltiplas LLMs
   - Sistema de cache para economizar chamadas
   - Suporte a zero-shot, few-shot, chain-of-thought

2. **consensus/consensus_calculator.py**
   - Extração e agregação de anotações
   - Aplicação da estratégia de resolução configurada
   - Classificação em consenso alto/médio/baixo

3. **consensus/consensus_metrics.py**
   - Cálculo de todas as métricas de concordância
   - Identificação de instâncias problemáticas

#### Pipeline de Fine-Tuning (`src/fine_tune_system/`)

4. **training/splits_aligner.py**
   - Alinhamento de splits HuggingFace com dados anotados
   - **Detecção e remoção de data leakage** via `text_id`

5. **training/cross_validator.py**
   - Implementação de cross-validation 5-fold
   - Treino e avaliação por fold

6. **fine_tune/supervised_fine_tuner.py**
   - Treinamento RoBERTa com HuggingFace Trainer
   - Avaliação com accuracy, precision, recall, F1

#### API REST (`src/api/`)

7. **server.py** (FastAPI)
   - Endpoints para experimentos e fine-tuning
   - Gerenciamento de jobs em background
   - Listagem de datasets disponíveis

### Scripts de Entrada

| Script | Função |
|--------|--------|
| `run_annotation.py` | Pipeline de anotação |
| `run_fine_tunning.py` | Pipeline de fine-tuning |
| `src/api/server.py` | Servidor REST |

### Configurações (`src/config/`)

| Arquivo | Conteúdo |
|---------|----------|
| `llms.py` | 25+ modelos open-source com providers e parâmetros |
| `datasets_collected.py` | 30+ datasets HuggingFace (waashk/) |
| `prompts.py` | Templates de prompts para classificação |
| `conflict_resolution.py` | Estratégias de resolução de conflitos |

---

## Integração com HuggingFace

### Datasets Suportados (30+)

| Tipo | Exemplos |
|------|----------|
| Notícias | AG News, Reuters-90 |
| Sentimento | MPQA, Yelp, SST1/SST2, Movie Reviews |
| Científico | ACM, DBLP, WOS, OHSUMED |
| Web/Social | WebKB, 20 Newsgroups, Twitter, TREC |

### Fluxo com HuggingFace

```python
# 1. Configurar dataset em src/config/datasets_collected.py
# 2. Executar anotação
poetry run python run_annotation.py
# 3. Executar fine-tuning
poetry run python run_fine_tunning.py
```

---

## Outputs Gerados

### Anotação

| Arquivo | Conteúdo |
|---------|----------|
| `dataset_anotado_final.csv` | Dataset completo com rótulos de consenso |
| `annotations_complete.csv` | Todas as anotações detalhadas por modelo |
| `high_confidence_annotations.csv` | Instâncias com consenso ≥ 80% |
| `needs_human_review.csv` | Casos problemáticos |
| `experiment_summary.json` | Estatísticas completas |
| `interactive_dashboard.html` | Dashboard interativo (Plotly) |

### Fine-Tuning

- Checkpoints de modelo por fold (`results/checkpoints/`)
- Métricas por fold (accuracy, F1, kappa)
- Métricas agregadas do experimento

---

## Questões de Pesquisa Abordadas

### Implementado

1. **Consenso entre LLMs diferentes** — tabela completa, métricas estatísticas
2. **Consenso interno de cada LLM** — múltiplas anotações, consistência interna
3. **Impacto de variações de parâmetros** — LLM hacking sistemático
4. **Estratégias para casos sem consenso** — 6 estratégias implementadas
5. **Validação com ground truth** — accuracy automática quando labels disponíveis
6. **Fine-tuning com rótulos de consenso** — RoBERTa com cross-validation
7. **Prevenção de data leakage** — detecção e remoção automática

### Para Discussão

1. Qual threshold de consenso é ideal para o domínio?
2. Quantas LLMs minimizam custo mantendo qualidade do consenso?
3. Few-shot learning melhora concordância entre modelos?
4. Como o fine-tuning com rótulos de consenso compara com ground truth?

---

## Tecnologias Utilizadas

### Core
- **Python 3.9+**, **Poetry**, **Jupyter**

### LLMs
- **Ollama**: Execução local de modelos open-source
- **LangChain**: Abstração de providers LLM
- **HuggingFace Transformers**: Fine-tuning e tokenização
- **Groq API**: Inferência rápida

### Análise
- **pandas, numpy**: Manipulação de dados
- **scikit-learn**: Métricas estatísticas
- **matplotlib, seaborn, plotly**: Visualizações

### API
- **FastAPI**: Servidor REST
- **Pydantic**: Validação de schemas

---

## Métricas de Sucesso

### Anotação
- **Taxa de consenso alto (≥80%)**: % de instâncias confiáveis
- **Cohen's Kappa médio**: concordância geral (>0.6 é bom)
- **Accuracy vs ground truth**: quando labels disponíveis

### Fine-Tuning
- **Accuracy, F1 por fold**: qualidade do modelo treinado
- **Variância entre folds**: estabilidade do treinamento
- **Comparação GT vs consenso**: impacto da qualidade dos rótulos

---

## Checklist do Sistema

### Implementado
- [x] Pipeline de anotação multi-LLM
- [x] Análise de consenso com métricas estatísticas
- [x] 6 estratégias de resolução de conflitos
- [x] Fine-tuning RoBERTa com cross-validation
- [x] Prevenção de data leakage
- [x] Rastreamento de instâncias inválidas (label=-1)
- [x] API REST (FastAPI)
- [x] Suporte a GPU (CUDA)
- [x] Cache de chamadas LLM
- [x] 30+ datasets HuggingFace configurados
- [x] Dashboard interativo

### Em Andamento
- [ ] Validação em escala (1000+ instâncias por dataset)
- [ ] Comparação fine-tuning GT vs consenso publicada
- [ ] Análise de custo-benefício documentada

---

## Referências do Projeto

- [README.md](../README.md) — Visão geral e instalação
- [INSTRUCOES.md](INSTRUCOES.md) — Guia de uso
- [GUIA_DATASETS.md](GUIA_DATASETS.md) — Integração HuggingFace
- `src/config/llms.py` — Todos os modelos disponíveis
- `src/config/datasets_collected.py` — Todos os datasets configurados

---

**Preparado para discussão com orientador**
