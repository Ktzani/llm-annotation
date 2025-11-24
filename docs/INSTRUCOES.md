# 📝 Instruções de Uso do Sistema

## 🎯 O Que Você Tem

Um **sistema completo e profissional** para anotação automática com LLMs. O código está bem estruturado, documentado e pronto para uso em pesquisa.

---

## 📦 Estrutura do Projeto

### Código Principal

**src/config/** - Configurações centralizadas
- `prompts.py` - Templates de prompts otimizados
- `llm_configs.py` - Configuração de todos os modelos LLM
- `experiment.py` - Parâmetros do experimento
- `evaluation.py` - Métricas de avaliação
- `conflict_resolution.py` - Estratégias de resolução de conflitos
- `dataset_config.py` - ⭐ Configuração de datasets HuggingFace

**src/llm_annotation_system/** - Código principal
- `llm_annotator.py` - Classe principal para anotação
- `consensus_analyzer.py` - Análise de consenso e métricas


**src/utils/** - Utilitarios
- `data_loader.py` - 
- `visualizer.py` - Geração de visualizações e dashboards

### Scripts de Execução

- `src/main.py` - Exemplo básico de uso
- `src/main_huggingface.py` - ⭐ Script principal com HuggingFace

### Notebooks

- `src/notebooks/analise_consenso_llms.ipynb` - ⭐ Notebook completo

### Documentação

- `docs/README.md` - Documentação técnica completa
- `docs/RESUMO_EXECUTIVO.md` - Sumário executivo
---

## 🚀 Como Começar

### Passo 1: Instalar Dependências

```bash
poetry install
```

### Passo 2: Configurar API Keys

Crie arquivo `.env` na raiz:

```env
OPENAI_API_KEY=sua-key-openai
ANTHROPIC_API_KEY=sua-key-anthropic
GOOGLE_API_KEY=sua-key-google
```

### Passo 3: Escolher Modo de Uso

#### Opção A: Com Datasets HuggingFace (RECOMENDADO)

```bash
# 1. Descobrir estrutura do seu dataset
poetry run python src/main_huggingface.py --modo descobrir --dataset waashk/seu-dataset

# 2. Configurar em src/config/dataset_config.py
# (use a sugestão gerada pelo comando acima)

# 3. Executar anotação
poetry run python src/main_huggingface.py --modo basico
```

#### Opção B: Com Dados Locais

```bash
# Executar exemplo básico
poetry run python src/main.py
```

#### Opção C: Notebook Jupyter

```bash
# Abrir notebook
poetry run jupyter notebook src/notebooks/analise_consenso_llms.ipynb
```

---

## 📊 O Que o Sistema Faz

### 1. Anotação Automática

- ✅ Múltiplas LLMs anotam cada texto
- ✅ Cada LLM anota múltiplas vezes (validação interna)
- ✅ Total: 15 anotações por instância (5 LLMs × 3 repetições)
- ✅ Sistema de cache (não repete chamadas)

### 2. Análise de Consenso

- ✅ Calcula consenso entre LLMs
- ✅ Calcula consenso interno de cada LLM
- ✅ Identifica casos problemáticos (2-2-1, empates, etc.)
- ✅ Métricas estatísticas completas (Cohen's Kappa, Fleiss', etc.)

### 3. Validação com Ground Truth (Opcional)

- ✅ Se dataset tem labels, valida automaticamente
- ✅ Calcula accuracy, precision, recall, F1
- ✅ Gera classification report
- ✅ Identifica categorias problemáticas

### 4. Validação de Parâmetros

- ✅ Testa diferentes temperaturas
- ✅ Testa diferentes top_p
- ✅ Analisa impacto nas anotações
- ✅ "LLM hacking" para otimização

### 5. Visualizações

- ✅ Heatmap de concordância entre modelos
- ✅ Distribuição de consenso
- ✅ Matriz de confusão
- ✅ Comparação de modelos
- ✅ Dashboard interativo (HTML)

### 6. Outputs Gerados

**CSVs:**
- `dataset_anotado_final.csv` - Dataset final anotado
- `annotations_complete.csv` - Todas as anotações detalhadas
- `high_confidence_annotations.csv` - Consenso ≥ 80%
- `needs_human_review.csv` - Casos problemáticos
- `pairwise_agreement.csv` - Acordo entre pares de modelos
- `confusion_matrix.csv` - Matriz de confusão

**Visualizações:**
- `agreement_heatmap.png` - Heatmap de concordância
- `consensus_distribution.png` - Distribuição de consenso
- `model_comparison.png` - Comparação de modelos
- `interactive_dashboard.html` - ⭐ Dashboard interativo

**Resumos:**
- `experiment_summary.json` - Estatísticas completas

---

## 🤗 Usar Datasets do HuggingFace

### Fluxo Completo

#### 1. Descobrir Estrutura

```bash
poetry run python src/main_huggingface.py --modo descobrir --dataset waashk/seu-dataset
```

**Output:**
```
📋 Estrutura do dataset:
   Colunas: ['text', 'label', 'id']
   
📝 Primeiros 3 exemplos...

💡 Sugestão de configuração:
"seu_dataset": {
    "path": "waashk/seu-dataset",
    "text_column": "text",
    ...
}
```

#### 2. Configurar Dataset

Edite `src/config/dataset_config.py`:

```python
HUGGINGFACE_DATASETS = {
    "meu_dataset": {
        "path": "waashk/nome-do-dataset",
        "text_column": "text",              # Da descoberta
        "label_column": "label",            # Opcional (para validação)
        "categories": None,                  # Extrair automaticamente
        "combine_splits": ["train", "test"], # Dataset completo!
        "sample_size": 100,                  # Começar pequeno
        "description": "Descrição do dataset"
    },
}
```

#### 3. Executar

```bash
poetry run python src/main_huggingface.py --modo basico
```

### Casos de Uso

**Dataset com Labels (Validação):**
```python
"dataset_validacao": {
    "path": "waashk/dataset-com-labels",
    "text_column": "text",
    "label_column": "label",  # ← Tem ground truth
    "categories": None,       # Extrair das labels
    "combine_splits": ["train", "test"],
    "sample_size": None,
}
```
**Resultado:** Sistema calcula accuracy automaticamente!

**Dataset sem Labels (Anotação Pura):**
```python
"dataset_novo": {
    "path": "waashk/textos-novos",
    "text_column": "content",
    "label_column": None,     # ← Sem labels
    "categories": ["A", "B", "C"],  # ← Você define
    "split": "train",
    "sample_size": None,
}
```
**Resultado:** Apenas anotações, sem validação

---

## 💡 Dicas Importantes

### Para Reduzir Custos

1. **Sempre começar com amostra pequena**
   ```python
   "sample_size": 100  # ← Validar antes de escalar
   ```

2. **Usar modelos mais baratos primeiro**
   - Teste com: GPT-3.5, Claude Sonnet, Gemini
   - Depois adicione: GPT-4, Claude Opus

3. **Aproveitar o cache**
   - Sistema salva respostas automaticamente
   - Não repete chamadas de API
   - Economiza ~40% em custos

### Para Melhorar Qualidade

1. **Ajustar prompts** em `src/config/prompts.py`
   - Adicione exemplos (few-shot learning)
   - Teste Chain-of-Thought para casos complexos
   - Seja específico nas instruções

2. **Testar diferentes configurações**
   ```python
   df = annotator.annotate_dataset(
       texts=texts,
       test_param_variations=True  # ← Testa variações
   )
   ```

3. **Analisar casos problemáticos**
   - Arquivo `needs_human_review.csv`
   - Entenda por que não há consenso
   - Ajuste prompts ou categorias conforme necessário

### Para Datasets Grandes

1. **Processar em batches**
   ```python
   batch_size = 500
   for i in range(0, len(texts), batch_size):
       batch = texts[i:i+batch_size]
       # Processar batch...
   ```

2. **Usar cache eficientemente**
   - Cache fica em `data/.cache/huggingface/`
   - Datasets baixados uma vez ficam em cache

---

## 🎓 Material para Apresentação

### Arquivos Prontos

1. **docs/RESUMO_EXECUTIVO.md**
   - Sumário executivo do projeto
   - Metodologia detalhada
   - Resultados esperados

2. **src/notebooks/analise_consenso_llms.ipynb**
   - Execute e gere os resultados
   - Salve com outputs visíveis
   - Use para apresentação

3. **results/figures/interactive_dashboard.html**
   - Dashboard interativo
   - Abra no navegador
   - Demonstre as análises

### Pontos para Discussão

1. **Metodologia Implementada**
   - Multi-LLM com análise de consenso
   - Validação interna por repetições
   - Estratégias de resolução de conflitos

2. **Questões de Pesquisa**
   - Qual threshold ideal de consenso?
   - Como lidar com casos 2-2-1?
   - Few-shot learning melhora resultados?
   - Qual configuração de parâmetros é melhor?

3. **Resultados e Validação**
   - Comparação com ground truth
   - Análise de concordância entre modelos
   - Custos vs qualidade

4. **Próximos Passos**
   - Validar em dataset maior
   - Otimizar custos
   - Preparar publicação

---

## 🔧 Customizações

### Adicionar Novos Modelos

Em `src/config/llm_configs.py`:

```python
LLM_CONFIGS["novo-modelo"] = {
    "provider": "openai",  # ou "anthropic", "google"
    "model_name": "nome-exato-do-modelo",
    "default_params": {
        "temperature": 0.0,
        "max_tokens": 50,
    },
    "alternative_params": {
        "temperature": [0.0, 0.3, 0.5],  # Para testes
    }
}
```

### Customizar Prompts

Em `src/config/prompts.py`:

```python
BASE_ANNOTATION_PROMPT = """
Seu prompt customizado aqui.

**Text to classify:**
{text}

**Available Categories:**
{categories}
"""
```

### Ajustar Parâmetros do Experimento

Em `src/config/experiment.py`:

```python
EXPERIMENT_CONFIG = {
    "num_repetitions_per_llm": 5,      # Mais repetições
    "consensus_threshold": 0.7,         # Threshold diferente
    "test_param_variations": True,      # Testar variações
}
```

---

## 📈 Estimativa de Custos

| Dataset | Chamadas API | Custo Estimado |
|---------|--------------|----------------|
| 100 textos | ~1.500 | $3-5 |
| 1.000 textos | ~15.000 | $30-50 |
| 10.000 textos | ~150.000 | $300-500 |

**Com cache e otimizações:** Redução de ~40%

**Dica:** Comece pequeno, valide metodologia, depois escale.

---

## ✅ Checklist

Antes de executar em produção:

- [ ] Dependências instaladas (`poetry install`)
- [ ] API keys configuradas no `.env`
- [ ] Dataset estruturado ou configurado (`dataset_config.py`)
- [ ] Testado com amostra pequena (`sample_size: 100`)
- [ ] Prompts revisados e otimizados
- [ ] Categorias bem definidas
- [ ] Resultados validados em amostra
- [ ] Entendido custos estimados
- [ ] Backup dos dados importantes

---

