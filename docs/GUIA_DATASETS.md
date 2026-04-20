# Guia de Datasets HuggingFace

Este guia explica como configurar e usar datasets do HuggingFace (namespace `waashk`) com o sistema de anotação.

---

## Início Rápido

### 1. Instalar Dependências

```bash
poetry install
```

### 2. Configurar Dataset

Edite `src/config/datasets_collected.py` e adicione (ou ajuste) seu dataset:

```python
DATASETS = {
    "meu_dataset": {
        "path": "<hf-user>/nome-do-dataset",
        "text_column": "text",
        "label_column": "label",   # ou None
        "categories": None,        # extrair automaticamente das labels
        "combine_splits": ["train", "test"],
        "sample_size": 100,        # começar pequeno
    },
}
```

### 3. Executar Anotação

```bash
poetry run python run_annotation.py
```

### 4. Executar Fine-Tuning

```bash
poetry run python run_fine_tunning.py
```

---

## Configuração Detalhada

### Campos do Dataset

```python
"nome_dataset": {
    # OBRIGATÓRIOS
    "path": str,                     # Path no HuggingFace (ex: "waashk/ag-news")
    "text_column": str,              # Coluna com textos

    # OPCIONAIS
    "label_column": str | None,      # Coluna com labels (None = sem ground truth)
    "categories": list | None,       # None = extrair automaticamente das labels
    "sample_size": int | None,       # None = carregar tudo
    "description": str,              # Descrição do dataset

    # PARA COMBINAR MÚLTIPLOS SPLITS
    "combine_splits": ["train", "test"],   # Combinar train + test
    # OU usar split único
    "split": "train",
}
```

---

## Casos de Uso

### Dataset com Labels (Validação)

```python
"dataset_validacao": {
    "path": "waashk/ag-news",
    "text_column": "text",
    "label_column": "label",    # ← com ground truth
    "categories": None,         # extrair automaticamente
    "combine_splits": ["train", "test"],
    "sample_size": None,
}
```

O sistema calcula accuracy, precision, recall e F1 automaticamente.

### Dataset sem Labels (Anotação Pura)

```python
"dataset_anotacao": {
    "path": "waashk/textos-novos",
    "text_column": "content",
    "label_column": None,               # ← sem ground truth
    "categories": ["Spam", "Ham"],      # ← você define as categorias
    "split": "train",
    "sample_size": None,
}
```

### Amostra para Teste

```python
"dataset_teste": {
    "path": "waashk/dataset-grande",
    "text_column": "text",
    "label_column": "category",
    "categories": None,
    "split": "train",
    "sample_size": 50,    # ← apenas 50 para validar rapidamente
}
```

---

## Datasets Pré-Configurados

O arquivo `src/config/datasets_collected.py` já contém 30+ datasets configurados:

| Tipo | Datasets |
|------|----------|
| **Notícias** | `ag_news`, `reuters90` |
| **Sentimento** | `mpqa`, `yelp_2013`, `yelp_2015`, `movie_reviews`, `sst1`, `sst2`, `vader_movie` |
| **Científico** | `acm`, `dblp`, `wos_11967`, `wos_5736`, `ohsumed` |
| **Web/Social** | `webkb`, `20newsgroups`, `twitter`, `trec` |
| **Médico** | `medline` |
| **Livros** | `books`, `pang_movie` |

Para usar um dataset já configurado, basta referenciar a chave no script de anotação.

---

## Prevenção de Data Leakage no Fine-Tuning

Ao executar `run_fine_tunning.py`, o sistema automaticamente:

1. Gera um `text_id` único para cada instância (hash do texto)
2. Identifica instâncias presentes tanto no treino quanto no teste/validação
3. Remove essas instâncias do conjunto de avaliação
4. Emite um warning com a contagem de instâncias removidas:

```
WARNING: 42 instâncias do teste aparecem no treino e foram removidas (data leakage)
```

Isso garante que métricas de avaliação reflitam capacidade real de generalização.

---

## Instâncias Inválidas

Quando uma LLM retorna uma resposta fora das categorias configuradas, a instância recebe `label = -1`. Antes do fine-tuning, o sistema:

1. **Loga** todas as instâncias com label=-1 para depuração
2. **Remove** essas instâncias antes do treinamento

Monitore o log para identificar padrões de falha de extração (pode indicar problemas no prompt ou nas categorias).

---

## Cache Local

Datasets baixados do HuggingFace ficam em cache:

```
data/.cache/huggingface/
```

Na segunda execução, o dataset é carregado do cache local — sem re-download.

Para limpar:

```bash
rm -rf data/.cache/huggingface/
```

---

## Troubleshooting

### "Column not found"

Verifique as colunas disponíveis no dataset:

```python
from datasets import load_dataset
ds = load_dataset("waashk/seu-dataset")
print(ds["train"].column_names)
```

Ajuste `text_column` e `label_column` conforme o resultado.

### "Dataset not found"

```bash
# Verificar se dataset é privado
huggingface-cli login
# Ou verificar em: https://huggingface.co/waashk
```

### "combine_splits não funciona"

Verifique quais splits existem no dataset:

```python
from datasets import load_dataset
ds = load_dataset("waashk/seu-dataset")
print(list(ds.keys()))   # ['train', 'test', ...]
```

Use apenas splits que existem em `combine_splits`.

### Desempenho lento

Para datasets grandes, use `sample_size` para testes iniciais:

```python
"sample_size": 500   # processar apenas 500 instâncias
```

---

## Checklist

Antes de executar anotação:

- [ ] Dataset identificado em https://huggingface.co/<hf-user>
- [ ] Configuração adicionada em `datasets_collected.py`
- [ ] Colunas `text_column` e `label_column` verificadas
- [ ] Testado com `sample_size: 100`
- [ ] API keys / Ollama configurados no `.env`

---

## Recursos

- **HuggingFace Datasets Docs**: https://huggingface.co/docs/datasets/
- **Seus datasets**: https://huggingface.co/waashk
- [INSTRUCOES.md](INSTRUCOES.md) — Guia completo do sistema
- [README.md](../README.md) — Visão geral e instalação
