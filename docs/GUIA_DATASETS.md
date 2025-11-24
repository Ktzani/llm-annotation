# 🤗 Guia de Uso com Datasets do HuggingFace

Este guia mostra como integrar seus datasets do HuggingFace (waashk) com o sistema de anotação.

---

## 🚀 Início Rápido (3 Passos)

### 1. Instalar Dependências

```bash
poetry install
```

### 2. Descobrir Estrutura dos Seus Datasets

```bash
python src/main_huggingface.py --modo descobrir --dataset waashk/seu-dataset
```

Isso mostra:
- ✅ Colunas disponíveis
- ✅ Features
- ✅ Exemplos
- ✅ Sugestão de configuração

### 3. Configurar e Executar

Edite `src/config/dataset_config.py` e adicione seu dataset:

```python
HUGGINGFACE_DATASETS = {
    "meu_dataset": {
        "path": "waashk/nome-do-dataset",
        "text_column": "text",
        "label_column": "label",  # ou None
        "categories": ["Cat1", "Cat2"],
        "split": "train",
        "sample_size": 100,  # Começar pequeno!
    },
}
```

Execute:

```bash
python src/main_huggingface.py --modo basico
```

---

## 📋 Configuração Detalhada

### Estrutura de Configuração

```python
"nome_dataset": {
    # OBRIGATÓRIOS
    "path": str,           # Path no HuggingFace
    "text_column": str,    # Coluna com textos
    "split": str,          # "train", "test", etc
    
    # OPCIONAIS
    "label_column": str,   # Coluna com labels (para validação)
    "categories": list,    # ou None (extrair automaticamente)
    "sample_size": int,    # ou None (carregar tudo)
    "description": str,    # Descrição do dataset
    
    # AVANÇADO: Combinar múltiplos splits
    "combine_splits": ["train", "test"],  # Usar dataset completo
}
```

### Exemplo Real

```python
"sentiment_reviews": {
    "path": "waashk/sentiment-reviews",
    "text_column": "review_text",
    "label_column": "sentiment",
    "categories": None,  # Extrair automaticamente
    "split": "train",
    "sample_size": 500,
    "description": "Reviews de produtos para análise de sentimento"
}
```

---

## 🎯 Casos de Uso

### Caso 1: Dataset com Labels (Validação)

Você tem labels para validar a qualidade:

```python
"dataset_validacao": {
    "path": "waashk/dataset-com-labels",
    "text_column": "text",
    "label_column": "label",  # ← Importante!
    "categories": None,  # Extrair das labels
    "split": "train",
    "sample_size": None,
}
```

O sistema automaticamente:
- ✅ Extrai categorias dos labels
- ✅ Calcula accuracy vs ground truth
- ✅ Gera relatório de validação

### Caso 2: Dataset sem Labels (Anotação Pura)

Você quer anotar do zero:

```python
"dataset_anotacao": {
    "path": "waashk/textos-nao-rotulados",
    "text_column": "content",
    "label_column": None,  # ← Sem labels
    "categories": ["Spam", "Ham", "Unsure"],  # ← Você define
    "split": "train",
    "sample_size": None,
}
```

### Caso 3: Dataset Completo (Todos os Splits)

Usar dataset inteiro para anotação:

```python
"dataset_completo": {
    "path": "waashk/meu-dataset",
    "text_column": "text",
    "label_column": None,
    "categories": ["A", "B", "C"],
    "combine_splits": ["train", "test", "validation"],  # ← Combinar!
    "sample_size": None,
}
```

### Caso 4: Amostra Pequena (Teste)

Começar com amostra para testar:

```python
"dataset_teste": {
    "path": "waashk/dataset-grande",
    "text_column": "text",
    "label_column": "category",
    "categories": None,
    "split": "train",
    "sample_size": 50,  # ← Apenas 50 para teste!
}
```

---

## 💻 Exemplos de Código

### Exemplo 1: Básico

```python
from dataset_config import load_hf_dataset
from llm_annotator import LLMAnnotator

# Carregar dataset
texts, categories, ground_truth = load_hf_dataset("meu_dataset")

# Configurar
api_keys = {...}
models = ["gpt-4-turbo", "claude-3-opus", "gemini-pro"]

# Anotar
annotator = LLMAnnotator(models, categories, api_keys)
df = annotator.annotate_dataset(texts, num_repetitions=3)
df = annotator.calculate_consensus(df)

# Validar (se houver ground truth)
if ground_truth:
    df['ground_truth'] = ground_truth
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(df['ground_truth'], df['most_common_annotation'])
    print(f"Accuracy: {acc:.2%}")
```

### Exemplo 2: Dataset Customizado

```python
from dataset_config import load_custom_dataset

# Carregar sem pré-configurar
texts, categories, labels = load_custom_dataset(
    hf_path="waashk/dataset-qualquer",
    text_column="minha_coluna",
    label_column=None,
    categories=["X", "Y", "Z"],
    combine_splits=["train", "test"],  # Dataset completo
    sample_size=100
)
```

### Exemplo 3: Como DataFrame

```python
from dataset_config import load_hf_dataset_as_dataframe

# Carregar como pandas DataFrame
df = load_hf_dataset_as_dataframe("meu_dataset")

# Análise exploratória
print(df.head())
print(df['ground_truth'].value_counts())

# Usar com anotador
texts = df['text'].tolist()
```

---

## 🔍 Descobrir Estrutura

Não sabe a estrutura do seu dataset?

### Método 1: Via Script

```bash
python src/main_huggingface.py --modo descobrir --dataset waashk/seu-dataset
```

### Método 2: Via Código

```python
from dataset_config import discover_dataset_structure

discover_dataset_structure("waashk/seu-dataset", num_examples=5)
```

Isso mostra:
```
📋 Estrutura do dataset:
   Colunas: ['text', 'label', 'id']
   Features: {'text': Value(dtype='string'), 'label': ClassLabel(...)}

📝 Primeiros 3 exemplos:
   Exemplo 1:
      text: Este é um texto exemplo...
      label: positivo
      id: 1
```

---

## 🏃 Executar

### Modo Básico

```bash
python src/main_huggingface.py --modo basico
```

Executa fluxo completo:
1. Carrega dataset configurado
2. Anota com múltiplas LLMs
3. Calcula consenso
4. Valida com ground truth (se disponível)
5. Gera visualizações
6. Salva resultados

### Modo Descobrir

```bash
python src/main_huggingface.py --modo descobrir --dataset waashk/seu-dataset
```

Descobre estrutura do dataset.

### Modo Customizado

```bash
python src/main_huggingface.py --modo customizado
```

Exemplo de carregamento sem pré-configurar.

### Modo Múltiplos

```bash
python src/main_huggingface.py --modo multiplos
```

Processa vários datasets em batch.

---

## 📁 Estrutura de Arquivos

```
src/
├── config/
│   └── dataset_config.py       ⭐ Configuração de datasets
├── llm_annotation_system/
│   ├── llm_annotator.py
│   └── ...
├── main.py                     Original (exemplo simples)
└── main_huggingface.py         ⭐ Novo (com HuggingFace)

data/
└── .cache/
    └── huggingface/            Cache local dos datasets

results/
├── dataset_anotado_final.csv   ⭐ Dataset anotado
├── figures/
└── ...
```

---

## 🐛 Troubleshooting

### Erro: "Column not found"

```python
# Verificar colunas disponíveis
python src/main_huggingface.py --modo descobrir --dataset waashk/seu-dataset

# Ajustar config
"text_column": "nome_correto_da_coluna"
```

### Erro: "Dataset not found"

```bash
# Verificar se dataset existe
# Ir em: https://huggingface.co/waashk

# Se for privado, fazer login
huggingface-cli login
```

### Dataset muito grande

```python
# Usar amostragem
"sample_size": 1000  # Apenas 1000 exemplos

# Ou processar em batches
for i in range(0, total, 1000):
    texts = load_dataset(...).select(range(i, i+1000))
```

### Combinar splits não funciona

```python
# Verificar splits disponíveis primeiro
discover_dataset_structure("waashk/seu-dataset")

# Ajustar lista
"combine_splits": ["train", "test"]  # Apenas os que existem
```

---

## 💡 Dicas

### 1. Sempre Começar Pequeno

```python
"sample_size": 100  # ← Começar com 100 textos
```

Validar que funciona, depois aumentar!

### 2. Cache Local

Datasets são salvos em cache:

```bash
# Ver cache
ls -lh data/.cache/huggingface/

# Limpar se necessário
rm -rf data/.cache/huggingface/
```

### 3. Validação com Ground Truth

Se seu dataset tem labels:

```python
# Sistema automaticamente:
# 1. Calcula accuracy
# 2. Gera classification report
# 3. Identifica categorias problemáticas
```

### 4. Processar em Batches

Para datasets grandes:

```python
# Dividir em partes
batch_size = 500
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    # Processar batch...
```

---

## ✅ Checklist

Antes de começar:

- [ ] `poetry install` executado
- [ ] Datasets identificados em https://huggingface.co/waashk
- [ ] Estrutura descoberta com `--modo descobrir`
- [ ] Configuração adicionada em `dataset_config.py`
- [ ] Testado com amostra pequena (`sample_size: 100`)
- [ ] API keys configuradas no `.env`
- [ ] Pronto para anotação completa! 🚀

---

## 📊 Fluxo Completo

```
1. DESCOBRIR           → python ... --modo descobrir
   ↓
2. CONFIGURAR          → Editar dataset_config.py
   ↓
3. TESTAR (amostra)    → sample_size: 100
   ↓
4. VALIDAR             → Verificar resultados
   ↓
5. ESCALAR             → sample_size: None (tudo)
   ↓
6. ANALISAR            → Ver dashboard e métricas
```

---

## 🎓 Recursos

- **HuggingFace Datasets**: https://huggingface.co/docs/datasets/
- **Seus datasets**: https://huggingface.co/waashk
- **Documentação do projeto**: README.md

---

**Boa sorte com suas anotações!** 🤗🚀
