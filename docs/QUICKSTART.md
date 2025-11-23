# 🚀 Guia de Início Rápido

## ⏱️ Em 5 Minutos

### 1. Instalação (2 minutos)

```bash
# Clone o repositório
git clone <seu-repo>
cd anotacao-automatica-llms

# Instale dependências
pip install -r requirements.txt
```

### 2. Configure API Keys (1 minuto)

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="sua_chave_aqui"

# Windows
set ANTHROPIC_API_KEY=sua_chave_aqui
```

### 3. Execute o Exemplo (2 minutos)

```bash
python exemplo_uso.py
```

**Pronto!** Você acabou de anotar seu primeiro dataset usando LLMs! 🎉

---

## 📊 Para Usar Seu Dataset

### Opção A: Linha de Comando

```bash
# 1. Crie a configuração
python main.py --create-default-config

# 2. Execute com seu dataset
python main.py \
  --dataset meu_dataset.json \
  --config experiment_config.json \
  --categories "categoria1" "categoria2" "categoria3"
```

### Opção B: Notebook Jupyter (Recomendado)

```bash
# 1. Inicie o Jupyter
jupyter notebook

# 2. Abra: experiment_notebook.ipynb

# 3. Modifique para seu dataset e execute!
```

---

## 📝 Formato do Dataset

Seu dataset deve estar em JSON ou JSONL:

**JSON:**
```json
[
  {"id": "1", "text": "Texto para anotar..."},
  {"id": "2", "text": "Outro texto..."}
]
```

**JSONL:**
```jsonl
{"id": "1", "text": "Texto para anotar..."}
{"id": "2", "text": "Outro texto..."}
```

---

## 🎯 Personalize o Prompt

Edite `config.py` e modifique `ANNOTATION_PROMPT_TEMPLATE`:

```python
MEU_PROMPT = """Você é um especialista em [SUA TAREFA].

**Categorias:**
{categories}

**Texto:**
{text}

**Classificação:**"""
```

---

## 📊 Onde Estão os Resultados?

Após a execução, confira:

```
./results/<nome_experimento>/
├── results.xlsx          # ← Abra este primeiro!
├── visualizations/       # ← Gráficos PNG
├── annotations.json      # ← Anotações detalhadas
└── report.txt           # ← Relatório em texto
```

---

## 🆘 Problemas Comuns

### "API key não encontrada"
```bash
# Configure a variável de ambiente:
export ANTHROPIC_API_KEY="sua_chave"
```

### "Módulo não encontrado"
```bash
# Instale as dependências:
pip install -r requirements.txt
```

### "Rate limit exceeded"
- Aguarde alguns minutos
- Ou reduza o número de instâncias no teste

---

## 📚 Próximos Passos

1. ✅ Execute o exemplo básico
2. ✅ Abra o notebook e explore
3. ✅ Teste com seu dataset pequeno
4. ✅ Ajuste o prompt
5. ✅ Execute experimento completo
6. ✅ Analise os resultados

---

## 🤝 Precisa de Ajuda?

1. Leia o README.md completo
2. Explore o notebook com exemplos
3. Confira os comentários no código
4. Entre em contato com o autor

---

**Boa pesquisa! 🚀📊**
