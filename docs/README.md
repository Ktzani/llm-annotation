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
- **Random**: Seleção aleatória entre top 2

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- API keys para as LLMs que deseja usar

### Instalação de Dependências

```bash
# Instalar dependências
pip install -r requirements.txt
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
jupyter notebook analise_consenso_llms.ipynb
```

Este notebook contém:
- Setup completo
- Exemplos de uso
- Análises detalhadas
- Visualizações
- Interpretação de resultados

### Opção 2: Script de Exemplo

```bash
python exemplo_uso.py
```

### Opção 3: Uso Programático

```python
from llm_annotator import LLMAnnotator
from consensus_analyzer import ConsensusAnalyzer

# Configurar
api_keys = {"openai": "...", "anthropic": "...", "google": "..."}
models = ["gpt-4-turbo", "claude-3-opus", "gemini-pro"]
categories = ["Positivo", "Negativo", "Neutro"]

# Anotar
annotator = LLMAnnotator(models, categories, api_keys)
df = annotator.annotate_dataset(texts, num_repetitions=3)
df = annotator.calculate_consensus(df)
```

## 📁 Estrutura do Projeto

```
├── config.py                      # Configurações e prompts
├── llm_annotator.py              # Classe principal de anotação
├── consensus_analyzer.py         # Análise de consenso
├── visualizer.py                 # Visualizações
├── exemplo_uso.py                # Script de exemplo
├── analise_consenso_llms.ipynb   # Notebook principal ⭐
├── requirements.txt              # Dependências
└── README.md                     # Este arquivo
```

## 📖 Guia Detalhado

### Customizar Prompts

Edite `config.py`:

```python
BASE_ANNOTATION_PROMPT = """You are an expert...
{text}
{categories}
"""
```

### Adicionar Novos Modelos

Em `config.py`:

```python
LLM_CONFIGS["seu-modelo"] = {
    "provider": "openai",
    "model_name": "nome-do-modelo",
    "default_params": {"temperature": 0.0}
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

1. **annotated_dataset_complete.csv**: Todas anotações
2. **high_confidence_annotations.csv**: Consenso ≥ 80%
3. **needs_human_review.csv**: Casos problemáticos
4. **experiment_summary.json**: Estatísticas

## 💡 FAQ

**Q: Quantos modelos usar?**  
A: Recomendamos 5 modelos de diferentes provedores.

**Q: Quantas repetições?**  
A: 3 repetições é um bom balanço entre confiabilidade e custo.

**Q: Como reduzir custos?**  
A: Use cache, amostras pequenas, e modelos mais baratos inicialmente.

**Q: O que fazer com casos sem consenso?**  
A: Depende do caso - revisão humana é mais confiável, voto majoritário é mais rápido.

## 📧 Próximos Passos

1. [ ] Configurar API keys em `.env`
2. [ ] Preparar seu dataset
3. [ ] Abrir o notebook `analise_consenso_llms.ipynb`
4. [ ] Seguir o passo a passo no notebook
5. [ ] Analisar resultados e visualizações
6. [ ] Apresentar para orientador

## 🙏 Agradecimentos

Orientador e colaboradores: Marcos, Celso, Washington

---

**⭐ Boa sorte com sua pesquisa!**
