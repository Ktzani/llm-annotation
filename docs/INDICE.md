# 📂 Índice do Sistema de Anotação Automática com LLMs

## 📁 Estrutura de Arquivos

```
llm_annotation_system/
│
├── 📘 DOCUMENTAÇÃO
│   ├── README.md                    (6 KB) - Documentação completa
│   ├── QUICKSTART.md                (3 KB) - Guia rápido
│   ├── RESUMO_EXECUTIVO.md          (9 KB) - Para orientador ⭐
│   └── INSTRUCOES_VICTOR.md         (9 KB) - Suas instruções ⭐
│
├── 💻 CÓDIGO PRINCIPAL
│   ├── config.py                    (6 KB) - Configurações e prompts
│   ├── llm_annotator.py            (17 KB) - Anotador principal
│   ├── consensus_analyzer.py       (16 KB) - Análise de consenso
│   └── visualizer.py               (18 KB) - Visualizações
│
├── 📓 NOTEBOOKS E SCRIPTS
│   ├── analise_consenso_llms.ipynb (27 KB) - Notebook principal ⭐⭐⭐
│   └── exemplo_uso.py               (5 KB) - Script de exemplo
│
└── 📦 DEPENDÊNCIAS
    └── requirements.txt             (373 B) - Pacotes necessários

Total: 11 arquivos (116 KB)
```

---

## 🎯 Por Onde Começar?

### 1️⃣ LEIA PRIMEIRO

```
📄 INSTRUCOES_VICTOR.md
```
→ Instruções específicas para você, Gabriel Catizani
→ Como configurar, executar e apresentar
→ Dicas e troubleshooting

### 2️⃣ DEPOIS LEIA

```
📄 QUICKSTART.md
```
→ Guia rápido de instalação e uso
→ 3 opções de execução
→ Próximos passos

### 3️⃣ EXECUTE

```
📓 analise_consenso_llms.ipynb
```
→ **ARQUIVO PRINCIPAL**
→ Notebook completo com toda análise
→ Use para apresentar ao orientador

### 4️⃣ APRESENTE

```
📄 RESUMO_EXECUTIVO.md
```
→ Material para o orientador
→ Metodologia, resultados, próximos passos
→ Questões para discussão

---

## 📚 Descrição Detalhada

### 📘 Documentação

#### README.md (6 KB)
- Visão geral do projeto
- Metodologia implementada
- Guia de instalação completo
- Estrutura do projeto
- FAQ e troubleshooting
- Estimativas de custo

#### QUICKSTART.md (3 KB)
- Instalação em 4 passos
- 3 opções de uso
- Verificação de resultados
- Dicas de otimização
- Troubleshooting básico

#### RESUMO_EXECUTIVO.md (9 KB)
- **Para apresentar ao orientador**
- Visão geral e objetivos
- Metodologia detalhada
- Módulos implementados
- Outputs gerados
- Questões de pesquisa
- Métricas de sucesso
- Análise de custos
- Próximos passos
- Checklist de entrega

#### INSTRUCOES_VICTOR.md (9 KB)
- **Suas instruções específicas**
- O que você tem agora
- Como começar passo a passo
- Dicas importantes
- Material para orientador
- Email sugerido para Celso/Washington
- Customizações possíveis
- Estimativas de custo
- Checklist de validação

---

### 💻 Código Principal

#### config.py (6 KB)
**O que tem:**
- 3 templates de prompts otimizados
  - BASE_ANNOTATION_PROMPT (zero-shot)
  - FEW_SHOT_PROMPT (com exemplos)
  - COT_PROMPT (Chain-of-Thought)
- Configurações de 5 LLMs
  - GPT-4 Turbo, GPT-3.5 Turbo
  - Claude 3 Opus, Claude 3 Sonnet
  - Gemini Pro
- Parâmetros do experimento
- Estratégias de resolução de conflitos
- Métricas de avaliação

**Quando usar:**
- Customizar prompts
- Adicionar novos modelos
- Ajustar parâmetros

#### llm_annotator.py (17 KB)
**O que tem:**
- Classe `LLMAnnotator` (370 linhas)
- Gerencia múltiplas LLMs
- Sistema de cache inteligente
- Suporte para diferentes prompts
- Anotação com repetições
- Teste de variações de parâmetros

**Métodos principais:**
- `annotate_dataset()` - Anota dataset completo
- `annotate_single()` - Anota um texto
- `calculate_consensus()` - Calcula consenso

#### consensus_analyzer.py (16 KB)
**O que tem:**
- Classe `ConsensusAnalyzer` (280 linhas)
- Calcula todas as métricas
  - Cohen's Kappa
  - Fleiss' Kappa
  - Krippendorff's Alpha
  - Hamming Distance
  - Jaccard Similarity
- Identifica instâncias problemáticas
- Analisa padrões de discordância

**Métodos principais:**
- `generate_consensus_report()` - Relatório completo
- `calculate_pairwise_agreement()` - Concordância par a par
- `identify_difficult_instances()` - Casos problemáticos

#### visualizer.py (18 KB)
**O que tem:**
- Classe `ConsensusVisualizer` (320 linhas)
- Gera todos os gráficos
  - Heatmap de concordância
  - Distribuição de consenso
  - Matriz de confusão
  - Comparação de modelos
  - Dashboard interativo (Plotly)

**Métodos principais:**
- `plot_agreement_heatmap()` - Matriz de concordância
- `plot_consensus_distribution()` - Distribuição
- `create_interactive_dashboard()` - Dashboard HTML

---

### 📓 Notebooks e Scripts

#### analise_consenso_llms.ipynb (27 KB) ⭐⭐⭐
**ARQUIVO PRINCIPAL PARA USO**

**Estrutura:**
1. Setup e Imports
2. Configuração de API Keys
3. Carregar Dataset
4. Configurar Modelos LLM
5. Inicializar Anotador
6. Executar Anotação
   - Com parâmetros padrão
   - Com variações de parâmetros
7. Calcular Consenso
8. Análise Detalhada
   - Métricas de distância
   - Matriz de concordância
9. Instâncias Problemáticas
   - Identificação
   - Estratégias de resolução
10. Visualizações
11. Análise de Parâmetros
12. Sumário e Recomendações
13. Exportar Resultados
14. Conclusões

**Use para:**
- Executar análise completa
- Apresentar ao orientador
- Gerar todos os outputs

#### exemplo_uso.py (5 KB)
**Script simplificado**

Demonstra uso básico:
- Configuração mínima
- Anotação de 10 textos
- Análise de consenso
- Visualizações
- Exportação de resultados

**Use para:**
- Teste rápido
- Entender o fluxo
- Execução automatizada

---

### 📦 Dependências

#### requirements.txt (373 B)
Pacotes necessários:
- pandas, numpy (dados)
- matplotlib, seaborn, plotly (visualização)
- openai, anthropic, google-generativeai (LLMs)
- scikit-learn, scipy (métricas)
- jupyter (notebooks)

**Instalação:**
```bash
pip install -r requirements.txt
```

---

## 🎨 Outputs Gerados

Quando você executar o sistema, ele criará:

```
results/
├── annotations_complete.csv          # Todas anotações
├── pairwise_agreement.csv           # Concordância entre modelos
├── pairwise_kappa.csv               # Cohen's Kappa
├── confusion_matrix.csv             # Matriz de confusão
├── most_confused_pairs.csv          # Pares confundidos
├── difficult_instances.csv          # Casos problemáticos
│
├── figures/                         # Visualizações
│   ├── agreement_heatmap.png        # Heatmap
│   ├── consensus_distribution.png   # Distribuição
│   ├── confusion_matrix.png         # Confusão
│   ├── model_comparison.png         # Comparação
│   ├── *_parameter_impact.png       # Impacto de parâmetros
│   └── interactive_dashboard.html   # Dashboard ⭐
│
└── final/                           # Resultados finais
    ├── annotated_dataset_complete.csv      # Completo
    ├── high_confidence_annotations.csv     # Alta confiança
    ├── needs_human_review.csv              # Para revisão
    └── experiment_summary.json             # Sumário JSON
```

---

## 🚀 Fluxo de Trabalho Recomendado

```
1. Ler INSTRUCOES_VICTOR.md
   ↓
2. Instalar dependências (requirements.txt)
   ↓
3. Configurar API keys (.env)
   ↓
4. Abrir analise_consenso_llms.ipynb
   ↓
5. Executar célula por célula
   ↓
6. Analisar resultados gerados
   ↓
7. Ler RESUMO_EXECUTIVO.md
   ↓
8. Preparar apresentação para orientador
```

---

## 💡 Dicas de Uso

### Para Economia
- ✅ Comece com amostra pequena (10-20 textos)
- ✅ Use cache (ativado por padrão)
- ✅ Teste com modelos mais baratos primeiro

### Para Qualidade
- ✅ Ajuste prompts em config.py
- ✅ Teste variações de parâmetros
- ✅ Analise casos problemáticos

### Para Apresentação
- ✅ Execute notebook completo
- ✅ Gere dashboard interativo
- ✅ Prepare RESUMO_EXECUTIVO.md
- ✅ Documente seus achados

---

## 📞 Suporte

**Dúvidas sobre:**
- Instalação → QUICKSTART.md
- Uso → analise_consenso_llms.ipynb
- Código → README.md
- Apresentação → RESUMO_EXECUTIVO.md

---

## ✅ Checklist Rápido

- [ ] Li INSTRUCOES_VICTOR.md
- [ ] Instalei dependências
- [ ] Configurei API keys
- [ ] Executei notebook de teste
- [ ] Analisei resultados
- [ ] Preparei material para orientador

---

**Pronto para começar! 🎉**

Você tem tudo que precisa para:
✅ Executar a pesquisa
✅ Analisar resultados
✅ Apresentar ao orientador
✅ Publicar paper

Boa sorte, Gabriel Catizani! 🚀
