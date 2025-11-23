# Resumo Executivo - Pesquisa em Anotação Automática com LLMs

## 📋 Visão Geral do Projeto

**Objetivo**: Reduzir custos humanos na anotação de datasets usando múltiplas LLMs com análise de consenso.

**Pesquisador**: Gabriel Catizani  
**Data**: Novembro 2025

---

## 🎯 Metodologia Implementada

### 1. Anotação Multi-LLM

- **5 LLMs** diferentes anotam cada instância do dataset
- **Modelos suportados**: GPT-4, GPT-3.5, Claude 3 (Opus/Sonnet), Gemini Pro, Cohere
- **Anotação redundante**: Cada LLM anota 3x a mesma instância (validação interna)

### 2. Análise de Consenso

Implementação de múltiplas métricas:
- **Cohen's Kappa**: Concordância par a par
- **Fleiss' Kappa**: Concordância geral entre múltiplos anotadores
- **Krippendorff's Alpha**: Concordância robusta
- **Hamming Distance**: Distância entre anotações
- **Entropia**: Medida de incerteza nas classificações

### 3. Validação de Parâmetros (LLM Hacking)

Testa sistematicamente se variações nos parâmetros das LLMs afetam os resultados:
- Temperature (0.0, 0.3, 0.5)
- Top-p (0.9, 0.95, 1.0)
- Max tokens

### 4. Estratégias de Resolução de Conflitos

Quando não há consenso claro (ex: empate 2-2-1):
1. **Voto majoritário**: Escolhe classe mais votada
2. **Threshold-based**: Aceita apenas se consenso ≥ X%
3. **Flag for review**: Marca para revisão humana
4. **Remove**: Remove instâncias ambíguas
5. **Weighted voting**: Voto ponderado por confiança do modelo

---

## 💻 Estrutura do Sistema

### Módulos Principais

1. **llm_annotator.py** (370 linhas)
   - Gerenciamento de múltiplas LLMs
   - Sistema de cache para economizar API calls
   - Suporte para diferentes prompts (zero-shot, few-shot, CoT)

2. **consensus_analyzer.py** (280 linhas)
   - Cálculo de todas as métricas de consenso
   - Identificação de instâncias problemáticas
   - Análise de padrões de discordância

3. **visualizer.py** (320 linhas)
   - Heatmaps de concordância
   - Distribuições de consenso
   - Matrizes de confusão
   - Dashboard interativo (Plotly)

4. **config.py** (200 linhas)
   - Prompts otimizados com técnicas de prompt engineering
   - Configurações de todos os modelos
   - Parâmetros do experimento

### Notebook de Análise

**analise_consenso_llms.ipynb**: Notebook completo com:
- Setup e configuração
- Execução passo a passo
- Análises detalhadas
- Visualizações inline
- Interpretação de resultados
- Exportação de dados

---

## 📊 Outputs Gerados

### Dados

1. **annotated_dataset_complete.csv**: Dataset completo com todas anotações
2. **high_confidence_annotations.csv**: Anotações com consenso ≥ 80%
3. **needs_human_review.csv**: Casos problemáticos que precisam revisão
4. **experiment_summary.json**: Sumário estatístico completo

### Métricas

- Matriz de concordância par a par entre todos os modelos
- Estatísticas de consenso por instância
- Identificação de categorias mais confundidas
- Análise de entropia (incerteza nas classificações)

### Visualizações

1. **agreement_heatmap.png**: Concordância entre modelos
2. **consensus_distribution.png**: Distribuição de scores de consenso
3. **confusion_matrix.png**: Matriz de confusão agregada
4. **model_comparison.png**: Comparação de performance
5. **interactive_dashboard.html**: Dashboard interativo completo

---

## 🔬 Questões de Pesquisa Abordadas

### ✅ Implementado

1. **Consenso entre LLMs diferentes**
   - Tabela de consenso completa
   - Métricas de distância e concordância
   - Identificação de casos de alto/médio/baixo consenso

2. **Consenso interno de cada LLM**
   - Múltiplas anotações da mesma instância
   - Cálculo de consistência interna
   - Identificação de modelos mais estáveis

3. **Impacto de variações de parâmetros**
   - Teste sistemático de diferentes configurações
   - Análise de estabilidade
   - "LLM hacking" para encontrar melhores settings

4. **Estratégias para casos sem consenso**
   - Múltiplas abordagens implementadas
   - Comparação de estratégias
   - Recomendações baseadas em métricas

### 🔄 Para Discussão

1. **Threshold ideal de consenso**
   - Qual percentual de consenso é suficiente?
   - Trade-off entre automação e qualidade
   - Depende do domínio e risco do erro

2. **Casos 2-2-1 ou similares**
   - Revisão humana vs. voto majoritário vs. remover
   - Custo-benefício de cada estratégia
   - Validação com ground truth

3. **Few-shot learning**
   - Adicionar exemplos melhora consenso?
   - Quantos exemplos são necessários?
   - Como selecionar bons exemplos?

4. **Otimização de custos**
   - Qual combinação de modelos minimiza custo?
   - É possível usar menos modelos mantendo qualidade?
   - Cache reduz custos significativamente?

---

## 📈 Métricas de Sucesso

### Quantitativas

- **Taxa de consenso alto** (≥80%): Indica % de instâncias confiáveis
- **Cohen's Kappa médio**: Indica concordância geral (>0.6 é bom)
- **Redução de custo humano**: % de instâncias que não precisam revisão
- **Tempo de anotação**: Comparado com anotação manual

### Qualitativas

- **Confiabilidade das anotações**: Validação com ground truth
- **Estabilidade dos modelos**: Variação interna baixa
- **Identificação de casos difíceis**: Sistema detecta ambiguidades

---

## 🚀 Próximos Passos

### Curto Prazo

1. **Validação com ground truth**
   - Comparar anotações automáticas com labels verdadeiros
   - Calcular accuracy, precision, recall
   - Identificar tipos de erros

2. **Otimização de prompts**
   - Testar few-shot learning
   - Comparar diferentes templates
   - Validar Chain-of-Thought

3. **Experimentos com parâmetros**
   - Análise sistemática de impacto
   - Identificar configurações ótimas
   - Documentar trade-offs

### Médio Prazo

1. **Escalar para datasets maiores**
   - Testar com 1000+ instâncias
   - Análise de custos em escala
   - Otimização de performance

2. **Domínios diferentes**
   - Testar em outras tarefas (NER, sumarização, etc.)
   - Avaliar generalização da metodologia
   - Adaptar para casos específicos

3. **Sistema de produção**
   - Pipeline automatizado
   - Interface para revisão humana
   - Monitoramento de qualidade

### Longo Prazo

1. **Publicação**
   - Paper descrevendo metodologia
   - Resultados comparativos
   - Contribuições para a área

2. **Ferramenta open-source**
   - Disponibilizar código
   - Documentação completa
   - Comunidade de usuários

---

## 💰 Análise de Custos (Estimativa)

### Por Instância

- GPT-4 Turbo: ~$0.01 (3 repetições)
- GPT-3.5 Turbo: ~$0.001
- Claude 3 Opus: ~$0.015
- Claude 3 Sonnet: ~$0.003
- Gemini Pro: ~$0.0005

**Total/instância**: ~$0.03 (5 modelos, 3 repetições cada)

### Comparação com Anotação Humana

- Anotador humano: $0.10-0.50/instância
- **Economia potencial**: 80-90% se consenso ≥ 80%
- **ROI**: Positivo a partir de 1000+ instâncias

### Otimizações Possíveis

- Cache reduz custos em ~40%
- Usar apenas 3 modelos: -40% custo
- Modelos mais baratos primeiro: -60% custo

---

## 📚 Material para Apresentação

### Para o Orientador

1. **Este resumo executivo**
2. **Notebook completo**: `analise_consenso_llms.ipynb`
3. **Dashboard interativo**: Visualização dinâmica dos resultados
4. **Sumário JSON**: Métricas quantitativas

### Para Banca/Publicação

1. Metodologia detalhada
2. Resultados experimentais
3. Comparação com baselines
4. Análise de custos
5. Código open-source

---

## 🎓 Contribuições Científicas

1. **Metodologia sistemática** para anotação com múltiplas LLMs
2. **Framework de análise de consenso** com múltiplas métricas
3. **Estratégias de resolução de conflitos** validadas empiricamente
4. **Análise de custo-benefício** de diferentes abordagens
5. **Sistema completo e reproduzível** disponível open-source

---

## 📞 Questões para Discussão

1. Qual threshold de consenso devemos usar como padrão?
2. Vale a pena investir em few-shot learning?
3. Como validar em domínios específicos?
4. Estratégia de publicação (venue, timing)?
5. Possibilidade de parceria com empresas?

---

**Preparado por**: Gabriel Catizani  
**Data**: Novembro 2025  
**Contato**: [seu-email]

---

## ✅ Checklist de Entrega

- [x] Sistema completo implementado
- [x] Notebook de análise documentado
- [x] Visualizações geradas
- [x] README com instruções
- [x] Código modular e bem estruturado
- [x] Exemplos de uso
- [ ] Validação com ground truth
- [ ] Análise de custos real
- [ ] Comparação com baselines
