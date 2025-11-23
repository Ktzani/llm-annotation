# 📝 Instruções Específicas para Gabriel Catizani

## 🎯 O Que Você Tem Agora

Implementei um **sistema completo e profissional** para sua pesquisa em anotação automática com LLMs. O código está bem estruturado, documentado e pronto para apresentar ao seu orientador.

---

## 📦 Arquivos Entregues

### Código Principal (4 módulos)

1. **config.py** (6.2 KB)
   - Prompts otimizados com prompt engineering
   - Configurações de todos os modelos LLM
   - Parâmetros do experimento
   - Estratégias de resolução de conflitos

2. **llm_annotator.py** (17 KB)
   - Classe principal LLMAnnotator
   - Gerencia múltiplas LLMs simultaneamente
   - Sistema de cache para economizar API calls
   - Suporte para diferentes prompts e parâmetros

3. **consensus_analyzer.py** (16 KB)
   - Classe ConsensusAnalyzer
   - Calcula todas as métricas (Cohen's Kappa, Fleiss, etc.)
   - Identifica instâncias problemáticas
   - Gera relatório completo

4. **visualizer.py** (18 KB)
   - Classe ConsensusVisualizer
   - Gera todos os gráficos
   - Dashboard interativo com Plotly
   - Exporta em múltiplos formatos

### Notebooks e Scripts

5. **analise_consenso_llms.ipynb** (27 KB) ⭐ **PRINCIPAL**
   - Notebook completo com análise passo a passo
   - Explicações detalhadas
   - Visualizações inline
   - Interpretação de resultados
   - **Use este para apresentar ao orientador**

6. **exemplo_uso.py** (4.5 KB)
   - Script de exemplo pronto para executar
   - Demonstra uso completo do sistema

### Documentação

7. **README.md** (6 KB)
   - Documentação completa do projeto
   - Guia de instalação e uso
   - FAQ e troubleshooting

8. **QUICKSTART.md** (2.7 KB)
   - Guia rápido para começar
   - 3 opções de uso
   - Dicas e otimizações

9. **RESUMO_EXECUTIVO.md** (8 KB)
   - Sumário executivo para o orientador
   - Metodologia detalhada
   - Resultados esperados
   - Próximos passos

10. **requirements.txt** (373 B)
    - Todas as dependências necessárias

---

## 🚀 Como Começar

### Passo 1: Baixar Arquivos

Todos os arquivos estão em `/mnt/user-data/outputs/llm_annotation_system/`

### Passo 2: Instalar Dependências

```bash
pip install -r requirements.txt
```

### Passo 3: Configurar API Keys

Você precisa de API keys para:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Google (Gemini)

Crie um arquivo `.env`:
```env
OPENAI_API_KEY=sua-key
ANTHROPIC_API_KEY=sua-key
GOOGLE_API_KEY=sua-key
```

### Passo 4: Executar

**RECOMENDADO**: Use o notebook Jupyter

```bash
jupyter notebook analise_consenso_llms.ipynb
```

---

## 📊 O Que o Sistema Faz

### 1. Anotação Automática

- ✅ 5 LLMs anotam cada texto
- ✅ Cada LLM anota 3x (validação interna)
- ✅ Total: 15 anotações por instância
- ✅ Sistema de cache (não repete chamadas)

### 2. Análise de Consenso

- ✅ Calcula consenso entre LLMs
- ✅ Calcula consenso interno de cada LLM
- ✅ Identifica casos problemáticos (2-2-1, etc.)
- ✅ Métricas estatísticas completas

### 3. Validação de Parâmetros

- ✅ Testa diferentes temperaturas
- ✅ Testa diferentes top_p
- ✅ Analisa impacto nas anotações
- ✅ "LLM hacking" para otimização

### 4. Visualizações

- ✅ Heatmap de concordância
- ✅ Distribuição de consenso
- ✅ Matriz de confusão
- ✅ Comparação de modelos
- ✅ Dashboard interativo

### 5. Outputs

- ✅ CSVs com todas as anotações
- ✅ CSVs com alta confiança (consenso ≥80%)
- ✅ CSVs com casos para revisão
- ✅ JSON com sumário estatístico
- ✅ PNGs com gráficos
- ✅ HTML com dashboard interativo

---

## 💡 Dicas Importantes

### Para Começar com Poucos Custos

1. **Use amostra pequena primeiro**
   - Teste com 10-20 textos
   - Valide que está funcionando
   - Depois escale

2. **Use modelos mais baratos**
   - Comece com: GPT-3.5, Claude Sonnet, Gemini
   - Depois adicione GPT-4 e Claude Opus

3. **Aproveite o cache**
   - Sistema salva respostas automaticamente
   - Não repete chamadas de API
   - Economiza muito dinheiro

### Para Melhorar Qualidade

1. **Ajuste os prompts** em `config.py`
   - Adicione exemplos (few-shot)
   - Teste Chain-of-Thought
   - Seja específico nas instruções

2. **Teste diferentes configurações**
   - Use `test_param_variations=True`
   - Analise qual funciona melhor
   - Documente seus achados

3. **Analise casos problemáticos**
   - Arquivo `needs_human_review.csv`
   - Entenda por que não há consenso
   - Ajuste prompts ou categorias

---

## 🎓 Para Apresentar ao Orientador

### Material Pronto

1. **RESUMO_EXECUTIVO.md**
   - Leia e customize conforme necessário
   - Adicione resultados reais quando tiver

2. **analise_consenso_llms.ipynb**
   - Execute e gere os resultados
   - Salve com outputs visíveis
   - Apresente este notebook

3. **Dashboard Interativo**
   - Em `results/figures/interactive_dashboard.html`
   - Abra no navegador
   - Mostre as visualizações

### Pontos para Discutir

1. **Metodologia implementada**
   - Multi-LLM com consenso
   - Validação interna
   - Estratégias de resolução

2. **Questões de pesquisa**
   - Threshold ideal de consenso?
   - O que fazer com casos 2-2-1?
   - Few-shot learning ajuda?

3. **Próximos passos**
   - Validar com ground truth
   - Testar em dataset maior
   - Otimizar custos

4. **Publicação**
   - Onde submeter?
   - Quando?
   - Colaborações?

---

## ✉️ Email Sugerido para Celso e Washington

```
Assunto: Validação de Prompt para Anotação Automática com LLMs

Olá Celso e Washington,

Estou desenvolvendo uma metodologia para anotação automática de datasets 
usando múltiplas LLMs com análise de consenso. Implementei um sistema 
completo que testa diferentes prompts e configurações.

Poderiam revisar o prompt base que estou usando? Está no arquivo config.py, 
linha 18 (BASE_ANNOTATION_PROMPT). Quero garantir que estou seguindo as 
melhores práticas de prompt engineering para classificação de textos.

Principais pontos:
- Prompt zero-shot com instruções claras
- Suporte para few-shot (adicionar exemplos)
- Chain-of-Thought para casos complexos

Agradeço muito o feedback de vocês!

Abraço,
Gabriel Catizani
```

---

## 🔧 Customizações Possíveis

### 1. Adicionar Novos Modelos

Edite `config.py` e adicione em `LLM_CONFIGS`:

```python
"novo-modelo": {
    "provider": "openai",  # ou anthropic, google
    "model_name": "nome-exato-do-modelo",
    "default_params": {"temperature": 0.0, "max_tokens": 50},
}
```

### 2. Mudar Categorias

No notebook ou script:

```python
categories = ["Sua", "Lista", "De", "Categorias"]
```

### 3. Customizar Prompts

Edite `config.py`:

```python
BASE_ANNOTATION_PROMPT = """
Seu prompt customizado aqui
{text}
{categories}
"""
```

### 4. Ajustar Parâmetros

Em `config.py` → `EXPERIMENT_CONFIG`:

```python
"num_repetitions_per_llm": 5,  # Aumentar repetições
"consensus_threshold": 0.7,     # Mudar threshold
"no_consensus_strategy": "...", # Mudar estratégia
```

---

## 📈 Estimativa de Custos

### Dataset Pequeno (100 textos)

- 5 modelos × 3 repetições = 15 anotações/texto
- Total: 1.500 chamadas de API
- **Custo estimado: $3-5**

### Dataset Médio (1.000 textos)

- Total: 15.000 chamadas de API
- Com cache: ~10.000 chamadas únicas
- **Custo estimado: $30-50**

### Dataset Grande (10.000 textos)

- Total: 150.000 chamadas
- Com cache e otimizações: ~100.000
- **Custo estimado: $300-500**

**Dica**: Comece pequeno, valide a metodologia, depois escale.

---

## ✅ Checklist de Validação

Antes de apresentar ao orientador:

- [ ] Instalei todas as dependências
- [ ] Configurei minhas API keys
- [ ] Executei o notebook com dataset de teste
- [ ] Gerei todas as visualizações
- [ ] Analisei os resultados
- [ ] Li o RESUMO_EXECUTIVO.md
- [ ] Customizei para meu caso específico
- [ ] Documentei achados importantes
- [ ] Preparei perguntas para discussão

---

## 🎯 Próximos Passos Sugeridos

### Curto Prazo (1-2 semanas)

1. Teste com seu dataset real (amostra pequena)
2. Valide que a metodologia faz sentido
3. Ajuste prompts e parâmetros
4. Apresente resultados preliminares ao orientador

### Médio Prazo (1-2 meses)

1. Execute em dataset completo
2. Valide com ground truth
3. Compare diferentes estratégias
4. Documente resultados para paper

### Longo Prazo (3-6 meses)

1. Escreva o paper
2. Prepare apresentação
3. Submeta para conferência/journal
4. Disponibilize código open-source

---

## 📞 Precisa de Ajuda?

Se tiver dúvidas:

1. Consulte o README.md
2. Veja exemplos no notebook
3. Analise o código (bem comentado)
4. Teste com datasets pequenos primeiro

---

## 🎉 Conclusão

Você agora tem um **sistema completo e profissional** para sua pesquisa. 
O código é modular, bem documentado, e pronto para apresentação acadêmica.

**Boa sorte com sua pesquisa!** 🚀

Você tem uma metodologia sólida, implementação robusta, e material excelente 
para apresentar ao seu orientador e eventualmente publicar.

---

Gabriel Catizani, espero que este sistema atenda suas necessidades. Qualquer dúvida, 
é só perguntar! 😊
