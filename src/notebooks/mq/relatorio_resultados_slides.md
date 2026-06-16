# Relatório de Resultados — Anotação Automática com Ensemble de LLMs Zero-Shot

> **Uso:** material para gerar um **slide de ~5 min** (Claude web). 10 slides, 1 por seção,
> linguagem enxuta, números em destaque.
> **Autor:** Gabriel Catizani Faria Oliveira — DCC/UFMG.

> 📷 **INSTRUÇÃO AO GERADOR DE SLIDES:** este relatório contém marcadores
> **`📷 [IMAGEM: ...]`** indicando onde entram gráficos que **o usuário fornecerá**.
> **Não invente nem gere essas figuras** — ao montar os slides, **peça ao usuário
> para enviar cada imagem listada** (pelos nomes/descrições dos marcadores) antes de
> finalizar a apresentação.

---

## ⚠️ Mudanças frente à proposta inicial (refletir nos slides)

- **Fatorial 2³** usa **Domínio (B) × Consenso (C) × Tamanho do texto (D)** —
  *o tamanho do texto substituiu o fator "arquitetura"* da proposta.
- **Consenso é uma partição por exemplo**: **2×1** = exatamente 2 dos 3 modelos
  concordam (maioria simples); **3×0** = os 3 concordam (unânime). Cada exemplo
  cai em **um único** grupo (2×1 + 3×0 = total).
- **Regressão** feita no **nível da anotação individual**, e **não** agregada por
  dataset; sem calibração (confiança saturada ≈ 1).
- **Comparação de sistemas** pareada **por fold** (não pelos 4 datasets) e via
  **intervalos de confiança** (sem teste de hipótese).
- Métricas efetivamente usadas: **F1-macro (principal)** e **Acurácia**.
- **Replicação só para MovieReview e Books** (5 execuções cada) — replicar os 4
  datasets seria custoso e sem ganho. Por isso o **fatorial usa apenas MovieReview
  (geral) e Books (técnico)**; AGNews e DBLP entram só na comparação de sistemas
  (P1) e na regressão (P3), com execução única.

---

## Slide 1 — Problema, perguntas e objetivos

- Anotação manual é cara e lenta. **LLMs zero-shot** podem anotar em escala, mas
  modelos isolados têm vieses/inconsistências → **ensemble (voto majoritário) de 3
  LLMs 8B**: Llama3.1, Qwen3, DeepSeek-R1.
- **Objetivo:** avaliar se o ensemble zero-shot gera rótulos úteis e se a confiança
  do modelo indica qualidade.
- **P1:** o desempenho *downstream* com rótulos do ensemble equivale ao treinado
  com rótulos reais?
- **P2:** domínio, grau de consenso e tamanho do texto afetam a qualidade da anotação?
- **P3:** a confiança (log-prob) prediz se a anotação está correta?

## Slide 2 — Métricas

- **F1-macro** — variável-resposta principal (controla desbalanceamento de classes).
- **Acurácia** — referência geral.
- *(Fleiss' Kappa fica para a versão completa.)*

## Slide 3 — Parâmetros

- **Controláveis:** modelo LLM (3) e dataset (4).
- **Fixados:** parâmetros de inferência no *default*; **prompt único** padronizado
  para todos os modelos/datasets.
- **Partição k-fold idêntica ao benchmark de referência** → comparabilidade sem
  viés de seleção.

## Slide 4 — Fatores estudados (variados)

Projeto **2³** (níveis −1 / +1):

| Fator | Nível −1 | Nível +1 |
|-------|----------|----------|
| **B — Domínio** | Geral (MovieReview) | Técnico (Books) |
| **C — Consenso** | Maioria simples (2×1) | Unanimidade (3×0) |
| **D — Tamanho do texto** | Curto | Longo (corte na mediana, por domínio) |

## Slide 5 — Técnicas

- **P1 — Comparação de sistemas:** intervalos de confiança (95%) da **diferença
  pareada por fold** (RoBERTa treinado com voto majoritário × com dados reais).
- **P2 — Projeto fatorial 2³ com replicação:** efeitos principais + interações;
  resposta = F1-macro e Acurácia.
- **P3 — Regressão linear** no nível da anotação: confiança dos modelos como
  preditor de a anotação estar correta.

## Slide 6 — Carga de trabalho (workload)

- **4 datasets** do benchmark de referência, com nº de classes distinto:

  | Dataset | Domínio | Nº de classes |
  |---|---|---|
  | MovieReview | Geral | 2 (binário) |
  | AGNews | Geral | 4 |
  | Books | Técnico | 8 |
  | DBLP | Técnico | 10 |

- **3 LLMs** anotando **todos os documentos** (dados já coletados).
- **Replicação só em MovieReview e Books** (5 execuções cada) → base do **fatorial**
  (20 obs. no 2², 40 no 2³). AGNews e DBLP: execução única.
- Regressão: **~208 mil anotações** individuais (4 datasets).

## Slide 7 — Execução

- Consenso entre os 3 modelos calculado de forma única e reaproveitado nas 3 análises.
- Fine-tuning RoBERTa com **10 folds** (5 no AGNews), comparado contra os folds do
  baseline treinado com dados reais.

## Slide 8 — Análise e interpretação (resultados principais)

### P1 — Voto majoritário vs dados reais (F1-macro, IC 95% da diferença)

Valores = **média do F1 entre os folds ± IC 95%**.

| Dataset | Majoritário (média F1 ± IC) | Real (média F1 ± IC) | Diferença das médias (IC 95%) | Conclusão |
|---|---|---|---|---|
| **MovieReview** | 0,888 ± 0,005 | 0,890 ± 0,007 | **+0,003 [−0,005; +0,010]** | **Equivalentes** |
| AGNews | 0,871 ± 0,005 | 0,917 ± 0,002 | +0,046 [+0,040; +0,052] | Real melhor |
| Books | 0,728 ± 0,006 | 0,872 ± 0,006 | +0,144 [+0,136; +0,152] | Real melhor |
| DBLP | 0,614 ± 0,007 | 0,814 ± 0,005 | +0,201 [+0,193; +0,208] | Real melhor |

> 📷 **[IMAGEM: forest plot dos IC 95% da diferença (real − majoritário) por dataset, com a linha do zero]**

### P2 — Fatorial 2³ (F1-macro: efeito 2q / % de variação)

- **Consenso (C): +0,30 → ~82% da variação** (unânime 3×0 ≫ maioria 2×1) — **fator dominante**.
- **Domínio (B): −0,13 → ~16%** (domínio técnico reduz a qualidade).
- **Tamanho (D) e interações:** pequenos (cada < 1,5%); **CD não significativo**.
- **Erro experimental** baixíssimo: **~0,26% da variação** (s²ₑ ≈ 8,7×10⁻⁵) →
  réplicas muito consistentes.
- **Intervalo de confiança (95%):** o IC de **todos os efeitos não contém 0**
  (efeitos relevantes), **exceto a interação CD** (consenso × tamanho), cujo IC
  contém 0. Como o erro é mínimo, a leitura relevante é o **tamanho do efeito /
  % de variação**.
- *(Acurácia: mesmo padrão — C ~83%, B ~16%; erro ~0,23%.)*
- ⚠️ **Possível viés:** no fatorial, MovieReview (geral, **2 classes**) vs Books
  (técnico, **8 classes**) diferem **tanto no domínio quanto no nº de classes**.
  Logo, o efeito de **B não pode ser atribuído só ao domínio** — parte pode vir da
  maior dificuldade multiclasse. Vale para a comparação entre datasets em geral.

> 📷 **[IMAGEM: gráfico de barras da % de variação por efeito do fatorial 2³ (C dominante, B em segundo)]**

### P3 — Confiança × acerto (análise por dataset)

- **Confiança saturada perto de 1** em todos os datasets → os pontos ficam **bem
  abaixo da diagonal y = x** ⇒ modelos **super-confiantes** (confiança ≫ acerto real).
- **Dentro de cada dataset** a relação é **fraca e não-linear** (quase plana, sobe
  só no bin do topo) e o **poder explicativo é baixo (R² ≈ 0,05–0,14)** → a
  confiança **discrimina mal** a qualidade da anotação.
- Cada dataset opera em um **patamar de acerto distinto** (MovieReview/AGNews altos,
  DBLP baixo) — por isso a análise é **por dataset**; agrupar tudo distorceria a curva.
- ⚠️ A **resposta binária não é o problema** (cabe um modelo de probabilidade linear /
  regressão com variáveis categóricas). O que limita é a **confiança saturada perto
  de 1** (pouca variância no preditor) e a **relação não-linear** → o ajuste capta
  pouco. Trabalho futuro: **transformar a confiança** e **calibrar** os log-probs.

> 📷 **[IMAGEM: curva taxa de acerto × confiança média, UMA por dataset (não agrupado); confiança saturada perto de 1, relação fraca/NÃO-linear]**

## Slide 9 — Conclusões

- **Anotação automática por voto majoritário é viável em domínio geral**
  (MovieReview: estatisticamente equivalente ao *ground truth*) — anota "de graça";
  **em domínios técnicos (DBLP, Books) ainda há perda relevante**.
- **O grau de consenso é o que mais importa**: exigir **unanimidade** eleva muito a
  qualidade — bom critério de filtragem.
- **Confiança isolada é um indicador fraco** (super-confiança dos LLMs, saturada
  perto de 1) e a relação com o acerto **não é linear** → trabalho futuro:
  **transformar a confiança** e **calibrar** os log-probs.
