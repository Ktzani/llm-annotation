# Relatório de Experimentos

## Redução do Custo de Anotação Humana por meio de Rotulação Assistida por LLMs e Seleção de Instâncias Sensível a Ruído

> **Escopo dos dados:** primeira replicação ("3 modelos") dos 4 datasets do benchmark de referência.
> **Data do relatório:** 2026-06-18
>
> | Dataset | Pasta do experimento |
> |---|---|
> | movie_review | `data/results/movie_review/2026-04-09_13-17-23` |
> | agnews | `data/results/agnews/2026-04-09_13-20-16 - rep1` |
> | books | `data/results/books/2026-04-09_13-21-37` |
> | dblp | `data/results/dblp/2026-04-09_14-05-21` |

---

## 1. Proposta

Este trabalho avalia se um **ensemble de três LLMs compactas** — Llama3.1-8B, Qwen3-8B e DeepSeek-R1-8B — combinadas via **voto majoritário**, produz anotações **comparáveis às obtidas por modelos fine-tunados** em quatro datasets de classificação de texto provenientes do benchmark de referência: **AGNews, Books, DBLP e Movie Review**. O propósito não é substituir a anotação humana, mas **reduzir seu custo** por meio de rotulação assistida por LLMs combinada a seleção de instâncias sensível a ruído.

Além da qualidade das anotações em si, investiga-se também se a **confiança dos modelos no momento da predição** — capturada pelo log-prob da classe predita — pode servir como indicador da qualidade da anotação, abrindo caminho para mecanismos de filtragem automática, além de outras técnicas de filtragem (como a seleção de instâncias BiO-IS [6]). O benchmark de comparação é um **RoBERTa-base fine-tunado com rótulos gold**, seguindo as partições k-fold da referência [3].

### Perguntas de pesquisa

- **P1:** o uso de um ensemble de LLMs zero-shot, combinado via voto majoritário, é capaz de produzir anotações com qualidade comparável às obtidas por modelos fine-tunados em dados anotados por humanos?
- **P2:** medidas de confiança (log-prob) e consenso entre modelos podem ser usadas como critérios eficazes para filtragem automática, melhorando a qualidade dos dados de treinamento?

> As três métricas de avaliação são: **acurácia** (desempenho geral), **F1-macro** (principal métrica, robusta a desbalanceamento) e **Fleiss' κ** (concordância entre modelos além do acaso).

---

## 2. Metodologia aplicada

Etapas sequenciais:

1. **Anotação com LLMs** — três LLMs (Llama3.1-8B, Qwen3-8B, DeepSeek-R1-8B) anotam, em **zero-shot**, todos os documentos de cada dataset (prompt `simpler`, 1 repetição por modelo).
2. **Métricas individuais por modelo** — acurácia e F1-macro de cada LLM isoladamente, antes da consolidação.
3. **Consolidação das anotações** — voto majoritário entre os 3 modelos, gerando um rótulo final único por documento (`resolved_annotation`).
4. **Métricas do voto majoritário** — acurácia, F1-macro e **Fleiss κ** (concordância) sobre os rótulos consolidados.
5. **Filtragem de instâncias ruidosas** — remoção de instâncias de baixo consenso / provável ruído. Casos de discordância total (1×1×1) são removidos como problemáticos; a seleção de instâncias **BiO-IS** (sensível a ruído) é aplicada para enxugar o conjunto de treino. A confiança (log-prob) é avaliada como sinal auxiliar de qualidade.
6. **Fine-tuning e comparação com o benchmark** — treino de **RoBERTa-base** sobre os rótulos do voto majoritário (**com e sem filtragem**), usando a **mesma validação cruzada k-fold (10 folds) do benchmark**, garantindo comparabilidade direta.

Estruturas de saída por experimento: `summary/`, `consensus/`, `instance_selection/`, `finetuning/{roberta-base, roberta-base-filter}/`, `graphics/`.

---

## 3. Etapas 1–2 — Anotação zero-shot e métricas individuais por modelo

Acurácia e F1-macro de cada LLM contra o *ground truth* (calculados de `summary/dataset_anotado_completo.csv`).

| Dataset | Modelo | Acurácia | F1-macro |
|---|---|---:|---:|
| movie_review | deepseek-r1-8b | 90,36% | 60,33%* |
| movie_review | qwen3-8b | **91,51%** | **91,51%** |
| movie_review | llama3.1-8b | 89,49% | 89,49% |
| agnews | deepseek-r1-8b | 85,32% | 84,90% |
| agnews | qwen3-8b | 86,56% | 86,34% |
| agnews | llama3.1-8b | **86,61%** | **86,45%** |
| books | deepseek-r1-8b | 68,37% | 62,32% |
| books | qwen3-8b | **72,46%** | 65,68% |
| books | llama3.1-8b | 72,06% | **73,07%** |
| dblp | deepseek-r1-8b | 62,58% | 54,80% |
| dblp | qwen3-8b | **63,40%** | 54,56% |
| dblp | llama3.1-8b | 59,95% | **59,39%** |

\* *Em movie_review o F1-macro do deepseek-r1-8b destoa da acurácia (90,4% acc / 60,3% F1), indício de que o modelo emitiu rótulos inválidos/fora do conjunto de classes em parte das amostras, penalizando o macro. Vale revisar o parsing das saídas desse modelo nesse dataset.*

**Leitura:** nenhum modelo domina em todos os datasets — `qwen3-8b` e `llama3.1-8b` se alternam como melhor individual. É exatamente esse cenário que justifica o ensemble: o voto majoritário combina os acertos complementares dos três.

---

## 4. Etapas 3–4 — Voto majoritário e concordância (`summary/sumario_experimento.json`)

Qualidade do rótulo consolidado por voto majoritário, validado contra o *ground truth*, e concordância entre modelos (Fleiss κ; κ de Cohen par a par em `consensus/cohens_kappa.csv`).

| Dataset | Nº textos | Nº classes | **Acurácia (ensemble)** | **F1-macro (ensemble)** | Fleiss κ | Interpret. | Alto (3×0) | Médio (2×1) | Baixo (1×1×1) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| movie_review | 10.653 | 2 | **91,35%** | 91,35% | 0,865 | Excelente | 9.577 | 1.076 | 0 |
| agnews | 127.600 | 4 | **86,76%** | 86,53% | 0,917 | Excelente | 115.551 | 11.758 | 0 |
| books | 33.594 | 8 | **72,35%** | 72,64% | 0,806 | Excelente | 24.810 | 8.195 | 0 |
| dblp | 38.128 | 10 | **63,56%** | 61,12% | 0,766 | Bom | 27.060 | 10.178 | 810 |

**Ensemble vs melhor modelo individual (acurácia):**

| Dataset | Melhor individual | Voto majoritário | Δ |
|---|---:|---:|---:|
| movie_review | 91,51% (qwen) | 91,35% | −0,16 |
| agnews | 86,61% (llama) | 86,76% | **+0,15** |
| books | 72,46% (qwen) | 72,35% | −0,11 |
| dblp | 63,40% (qwen) | 63,56% | **+0,16** |

**Leitura:**
- O voto majoritário fica **no nível do melhor modelo individual** (diferenças ≤ 0,16 pp), porém com **mais robustez**: não depende de saber de antemão qual LLM é a melhor para cada dataset, e suaviza erros idiossincráticos de cada modelo.
- A concordância é **de boa a excelente** (Fleiss κ 0,77–0,92), caindo conforme cresce o número de classes/dificuldade (movie_review 2 classes → dblp 10 classes).
- Só `dblp` teve casos 1×1×1 (810), removidos como problemáticos antes do fine-tuning.

---

## 5. Etapa 5a — Confiança (log-prob) como indicador de qualidade

Confiança média da predição por modelo, separando instâncias em que o rótulo consolidado **acertou** vs **errou** o *ground truth* (de `*_rep1_conf`).

| Dataset | Modelo | Conf. (corretas) | Conf. (erradas) | Δ (separabilidade) |
|---|---|---:|---:|---:|
| movie_review | llama3.1-8b | 0,960 | 0,821 | **0,139** |
| agnews | llama3.1-8b | 0,978 | 0,897 | **0,081** |
| books | llama3.1-8b | 0,942 | 0,902 | 0,040 |
| dblp | llama3.1-8b | 0,949 | 0,839 | **0,110** |
| (todos) | qwen3-8b | ~1,000 | ~1,000 | ≈ 0 |
| (todos) | deepseek-r1-8b | ~0,99 | ~0,99 | ≈ 0 |

**Leitura — achado importante:** apenas a confiança do **llama3.1-8b** é **discriminativa** (corretas consistentemente mais confiantes que erradas, gap de até 0,14). `qwen3-8b` e `deepseek-r1-8b` **saturam em ~1,0** independentemente de acerto/erro — seu log-prob **não serve** como filtro de qualidade. Portanto, usar confiança como mecanismo de filtragem automática só é viável com modelos calibrados (no ensemble, o llama); para os demais, é preciso recorrer a sinais de consenso ou a técnicas como o BiO-IS.

---

## 6. Etapa 5b — Seleção de Instâncias BiO-IS (`instance_selection/instance_selection_report.json`)

O BiO-IS (sensível a ruído) treina um classificador fraco (`LogisticRegression`) e separa as remoções em **redundantes** (bem representadas por vizinhos da mesma classe) e **ruído** (atípicas / provável erro). Parâmetros: `beta=0.25`, `theta=0.5`, `random_state=42`.

| Dataset | Total | Mantidas | Removidas | Redundantes | Ruído | Redução | Acc. clf. fraco | F1-macro clf. fraco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| movie_review | 10.653 | 6.717 | 3.936 | 2.663 | 1.273 | 36,9% | 0,761 | 0,761 |
| agnews | 127.309 | 92.024 | 35.285 | 31.827 | 3.458 | 27,7% | 0,946 | 0,942 |
| books | 33.005 | 21.916 | 11.089 | 8.251 | 2.838 | 33,6% | 0,828 | 0,829 |
| dblp | 38.048* | 25.414 | 12.634 | 9.512 | 3.122 | 33,2% | 0,836 | 0,770 |

\* `dblp`: o relatório parte de 38.048 instâncias rotuladas; o notebook de análise, que também exclui as 1×1×1, trabalha com 37.238. A maior parte das remoções é por **redundância**, não por erro.

---

## 7. Etapa 6 — Fine-tuning RoBERTa-base e comparação (Sem filtro vs Com filtro)

Validação cruzada (10 folds, **mesmos folds do benchmark**), bloco `cv` de `finetuning/<variante>/roberta-base_fine_tuning_results.json`. Média ± desvio entre folds.

| Dataset | Variante | Acurácia (CV) | F1-macro (CV) | Épocas (méd.) |
|---|---|---:|---:|---:|
| movie_review | Sem filtro | 88,79 ± 0,63 | 88,78 ± 0,64 | 11,3 |
| movie_review | **Com filtro** | 88,73 ± 0,50 | 88,72 ± 0,50 | 12,1 |
| agnews | Sem filtro | 87,28 ± 0,31 | 87,14 ± 0,34 | 7,2 |
| agnews | **Com filtro** | **87,71 ± 0,27** | **87,61 ± 0,27** | 6,6 |
| books | Sem filtro | 71,69 ± 0,79 | 72,80 ± 0,75 | 10,4 |
| books | **Com filtro** | 71,56 ± 1,03 | 72,60 ± 0,92 | 7,4 |
| dblp | Sem filtro | 63,56 ± 0,95 | 61,39 ± 0,94 | 7,4 |
| dblp | **Com filtro** | 63,51 ± 0,40 | 61,13 ± 0,42 | 11,3 |

**Δ (Com filtro − Sem filtro), em pontos percentuais:**

| Dataset | ΔAcurácia | ΔF1-macro | Dados removidos pelo BiO-IS |
|---|---:|---:|---:|
| movie_review | −0,06 | −0,06 | 36,9% |
| agnews | **+0,43** | **+0,47** | 27,7% |
| books | −0,13 | −0,20 | 33,6% |
| dblp | −0,05 | −0,26 | 33,1% |

**Leitura:**
- O filtro **preserva a performance treinando com ~28–37% menos dados** — as diferenças ficam dentro do desvio entre folds (**estatisticamente equivalentes**), com leve **ganho em agnews** (+0,43 acc / +0,47 F1).

### 7.1 Comparação com o benchmark humano [3] (F1-macro)

Baseline = RoBERTa-base fine-tunado com **rótulos gold (humanos)**, mesmos folds k-fold da referência [3]. Valores em F1-macro, média (desvio).

Valores em F1-macro como `média (IC 95%)`, com o intervalo de confiança calculado sobre os folds (t-Student, `t·s/√n`). Folds: 10 para movie_review/books/dblp, 5 para agnews — em ambos os lados.

| Dataset | **Benchmark humano [3]** | LLM sem filtro | Δ (sem − humano) | LLM com filtro | Δ (com − humano) |
|---|---:|---:|---:|---:|---:|
| movie_review | 89,0 (±0,7) | 88,78 (±0,48) | **−0,22** | 88,72 (±0,38) | **−0,28** |
| agnews | 91,7 (±0,2) | 87,14 (±0,47) | −4,56 | 87,61 (±0,37) | −4,09 |
| books | 87,2 (±0,6) | 72,80 (±0,57) | −14,40 | 72,60 (±0,69) | −14,60 |
| dblp | 81,4 (±0,5) | 61,39 (±0,71) | −20,01 | 61,13 (±0,32) | −20,27 |

> **Nota:** o valor entre parênteses é o **IC 95% sobre os folds** dos dois lados (benchmark = coluna `macro_avg_ic` da referência [3]; LLM = recalculado dos folds). Em movie_review os intervalos **se sobrepõem** (88,78 ±0,48 vs 89,0 ±0,7) ⇒ empate estatístico. Em agnews os intervalos **não** se sobrepõem por pouco (limite inf. humano 91,5 vs sup. LLM ~88,0), e em books/dblp a separação é ampla ⇒ gaps reais e significativos.

**Leitura — o achado central de P1:** a proximidade com a anotação humana **depende fortemente da dificuldade da tarefa**:
- **movie_review (2 classes):** o pipeline assistido por LLMs **iguala** o benchmark humano (gap ≈ 0,2–0,3 pp, dentro do desvio). Qualidade comparável **confirmada**.
- **agnews (4 classes):** fica ~4 pp abaixo — próximo, mas com folga ainda perceptível.
- **books (8 classes) e dblp (10 classes):** gaps grandes (−14 e −20 pp). Aqui a anotação automática **não** alcança a humana; o sinal ruidoso dos LLMs nas tarefas multiclasse difíceis (já visível na acurácia vs ground truth de 72% e 64%) se propaga ao classificador.

O filtro BiO-IS não fecha esse gap (ele preserva, não recupera, performance) — mas também quase não custa: mantém o nível treinando com bem menos dados.

---

## 8. Validação da qualidade do filtro (`src/notebooks/analise_instance_selection.ipynb`)

Pergunta central: **o filtro concentra os erros de anotação nas instâncias que descarta?** Universo = `mantidas + removidas`, excluindo as 1×1×1.

### 8.1 Síntese consolidada (Seção 10 do notebook)

| Dataset | Redução % | % remov. corretas | % remov. erradas | % alto consenso (3×0) | **Erro removidas %** | **Erro mantidas %** | **Recall de erros %** |
|---|---:|---:|---:|---:|---:|---:|---:|
| movie_review | 36,9 | 89,8 | 10,2 | 88,4 | 10,2 | 7,8 | 43,4 |
| agnews | 27,7 | 85,6 | 14,4 | 89,3 | 14,4 | 12,8 | 30,2 |
| books | 33,6 | 70,7 | 29,3 | 72,6 | 29,3 | 26,8 | 35,6 |
| dblp | 33,1 | 63,2 | 36,8 | 70,8 | 36,8 | 34,4 | 34,4 |

A taxa de erro é **sempre maior nas removidas do que nas mantidas** — o filtro **prioriza descartar o que os modelos erraram**. De 30% a 43% de todas as anotações erradas foram removidas (recall de erros).

### 8.2 Onde os erros se concentram

**Por nível de consenso** (% de erro entre as removidas):

| Dataset | Alto (3×0) | Baixo (2×1) |
|---|---:|---:|
| movie_review | 6,2 | 40,6 |
| agnews | 10,4 | 48,1 |
| books | 21,5 | 50,0 |
| dblp | 27,1 | 60,4 |

**Por motivo de remoção** (% de erro):

| Dataset | Ruído | Redundante |
|---|---:|---:|
| movie_review | 20,0 | 5,4 |
| agnews | 52,3 | 10,3 |
| books | 48,3 | 22,7 |
| dblp | 65,2 | 28,2 |

Confirma o critério do BiO-IS: o que ele marca como **ruído** tem taxa de erro 2× a 6× maior do que o que marca como **redundante**, e os erros se concentram no **baixo consenso (2×1)**.

---

## 9. Conclusões — respondendo às perguntas de pesquisa

### P1 — Anotação por ensemble vs fine-tuning humano

**Resposta: a qualidade é comparável à humana em tarefas de baixa granularidade, mas degrada com o número de classes/dificuldade.** Comparando F1-macro contra o benchmark humano [3] nos mesmos folds (§7.1):
- **movie_review (2 classes): empate** com a anotação humana (gap ≈ 0,2–0,3 pp). P1 **confirmada**.
- **agnews (4 classes):** ~4 pp abaixo — próximo.
- **books (8) e dblp (10):** gaps de −14 e −20 pp — P1 **não se sustenta** nessas tarefas.

Internamente o ensemble é sólido: o voto majoritário fica no nível do melhor modelo individual (Δ ≤ 0,16 pp) com mais robustez e concordância boa-a-excelente (Fleiss κ 0,77–0,92). A limitação é a **qualidade da anotação vs ground truth** nas tarefas multiclasse difíceis (72% em books, 64% em dblp), que se propaga ao classificador. **Conclusão de P1:** ensembles de LLMs compactas são uma alternativa viável e de baixo custo à anotação humana em tarefas com poucas classes; para muitas classes, ainda há gap relevante.

### P2 — Confiança e consenso como critérios de filtragem

**Resposta: o consenso é um critério eficaz; o log-prob só é útil em modelos calibrados.**
- **Confiança (log-prob):** discrimina acerto de erro **apenas no llama3.1-8b** (gap até 0,14 entre corretas/erradas); `qwen3-8b` e `deepseek-r1-8b` saturam em ~1,0 e **não servem** como sinal de qualidade. Filtragem por confiança exige modelos calibrados — alinhado às ressalvas de [7, 8].
- **Consenso + BiO-IS [6]:** a seleção de instâncias **preserva a performance do fine-tuning usando ~28–37% menos dados** (diferenças dentro do desvio entre folds; leve ganho em agnews: +0,43 pp acc). O filtro é uma **limpeza dirigida** — descarta proporcionalmente mais anotações erradas do que mantém, concentrando os erros no baixo consenso (2×1) e nas instâncias marcadas como ruído. Isso confirma o consenso entre modelos como critério de filtragem eficaz.

### Síntese

O pipeline assistido por LLMs **iguala a anotação humana em movie_review** e fica próximo em agnews, a custo muito menor; em books e dblp o gap ainda é grande. O **consenso** mostrou-se um critério de filtragem eficaz e o **BiO-IS** reduz o dataset sem perda de performance; o **log-prob** só é aproveitável em modelos calibrados. Como próximos passos naturais: investigar por que a anotação degrada nas tarefas multiclasse (confusões sistemáticas entre classes vistas nos *classification reports*) e testar filtragem combinando consenso + confiança calibrada.

---

### Apêndice — Arquivos-fonte por dataset

Para cada `<pasta_experimento>` da tabela do topo:

- Anotação + métricas por modelo: `summary/dataset_anotado_completo.csv`
- Consolidação + concordância: `summary/sumario_experimento.json`, `consensus/cohens_kappa.csv`, `consensus/pairwise_agreement.csv`
- Fine-tuning sem filtro: `finetuning/roberta-base/roberta-base_fine_tuning_results.json`
- Fine-tuning com filtro: `finetuning/roberta-base-filter/roberta-base_fine_tuning_results.json` (em `books`: `roberta-base_filter`)
- Instance selection: `instance_selection/instance_selection_report.json` (+ `dataset_filtrado.csv`, `instancias_removidas.csv`)
- Análise consolidada do filtro: [`src/notebooks/analise_instance_selection.ipynb`](../src/notebooks/analise_instance_selection.ipynb)

---

### Referências

- **[3]** CUNHA, W.; ROCHA, L.; GONÇALVES, M. A. *A thorough benchmark of automatic text classification: From traditional approaches to large language models.* arXiv:2504.01930, 2025. — **benchmark de referência** (RoBERTa-base com rótulos gold, partições k-fold).
- **[6]** CUNHA, W. et al. *A noise-oriented and redundancy-aware instance selection framework.* ACM TOIS, v. 43, n. 2, 2025. — **BiO-IS**, técnica de seleção de instâncias sensível a ruído.
- **[7]** FARR, D. et al. *LLM confidence evaluation measures in zero-shot CSS classification.* arXiv:2410.13047, 2024.
- **[8]** LI, M. et al. *CoAnnotating: Uncertainty-guided work allocation between human and large language models for data annotation.* EMNLP 2023, p. 1487–1505.
