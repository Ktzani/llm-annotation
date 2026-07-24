# Plano — Anotação em 2 Fases (Filtro de Classes + LLM Zero-Shot com Espaço Reduzido)

## Contexto

Hoje o pipeline de anotação apresenta ao LLM **todas** as classes do dataset no prompt e pede que ele escolha uma. Queremos reduzir o espaço de decisão do LLM **sem** deixá-lo de ser zero-shot: um classificador leve (começando por Logistic Regression) filtra as `k` classes mais prováveis por texto (Fase 1) e o LLM anota exatamente como hoje, mas vendo apenas essas `k` candidatas (Fase 2). Toda a supervisão vive no classificador; o prompt do LLM continua sem exemplos rotulados, sem scores, sem ranking — a **única** diferença é o tamanho da lista de classes.

O objetivo experimental é medir o trade-off entre **redução de contexto** e **teto de acerto**: o `recall@k` do filtro (com que frequência a classe verdadeira está entre as top-k) é o teto de acurácia da Fase 2, porque o que o filtro descarta o LLM não recupera.

### Estrutura de dados definida (aninhada)

- Cada dataset já possui **folds externos** (train/test) pré-construídos no HF Hub `waashk/<dataset>` como `train_fold_{N}.parquet` / `test_fold_{N}.parquet`, auto-descobertos incrementando `N` até faltar arquivo (padrão já usado no fine-tuning, ver `src/systems/fine_tune_system/pipeline.py:335-382`).
- **Esta feature opera apenas dentro do conjunto de treino de cada fold externo.** O `test_fold` externo **não** é tocado (fica reservado para o fine-tuning downstream — evita vazamento na avaliação final).
- Dentro do treino de um fold externo, dividimos em `N_inner` pedaços (ex.: 5) de forma estratificada e semeada. Um pedaço designado vira **held-out**; o LR treina nos demais `N_inner - 1` pedaços.
- No held-out (que tem ground-truth): (a) medimos `recall@k` do LR e (b) o LLM anota zero-shot com as top-k candidatas. Sem vazamento — o LR nunca viu o pedaço que prevê.
- Itera **todos** os folds externos.

### Decisões fechadas com o usuário

| Decisão | Escolha |
|---|---|
| Alvo da anotação | Um pedaço held-out de dentro do treino (não o test fold externo) |
| CV interno | **Um** pedaço held-out (sem rotação) |
| Folds externos | Iterar **todos** |
| Numeração das classes no prompt | **Índices canônicos fixos** (subconjunto das linhas `- idx: label`, índices originais preservados) |
| Resposta fora das k candidatas | **Inválida → -1** |
| `k` na Fase 2 (LLM) | **k único por run** (configurável); a varredura `recall@k` é só validação barata |

---

## Diagnóstico do código atual (mapa do fluxo)

### Fluxo de anotação (baseline — não muda)
`src/run_annotation.py` → `AnnotationPipeline` (`src/systems/llm_annotation_system/pipeline.py`) → `load_texts` (via `load_hf_dataset`) → `run_dataset` → `LLMAnnotator.annotate_dataset` → `_annotate_text` → `_annotate_model` → `AnnotationEngine.annotate` → `_annotate_rep` → cache/LLM → `ResponseProcessor` → `evaluate_model_metrics`.

### Pontos críticos mapeados

1. **Template é renderizado UMA vez** no construtor do `AnnotationEngine` (`_prepare_template`, `annotation_engine.py:47,151-203`): as categorias são "assadas" em `self.template` e só `{text}` varia por texto. O bloco `{categories}` vira linhas `- {idx}: {label}` a partir de `LABEL_MEANINGS[dataset]` (ou fallback enumerado). → **Para lista variável por texto, o prompt precisa ser renderizado por-texto.**

2. **Chave de cache NÃO inclui as categorias** (`cache_manager.py:59-72`): `md5(f"{model}|{text}|{json.dumps({'rep': rep})}")`. → **Uma rodada filtrada colidiria com o cache do baseline e retornaria a resposta de todas as classes.** Ponto de vazamento nº 1 a corrigir.

3. **`ResponseProcessor`** (`response_processor.py:24-65`) extrai um int e valida `value in self.categories`; fora da lista → `-1`. As `categories` são os rótulos canônicos (ints). → Para a Fase 2, a validação precisa ser contra o **subconjunto k** apresentado.

4. **Ground-truth** = coluna `label` do HF (`data_loader.py:148-151`). `load_hf_dataset_as_dataframe` (`data_loader.py:174-198`) devolve `df[text, label]` + descrições. Fold: passar `hf_file="train_fold_{N}.parquet"` via `DatasetConfig` (já suportado, `data_loader.py:56-71`).

5. **`text_id` = `md5(text.strip())`** (`get_text_id_from_text.py`). Joins entre o mapa de candidatas e as anotações dependem de computar `text_id` sobre **o mesmo objeto de texto** — atenção à memória `[[text-id-canonical-key-mismatch]]`.

6. **Métricas** (`evaluate_model_metrics.py`) operam em `{model}_consensus` vs `ground_truth`, tratando `-1` como classe de erro; produz `accuracy`, `f1_macro`, `precision/recall_macro`, `coverage`. Reaproveitável sem mudança.

### Ativos reutilizáveis (não reinventar)
- **`TextVectorizer`** (TF-IDF) — `src/systems/instance_selection_system/core/text_vectorizer.py` (config `TFIDF_CONFIG` em `src/config/instance_selection.py`). ⚠️ hoje só expõe `fit_transform`; o filtro precisa de `fit`/`transform` separados (o `TfidfVectorizer` interno já suporta) para não vazar o held-out no vocabulário.
- **Padrão de plugin/fábrica** — `InstanceSelectionMixin` (`core/base.py`), `get_selector` (`selection/selector_factory.py`), `BIOIS` (`selection/biois.py`) usa `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)` + `StratifiedKFold(shuffle=True, random_state=42)`. **Espelhar exatamente esse padrão** para o filtro de classes.
- **Padrão de config aditiva opcional** — `FineTuningInstanceSelectionConfig` (`src/api/schemas/fine_tuning/instance_selection.py`) com `enabled: bool` + `params: Dict[str,Any]`, plugado em `ExperimentRequest` via `default_factory`. **Espelhar para `class_filter`.**
- **Descoberta de folds** — laço `while True` incrementando `fold` (`fine_tune_system/pipeline.py:335-382`). Portável.

---

## Arquitetura proposta

Tudo **aditivo e atrás de flag** (`class_filter.enabled=False` por padrão). O `AnnotationPipeline` atual e o `run_annotation.py` continuam **idênticos**.

### Novo pacote `src/systems/class_filter_system/` (espelha `instance_selection_system/`)

```
src/systems/class_filter_system/
├── core/
│   └── base.py                    # ClassFilterBase (interface plugável)
├── classifiers/
│   └── logistic_regression.py     # LogisticRegressionClassFilter (1ª impl.)
├── factory.py                     # get_class_filter(method, ...) — espelha get_selector
├── validation/
│   └── recall_at_k.py             # recall@k sweep + topk_candidates
├── folds/
│   └── inner_split.py             # split estratificado/semeado treino → (fit, held-out)
└── pipeline.py                    # TwoPhaseAnnotationPipeline (orquestra folds + fases)
```

### Interface do filtro (plugável)
`ClassFilterBase` (ABC) — contrato mínimo:
- `fit(self, texts: list[str], labels: list[int]) -> self` — treina vetor + classificador **só** no fit part.
- `predict_proba(self, texts) -> np.ndarray` — matriz `n_texts × n_classes`, colunas alinhadas a `self.classes_`.
- `classes_ : list[int]` — rótulos canônicos vistos no treino.
- (helper) `topk(self, texts, k) -> list[list[int]]` — deriva top-k de `predict_proba`.

`LogisticRegressionClassFilter(ClassFilterBase)`: `TextVectorizer().fit(fit_texts)` + `LogisticRegression` (mesmos hiperparâmetros do BIOIS), `classes_` = `model.classes_`. `factory.get_class_filter("logistic_regression", random_state=42, **params)` valida contra um dict de estratégias (novo, em `src/config/class_filter.py`) e despacha.

### Config nova: `ClassFilterConfig` (`src/api/schemas/annotation_experiment/class_filter.py`)

| Campo | Tipo | Default | Papel |
|---|---|---|---|
| `enabled` | `bool` | `False` | Liga/desliga a estratégia |
| `method` | `str` | `"logistic_regression"` | Filtro plugável |
| `k` | `int` (ge=1) | `3` | k da Fase 2 (LLM), configurável |
| `k_sweep` | `list[int]` | `[2,3,4,5]` | k's do relatório recall@k (só validação) |
| `n_inner_folds` | `int` (ge=2) | `5` | Nº de pedaços do treino |
| `holdout_fold_index` | `int` | `0` | Qual pedaço interno é held-out |
| `neutral_order` | `str` | `"canonical"` | Ordem das candidatas no prompt |
| `random_state` | `int` | `42` | Semente do split interno e do filtro |
| `params` | `dict` | `{}` | Hiperparâmetros repassados ao filtro |
| `train_fold_pattern` | `str` | `"train_fold_{fold}.parquet"` | Descoberta de folds externos |

Plugada em `ExperimentRequest` (`experiment.py`) como `class_filter: ClassFilterConfig = Field(default_factory=ClassFilterConfig)`. Como tem `default_factory` e `enabled=False`, **todos os JSONs existentes continuam válidos e com comportamento idêntico**.

### Injeção da lista reduzida no engine (aditiva)
`AnnotationEngine` e `LLMAnnotator` ganham parâmetro **opcional** `candidates_by_text_id: dict[str, list[int]] | None = None`:
- `None` (default) → caminho de hoje, byte-idêntico (template congelado, chave de cache `{"rep": rep}`).
- fornecido → por texto:
  - `candidate_indices = candidates_by_text_id.get(text_id)`; renderiza `{categories}` só com esse subconjunto, **ordenado por índice canônico ascendente** (ordem neutra — esconde o ranking do LR), reusando a mesma lógica/descrições de `_prepare_template` (extraída para helper `_render_categories_str(indices)`).
  - chave de cache passa a `{"rep": rep, "cand": sorted(candidate_indices)}` → nunca colide com o baseline.
  - `ResponseProcessor` valida contra o subconjunto k (fora → `-1`). Implementação: passar as categorias válidas por-texto no ponto de parse (o processor ganha um parâmetro opcional `valid_categories` no `extract_*`, default = `self.categories`).

### Orquestrador `TwoPhaseAnnotationPipeline`
Ativado por um novo entry point `src/run_two_phase_annotation.py` (espelha `run_annotation.py`) quando `class_filter.enabled=True`. Por fold externo `f`:
1. Carrega `train_fold_{f}.parquet` → `df_train[text, label]`; calcula `text_id`.
2. `inner_split(df_train, n_inner_folds, holdout_fold_index, seed)` → `(df_fit, df_holdout)` estratificado.
3. `filter = get_class_filter(...); filter.fit(df_fit.text, df_fit.label)`.
4. `proba = filter.predict_proba(df_holdout.text)`.
5. `recall_at_k_sweep(df_holdout.label, proba, filter.classes_, k_sweep)` → `recall_at_k.csv`.
6. `candidates_by_text_id` = top-`k` canônico por texto do held-out.
7. `LLMAnnotator(..., candidates_by_text_id=...).annotate_dataset(df_holdout.text)` → anota **só** o held-out com espaço reduzido.
8. Merge com GT do held-out → `evaluate_model_metrics`.
9. Persiste por fold em `results/<dataset>/two_phase/fold_{f}/<timestamp>/`: `annotations.csv`, `model_metrics.csv`, `recall_at_k.csv`, `filter_report.json`.

Ao final, agrega `recall@k` e métricas do LLM (média ± desvio entre folds).

**Baseline comparável (apples-to-apples):** o mesmo orquestrador com `enabled=False` (ou uma flag `filter_active=False`) seleciona **exatamente o mesmo held-out** e anota com **todas** as classes → baseline sobre os mesmos textos. Assim a comparação isola só o efeito da redução do espaço.

---

## Mudança arquivo a arquivo (conceitual)

**Novos arquivos**
- `src/config/class_filter.py` — dict `CLASS_FILTER_STRATEGIES` + defaults (semente, TF-IDF já vem de `instance_selection`).
- `src/api/schemas/annotation_experiment/class_filter.py` — `ClassFilterConfig` (Pydantic, tabela acima).
- `src/systems/class_filter_system/core/base.py` — `ClassFilterBase`.
- `src/systems/class_filter_system/classifiers/logistic_regression.py` — `LogisticRegressionClassFilter`.
- `src/systems/class_filter_system/factory.py` — `get_class_filter`.
- `src/systems/class_filter_system/validation/recall_at_k.py` — `recall_at_k_sweep`, `topk_candidates`.
- `src/systems/class_filter_system/folds/inner_split.py` — split estratificado semeado.
- `src/systems/class_filter_system/pipeline.py` — `TwoPhaseAnnotationPipeline`.
- `src/run_two_phase_annotation.py` — entry point.
- `src/api/experiments/annotation/two_phase_local.json` — experimento exemplo (`enabled=true`, dataset pequeno como `sst1`/`agnews`, `sample`/1 fold para smoke test).

**Arquivos alterados (aditivo, retrocompatível)**
- `src/api/schemas/annotation_experiment/experiment.py` — +1 campo `class_filter` com `default_factory`.
- `src/systems/llm_annotation_system/pipeline.py` — `AnnotationConfig._apply_experiment` lê `exp.class_filter` (só armazena; caminho legado não usa).
- `src/systems/llm_annotation_system/annotation/llm_annotator.py` — repassa `candidates_by_text_id` opcional ao engine.
- `src/systems/llm_annotation_system/annotation/annotation_engine.py` — helper `_render_categories_str(indices)` extraído; prompt+chave-de-cache por-texto quando há candidatas; caminho `None` inalterado.
- `src/systems/llm_annotation_system/core/response_processor.py` — `valid_categories` opcional no parse.
- `src/systems/instance_selection_system/core/text_vectorizer.py` — expor `fit`/`transform` separados (aditivo; `fit_transform` mantém-se). *Alternativa sem tocar aqui:* o filtro instancia seu próprio `TfidfVectorizer` — decidir na etapa 2.

**Nenhuma** renomeação em massa. **Nenhuma** mudança no caminho legado de anotação.

---

## Plano de execução (etapas pequenas e testáveis)

1. **Config + schema (sem lógica).** Criar `ClassFilterConfig` e plugar em `ExperimentRequest`. **Teste:** carregar um JSON antigo → idêntico; carregar um JSON com bloco `class_filter` → parseia; default `enabled=False`.
2. **Filtro plugável.** `ClassFilterBase` + `LogisticRegressionClassFilter` + `factory` + `CLASS_FILTER_STRATEGIES`. **Teste unitário:** `fit`/`predict_proba` num toy dataset; `classes_` corretas; sem vazamento (vetor `fit` só no fit part).
3. **Split interno + recall@k.** `inner_split` e `recall_at_k_sweep`/`topk_candidates`. **Teste:** determinismo por seed; `recall@k` monotônico não-decrescente em k; `recall@n_classes == 1.0`.
4. **Injeção por-texto no engine.** `_render_categories_str`, prompt+cache-key condicionais, `valid_categories` no processor. **Teste:** com `candidates=None`, prompt e chave de cache byte-idênticos ao baseline (snapshot); com candidatas, prompt mostra só o subconjunto em ordem canônica e a chave muda; resposta fora do k → `-1`.
5. **Orquestrador + entry point.** `TwoPhaseAnnotationPipeline`, `run_two_phase_annotation.py`, experimento exemplo. **Teste (smoke):** 1 fold, dataset pequeno, `k=3`; gera `recall_at_k.csv`, `annotations.csv`, `model_metrics.csv`; held-out do LLM = held-out do recall@k.
6. **Iteração de todos os folds + agregação.** **Teste:** roda os N folds; relatório agregado média±desvio; baseline (`filter_active=False`) sobre os mesmos held-outs para comparação.
7. **Documentação** curta no `docs/` referenciando este plano.

Cada etapa é verificável isoladamente; nada depende de etapas futuras para rodar.

---

## Riscos e pontos de vazamento (e mitigação)

1. **Cache colidindo baseline × filtrado** *(crítico)* — chave atual ignora categorias. **Mitigação:** incluir `"cand"` na chave quando há filtro; baseline mantém `{"rep": rep}`.
2. **Vazamento fit → held-out** *(crítico p/ validade)* — TF-IDF/LR não podem ver o held-out. **Mitigação:** `fit` só em `df_fit`, `transform` no held-out; nunca `fit_transform` no held-out. Cobrir com teste.
3. **Ranking/score vazando no prompt** — proibido. **Mitigação:** candidatas **ordenadas por índice canônico**, nunca por probabilidade; nenhum score/rank no prompt; mesmas descrições de hoje.
4. **`text_id` inconsistente no join** — memória `[[text-id-canonical-key-mismatch]]`. **Mitigação:** computar `text_id` com `get_text_id_from_text` sobre **o mesmo texto** usado na anotação; validar unicidade e cobertura do merge (assert 100% dos held-out têm candidatas).
5. **Teto vs realidade** — `recall@k` deve ser medido no **mesmo** held-out e com o **mesmo** LR que geram as candidatas do LLM, senão o "teto" não bate. **Mitigação:** um único `predict_proba` alimenta recall@k e candidatas.
6. **Datasets sem `LABEL_MEANINGS` nomeado** (twitter/medline) — **Mitigação:** reusar o mesmo fallback do `_prepare_template`, só restrito ao subconjunto; formato idêntico ao baseline.
7. **Test fold externo** — não usar aqui, para não contaminar a avaliação do fine-tuning downstream.
8. **Regressão no baseline** — **Mitigação:** teste de snapshot garantindo prompt e chave de cache idênticos quando `candidates=None`.

---

## Pontos em aberto (precisam de decisão sua)

1. **`TextVectorizer`:** adicionar `fit`/`transform` a ele (reuso máximo) **ou** o filtro instanciar seu próprio `TfidfVectorizer` (zero toque em código existente)? Recomendo o segundo para isolamento total na 1ª versão.
2. **Baseline comparável:** gerar automaticamente a rodada baseline (todas as classes) sobre o mesmo held-out dentro do orquestrador, **ou** deixar como run separado? Recomendo automático (garante mesmos textos).
3. **Local dos resultados:** `results/<dataset>/two_phase/fold_{f}/<timestamp>/` está bom, ou prefere um layout que case com o que os notebooks de análise (`src/notebooks/`) já esperam?
4. **`holdout_fold_index` default = 0** — algum pedaço específico preferido (ex.: o "quinto")?
5. **Estratificação do split interno:** `StratifiedKFold` (semeado, como o BIOIS) — confirma? Alguma classe rara pode inviabilizar estratificação em datasets com muitas classes (reut90/wos); nesse caso caio para `KFold` com aviso. Ok?

---

## Verificação end-to-end (após implementação)

- **Unit:** `pytest` nas etapas 2–4 (filtro, split, recall@k, snapshot de prompt/cache).
- **Smoke:** `python -m src.run_two_phase_annotation` com `two_phase_local.json` (1 fold, dataset pequeno, `k=3`, `cache.enabled=false`) → confere geração de `recall_at_k.csv`, `annotations.csv`, `model_metrics.csv` e alinhamento held-out.
- **Regressão baseline:** `python -m src.run_annotation` (config antiga) → resultado idêntico ao atual; opcionalmente comparar chave de cache/prompt via log.
- **Full:** rodar todos os folds de um dataset médio; validar `recall@k` como teto (acurácia do LLM ≤ recall@k por fold) e o trade-off k vs acerto.
