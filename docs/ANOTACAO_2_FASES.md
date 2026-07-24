# Anotação em 2 Fases (Filtro de Classes + LLM Zero-Shot)

Reduz o espaço de classes apresentado ao LLM sem tirá-lo do regime zero-shot.
Ver o desenho completo em [`PLANO_ANOTACAO_2_FASES.md`](../PLANO_ANOTACAO_2_FASES.md).

## Ideia

- **Fase 1 (filtro):** um classificador leve (Regressão Logística sobre TF-IDF)
  treina no conjunto de treino e devolve as **top-k** classes mais prováveis por texto.
- **Fase 2 (LLM):** o LLM anota exatamente como no baseline, mas vê **só as k
  candidatas**, em ordem canônica fixa. Nenhum exemplo rotulado, score ou ranking
  entra no prompt — a única mudança é o tamanho da lista de classes.

Por fold externo: o treino é dividido em `n_inner_folds` pedaços; o classificador
treina em `n-1` e o pedaço **held-out** (que tem ground-truth) é onde medimos o
`recall@k` (teto de acurácia da Fase 2) **e** onde o LLM anota. Sem vazamento: o
classificador nunca vê o pedaço que prevê. O `test_fold` externo não é tocado.

## Como rodar

1. Configure o experimento em `src/api/experiments/annotation/two_phase_local.json`
   (exemplo já incluso). O bloco relevante:

   ```json
   "class_filter": {
     "enabled": true,
     "method": "logistic_regression",
     "k": 3,
     "k_sweep": [2, 3, 4],
     "n_inner_folds": 5,
     "holdout_fold_index": 0,
     "random_state": 42
   }
   ```

2. Execute por um dos dois caminhos:

   **CLI:**
   ```bash
   python -m src.run_two_phase_annotation
   ```

   **API:** submeta o experimento normalmente (rota de anotação). O runner
   ramifica automaticamente para o pipeline de 2 fases quando
   `class_filter.enabled=true` — com `false` (default), roda o baseline clássico
   sem qualquer mudança.

   `run_baseline` (default `false`) também anota o mesmo held-out com **todas** as
   classes, para comparação apples-to-apples. Vale para os dois caminhos (é lido
   da config, não hardcoded). Dobra o custo de chamadas ao LLM.

## Saídas (`data/results/<dataset>/two_phase/<timestamp>/`)

- `recall_at_k_all_folds.csv`, `recall_at_k_aggregated.csv` — teto por fold e agregado.
- `fold_{f}/recall_at_k.csv`, `fold_{f}/filter_report.json` — Fase 1 por fold.
- `fold_{f}/filtered/annotations.csv` + `model_metrics.csv` — Fase 2 (espaço reduzido).
- `fold_{f}/baseline/...` — baseline (todas as classes), se `run_baseline=True`.

## Retrocompatibilidade

- `class_filter.enabled=False` (default) → o pipeline de anotação roda **idêntico**
  ao baseline. O `run_annotation.py` clássico não muda.
- O filtro é plugável: novos métodos entram em `src/config/class_filter.py` +
  `src/systems/class_filter_system/factory.py` sem alterar o restante.
- `k` é configurável e varrível (`k` na Fase 2; `k_sweep` no relatório).
