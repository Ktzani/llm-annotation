# Exemplo de dataset local: `exemplo_tickets`

30 tickets de atendimento (10 por classe), classificados em **financeiro**, **suporte técnico**
e **comercial**. Serve de modelo para cadastrar um dataset proprietário, ou qualquer dataset
que não está no HuggingFace.

## Estrutura da pasta

```
data/datasets/exemplo_tickets/     ← o nome da pasta vira o dataset_name
├── dataset.json                   ← manifesto: diz como ler os arquivos
├── data.csv                       ← todos os textos (é o que a anotação usa)
├── train_fold_0.csv / test_fold_0.csv   ← folds (só para fine-tuning e anotação em 2 fases)
└── train_fold_1.csv / test_fold_1.csv
```

Toda pasta em `data/datasets/` que tiver um `dataset.json` é registrada sozinha. Não é
preciso editar código nem reiniciar a API.

## O manifesto (`dataset.json`)

```json
{
    "path": "data.csv",                  // arquivo principal (relativo a esta pasta)
    "text_column": "texto",              // coluna com o texto a classificar
    "label_column": "classe",            // coluna com o rótulo verdadeiro; null se não houver
    "label_meanings": {                  // código → nome de cada classe
        "0": "financeiro",
        "1": "suporte técnico",
        "2": "comercial"
    },
    "prompt": "Support ticket",          // como o texto é chamado no prompt
    "description": "...",                // livre, só documentação
    "read_kwargs": {"sep": ";"}          // opções do pandas.read_csv (aqui: separador ;)
}
```

(Comentários `//` não são permitidos em JSON; eles estão aqui só para explicar.)

- **`label_meanings`** é o que a LLM vê no prompt e responde com o código:
  ```
  - 0: financeiro
  - 1: suporte técnico
  - 2: comercial
  ```
- **Coluna de rótulo**: pode ter o **nome** da classe (como neste exemplo, e sem diferenciar
  maiúsculas) ou o **código** (`0`, `1`, `2`). Os dois formatos são convertidos para o código.
  Se aparecer um valor que não está em `label_meanings`, o carregamento falha e lista esses valores.
- **Sem ground truth?** Use `"label_column": null`. As classes válidas saem de `label_meanings`
  e as métricas de acurácia são puladas.
- **CSV exportado do Excel** com acentos quebrados: `"read_kwargs": {"sep": ";", "encoding": "latin-1"}`.
- Formatos aceitos: `.csv`, `.tsv`, `.parquet`, `.json`, `.jsonl`, `.xlsx`, `.xls`.

## Como rodar a anotação

O experimento pronto está em `src/api/experiments/annotation/exemplo_dataset_local.json`.
O único campo que muda em relação a um dataset do HF é `"dataset_name": "exemplo_tickets"`.

**Via script:** em `src/run_annotation.py`, troque
```python
run_type = "dataset"
experiment = "exemplo_dataset_local"
```
e rode `poetry run python -m src.run_annotation`.

**Via API:** faça `POST /experiments` com o conteúdo do JSON no corpo. No Docker, troque os
caminhos de `cache.dir` e `results.dir` para `/app/data/.cache` e `/app/data/results`. O
dataset já aparece em `/app/data/datasets` pelo volume montado.

Os resultados saem em `data/results/exemplo_tickets/<data>/annotations.csv`, com as métricas
contra a coluna `classe`.

## Fine-tuning e anotação em 2 fases

Esses fluxos procuram `train_fold_{k}` e `test_fold_{k}` na pasta. Os folds são descobertos
em sequência (0, 1, ...) até faltar um. Neste exemplo são 2 folds estratificados: cada
`test_fold` tem 5 textos de cada classe, e o `train_fold` do mesmo k tem os outros 15.

- Os textos dos folds precisam ser **idênticos** aos de `data.csv`, porque o alinhamento é
  feito pelo hash do texto.
- Se o experimento pedir `train_fold_0.parquet`, a pasta pode ter `train_fold_0.csv`: vale o
  mesmo nome com qualquer extensão aceita.
- Sem folds, a anotação funciona normalmente, mas o fine-tuning não tem onde treinar.

Depois de anotar, basta usar `"dataset_name": "exemplo_tickets"` no JSON de fine-tuning
(ver `src/api/experiments/fine_tuning/local_job.json`).

## Para cadastrar o seu dataset

1. Copie esta pasta com outro nome, ex.: `data/datasets/meus_contratos/`.
2. Coloque seus arquivos e ajuste `text_column`, `label_column`, `label_meanings` e `prompt`.
3. Confira se foi reconhecido:
   ```python
   from src.utils.data_loader import load_hf_dataset
   from src.api.schemas.annotation_experiment.dataset import DatasetConfig
   texts, categorias, gt = load_hf_dataset("meus_contratos", "data/.cache", DatasetConfig())
   ```
   ou consulte `GET /datasets` na API.
4. Use `"sample_size": 10` no experimento para um primeiro teste rápido.

A pasta `data/` está no `.gitignore`, então dados proprietários não vão para o repositório. A
única exceção é este exemplo, liberado no `.gitignore`. Datasets novos em `data/datasets/`
ficam ignorados automaticamente.
