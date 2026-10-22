# Anotação binária (sim/não por classe)

Em vez de pedir a classe do texto, cada modelo responde **sim ou não para cada classe**, com um
prompt próprio por classe. Com 52 classes e 3 modelos, cada texto recebe 52 × 3 perguntas.

O resultado de cada texto é o conjunto de sim/não das classes (não há rótulo único). São duas etapas,
como na anotação multiclasse:

1. **Anotação** (`POST /binary-annotation`): as respostas de cada modelo, uma planilha por modelo.
2. **Consenso** (`POST /binary-annotation/consensus`, depois): o consenso existente (`ConsensusPipeline`,
   sem alteração) sobre cada item mensagem × classe, e a planilha do consenso.

| Arquivo | Para que serve |
|---|---|
| `exemplo_tickets.json` | Experimento com o dataset `exemplo_tickets` e os prompts de `data/datasets/exemplo_tickets/class_prompts.json` |
| `cenario_*.json` | Um por cenário do ClassInd (`data/datasets/CENARIO_*`), com `keep_columns: ["message_id"]` |

Rodar: `POST /binary-annotation` com o conteúdo do arquivo no corpo, ou `mode = "binary"` em
`src/run_annotation.py` com `experiment = "exemplo_tickets"`.

O job entra no mesmo registro da anotação: status em `GET /annotation/{id}` e cancelamento em
`POST /annotation/{id}/cancel`, como na anotação multiclasse.

Consenso: `POST /binary-annotation/consensus` com o corpo do `POST /consensus` sem `categories`
(`dataset_name`, `results_dir`, `specific_date` = a data da execução dentro de `binary/` ou `"latest"`,
`consensus_threshold`, `consensus_strategy`). Status em `GET /consensus/{id}`. Pelo script:
`mode = "binary"` em `src/run_consensus.py`.

---

## Os prompts das classes (`class_prompts.json`)

Fica ao lado do `dataset.json`, em `data/datasets/<nome>/class_prompts.json` (outro nome ou caminho:
`class_prompts_file`; ou os prompts direto no experimento, em `class_prompts`). Toda classe do
`label_meanings` precisa de um prompt. A chave aceita o código (`"0"`) ou o nome da classe.

Cada classe aceita uma de duas formas:

```jsonc
{
    // 1) Descrição + exemplos, preenchidos no template padrão (BINARY_CLASS_PROMPT, o prompt da juíza ClassInd)
    "0": {
        "description": "Descrição do rótulo, como no ClassInd...",
        "examples": [
            "Situação que ilustra o rótulo.",                          // texto livre → "- <texto>"
            {"text": "mensagem de exemplo", "answer": "SIM"}           // → "- Mensagem: ... → Resposta: SIM"
        ]
    },
    // 2) Prompt completo da classe, com {text} onde entra a mensagem
    "2": "... <mensagem>{text}</mensagem> ... Responda {\"resposta\": \"SIM\"} ou {\"resposta\": \"NAO\"}"
}
```

- O template padrão está em `src/config/prompts.py` (`BINARY_CLASS_PROMPT`). Os marcadores dele são
  `{text}` (a mensagem), `{description}` (o campo `prompt` do `dataset.json`, ex.: "mensagem do chat ao
  vivo da Twitch"), `{class_name}` (o nome da classe no `label_meanings`), `{class_description}` e
  `{examples}` (um exemplo por linha).
- `custom_prompt` no experimento substitui esse template. Marcadores disponíveis: `{text}`,
  `{class_name}`, `{class_code}`, `{class_description}`, `{examples}`, `{description}`, `{description_lower}`.
  Um marcador desconhecido (ex.: `{message}`) para a execução com erro. Outras chaves, como o JSON do
  formato de saída, ficam como texto.
- `answer` dos exemplos: `SIM`/`NAO`, `sim`/`não` ou `yes`/`no`. Entram no prompt como escritos.
- Leitura da resposta da LLM (o raciocínio em `<think>`/`<thinking>`/`<reasoning>` é ignorado, mesmo
  sem a tag de abertura ou cortado no meio):
  1. o campo de resposta, com ou sem JSON, markdown e texto em volta: `{"resposta": "SIM"}`,
     `{'resposta': 'nao'}`, `{"resposta": true}`, `Resposta: SIM`, `**Resposta final:** NÃO`,
     `A resposta é SIM`, `{"answer": "yes"}`. Sem `resposta`/`answer`, vale `rotulo`/`decisão`/`label`...
     só se o valor for sim/não. Valores diferentes no mesmo texto (ex.: `"SIM" ou "NAO"`) → `-1`;
  2. sem campo, só uma resposta curta e isolada no começo: `SIM`, `**Não.**`, `Sim, ...`, `SIM - ...`;
  3. qualquer outra coisa → `-1` (ex.: `talvez`, `N/A`, `Não há indícios...`, `No chat...`). Melhor
     inválida do que chutada: "no" e "não" aparecem demais no meio de frases em português.
  Valores aceitos: SIM/S/yes/true/verdadeiro/1 e NÃO/NAO/N/no/false/falso/0, sem diferenciar maiúsculas.

---

## Os campos do experimento

```jsonc
"dataset_config": { "sample_size": null, "random_state": 42, "remove_texts": true },  // como na multiclasse
"keep_columns": ["message_id"],              // colunas do dataset levadas ao binary_labels.csv (ausentes: aviso)
"class_prompts_file": "class_prompts.json",  // relativo a data/datasets/<dataset_name>/ ou absoluto
"annotation": {
    "use_alternative_params": false,  // igual à anotação multiclasse
    "model_strategy": "parallel",     // modelos em paralelo ou sequencial
    "class_strategy": "parallel",     // classes de um texto em paralelo ou sequencial (no lugar das repetições)
    "max_concurrent_classes": 5,      // classes perguntadas ao mesmo tempo, por modelo (os textos vão um por vez)
    "keep_alive": "24h"
},
"cache": { "dir": "data/.cache" },    // só para o download de datasets do HF: não há cache de respostas
"results": { "intermediate": 100, "dir": "data/results" }  // salva a cada N textos
```

Não há repetições por modelo nem cache de respostas: o `intermediate.csv` guarda todas.

---

## Saída

```
<results>/<dataset>/binary/
├── intermediate.csv                 ← checkpoint único do dataset: uma linha por (texto, classe, modelo)
└── <data>/
    ├── binary_labels_<modelo>.csv   ← anotação: uma por modelo, uma coluna por classe (1, 0 ou -1)
    ├── annotations.csv              ← anotação: um item por (texto, classe) com {modelo}_consensus, entrada do consenso
    ├── prompts.json                 ← anotação: prompt usado em cada classe
    ├── class_metrics.csv            ← anotação, só com ground truth: precisão/recall/F1 de cada modelo por classe
    ├── binary_labels.csv            ← consenso: uma coluna por classe com o resolved_annotation do consenso
    ├── consensus/                   ← consenso: artefatos do ConsensusPipeline (dataset_consenso.csv, kappas...)
    └── summary/                     ← consenso: alta_confianca.csv, necessita_revisao.csv, sumario_experimento.json
```

- As planilhas `binary_labels*.csv` têm uma linha por linha do dataset carregado (ex.: 744 mensagens
  do CENARIO_TESTE), com as `keep_columns` na frente (ex.: `message_id`). Textos repetidos (ex.: "ola")
  são anotados uma vez só e repetem o mesmo conjunto em cada linha.
- 1 = SIM, 0 = NAO, -1 = resposta inválida (nem SIM nem NAO reconhecível).
- No consenso, cada item é o par mensagem × classe (o `text_id` do `annotations.csv` é
  `<text_id>:<classe>`). Com 3 modelos e o padrão (`majority_vote`, limiar 0,8), vence a resposta da
  maioria; no empate (uma de cada), o consenso existente fica com a primeira e marca o item como problemático.
- Rodar o mesmo dataset de novo retoma do `intermediate.csv`. Com `remove_texts: true` (como na
  multiclasse), os textos que já estão no checkpoint saem da lista antes de anotar ("Removidos N textos
  já anotados"); as planilhas continuam com todas as linhas do dataset. Com `false`, o checkpoint ainda
  pula as perguntas já respondidas e só faz as que faltam (ex.: um modelo novo adicionado ao experimento).
- Sem ground truth, as métricas são puladas e o resto sai igual. Com ground truth, o do item é 1 quando
  a classe é a verdadeira do texto, e a validação do consenso compara contra isso.
