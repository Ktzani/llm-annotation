# Exemplos de anotação para datasets generalistas

Experimentos prontos para um dataset que **não vem do HuggingFace** — proprietário, interno
ou qualquer pasta registrada em `data/datasets/<nome>/` com um `dataset.json`
(ver [`docs/GUIA_DATASETS.md`](../../../../../docs/GUIA_DATASETS.md) e
`data/datasets/exemplo_tickets/LEIA-ME.md`).

Todos usam o dataset de exemplo `exemplo_tickets`. Para o seu dataset, troque apenas o
`dataset_name` pelo nome da pasta — nenhum outro campo muda por o dataset ser local.

| Arquivo | Para que serve |
|---|---|
| `basico.json` | Menor configuração possível: 3 modelos, 1 repetição, sequencial |
| `completo.json` | Todos os campos da anotação preenchidos, comentados abaixo |
| `few_shot.json` | Prompt few-shot com exemplos rotulados |
| `custom_prompt.json` | Prompt próprio, escrito em português, no lugar dos templates |

Rodar: `POST /annotation` com o conteúdo do arquivo no corpo, ou apontar o
`experiment` em `src/run_annotation.py` para `generalista/<arquivo>` (sem `.json`).

---

## Os campos do experimento

```jsonc
"dataset_name": "exemplo_tickets",   // nome da pasta em data/datasets/

"dataset_config": {
    "split": null,            // arquivo da pasta a usar (ex.: "train_fold_0"); null = o padrão
    "hf_file": null,          // arquivo exato, quando o nome não segue o padrão de split
    "combine_splits": null,   // junta vários arquivos: ["train_fold_0", "train_fold_1"]
    "sample_size": 10,        // amostra para teste; null = dataset inteiro
    "random_state": 42,       // semente da amostragem
    "remove_texts": true      // pula textos já anotados antes (checkpoint do dataset)
},

"models": ["qwen3-8b", "llama3.1-8b"],   // chaves de src/config/llms.py

"prompt_type": "base",   // base | simpler | chain_of_thought | few_shot
"custom_prompt": "",     // se preenchido, substitui o prompt_type inteiro

"annotation": {
    "num_repetitions_per_llm": 3,   // repetições por modelo (consenso interno)
    "use_alternative_params": false,// false | "alt1" | ["base","alt1"] | {"qwen3-8b": "alt2"}
    "model_strategy": "parallel",   // modelos em paralelo ou sequencial
    "rep_strategy": "parallel",     // repetições em paralelo ou sequencial
    "max_concurrent_texts": 5,      // textos simultâneos
    "keep_alive": "24h",            // tempo que o Ollama mantém o modelo na VRAM
    "few_shot_examples": null       // obrigatório só quando prompt_type = "few_shot"
},

"cache": { "enabled": false, "dir": "data/.cache" },
"results": { "intermediate": 50, "dir": "data/results" },  // salva a cada N textos
```

No Docker, troque `dir` por `/app/data/.cache` e `/app/data/results`.

---

## Os tipos de prompt

As classes vêm do `label_meanings` do manifesto e são renderizadas como `- <código>: <nome>`;
a LLM responde com o código. O `prompt` do manifesto (ex.: `"Support ticket"`) entra como o
nome do tipo de texto dentro do prompt.

| `prompt_type` | Quando usar |
|---|---|
| `simpler` | Uma frase direta. É o mais usado nos experimentos da tese |
| `base` | Instruções detalhadas de anotador especialista |
| `chain_of_thought` | Pede a análise passo a passo antes da classificação |
| `few_shot` | Mostra exemplos rotulados antes do texto a classificar |

### Few-shot

Os exemplos vão no experimento, em `annotation.few_shot_examples`:

```json
"prompt_type": "few_shot",
"annotation": {
    "few_shot_examples": [
        {"text": "A cobrança veio duplicada neste mês.", "category": "0"},
        {"text": "O sistema trava na tela de login.", "category": "suporte técnico"}
    ]
}
```

- `category` aceita o **código** (`"0"`) ou o **nome** da classe (`"suporte técnico"`,
  sem diferenciar maiúsculas). Nos dois casos o prompt mostra o código, que é o formato
  pedido na resposta.
- Os mesmos exemplos valem para todos os modelos e todos os textos — eles entram uma vez
  no template, no lugar do `{examples}`.
- Escolha exemplos que **não estão** no conjunto a anotar, para não medir o modelo no que
  ele já viu.
- Sem `few_shot_examples`, a API recusa o experimento com 422.
- Few-shot **não combina** com `class_filter.enabled=true`: a anotação em 2 fases exige o
  LLM zero-shot em relação aos dados rotulados (também recusado com 422).

O prompt final fica assim (trecho):

```
You are an expert data annotator. Below are examples of correctly classified texts:

Text: A cobrança veio duplicada neste mês.
Category: 0

Text: O sistema trava na tela de login.
Category: 1

Now, classify the following text using the same criteria:

**Text to classify (Support ticket):**
{text}

**Available Categories:**
- 0: financeiro
- 1: suporte técnico
- 2: comercial
```

### Prompt próprio (`custom_prompt.json`)

`custom_prompt` substitui o template inteiro e **ignora o `prompt_type`** (o campo continua
no JSON só porque é obrigatório). Útil para escrever o prompt no idioma do dataset ou com as
regras do seu domínio:

```json
"custom_prompt": "Você é um analista de atendimento...\n\n{description}: {text}\n\nCategorias disponíveis:\n{categories}\n\nCategoria (apenas o número):"
```

- Marcadores disponíveis: `{text}`, `{categories}`, `{description}` e `{description_lower}`.
  Qualquer outra chave entre `{}` quebra a formatação do template.
- O `{text}` é obrigatório — é onde cada texto do dataset entra.
- Peça **só o número da categoria**: a resposta passa pelo mesmo extrator dos outros prompts,
  e texto solto vira rótulo inválido (`-1`).
- Quebras de linha no JSON vão como `\n`.
