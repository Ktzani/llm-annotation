# Tutorial — Anotação binária com LLMs na máquina do laboratório

Este guia é para quem vai **rodar a anotação binária sozinho** na máquina do laboratório
(Neptune, RTX 6000 Ada de 48 GB). Não é preciso mexer em código: tudo é feito pelo terminal
(para ligar os serviços) e pelo navegador (Swagger, para disparar e acompanhar).

---

## 1. O que a anotação binária faz

Em vez de pedir à LLM "qual é a classe desta mensagem?", cada modelo responde **SIM ou NÃO
para cada classe**, com um prompt próprio por classe.

- Um dataset com 23 classes e 3 modelos gera **23 × 3 = 69 perguntas por mensagem**.
- Os 3 modelos usados: **qwen3-8b**, **mistral-7b** e **deepseek-r1-8b**, todos com o
  raciocínio (*thinking*) **desligado**: respondem direto `{"resposta": "SIM"}` ou `{"resposta": "NAO"}`.
- Depois da anotação, o **consenso** junta os 3 votos de cada par mensagem × classe
  (vence a maioria) e gera a planilha final.

São duas etapas, sempre nesta ordem:

| Etapa | O que faz | Rota da API |
|---|---|---|
| 1. Anotação | Pergunta SIM/NÃO a cada modelo, para cada classe | `POST /binary-annotation/` |
| 2. Consenso | Junta os votos dos 3 modelos | `POST /binary-annotation/consensus` |

---

## 2. Onde fica cada coisa

Tudo fica na pasta `~/llm` do usuário `kenia`:

| Caminho | O que é |
|---|---|
| `~/llm/llm-annotation/` | O projeto (código, experimentos, dados e resultados) |
| `~/llm/env.sh` | Variáveis de ambiente (configuração do Ollama e do Python) |
| `~/llm/ollama/` | O Ollama usado pelo projeto (**versão 0.17.1**) |
| `~/llm/ollama-models/` | Os modelos baixados (qwen3, mistral, deepseek) |

Dentro do projeto:

| Caminho | O que é |
|---|---|
| `data/datasets/<NOME>/` | Um dataset: `dataset.json`, o arquivo de mensagens e o `class_prompts.json` |
| `src/api/experiments/binary_annotation/` | Os JSONs dos experimentos binários (um por cenário) |
| `data/results/<NOME>/binary/` | Os resultados de cada dataset |

---

## 3. Como acessar a máquina

Pelo **VS Code com a extensão Remote - SSH**. O acesso passa pelo servidor do DCC
(`login.dcc.ufmg.br`) e de lá vai para a máquina do laboratório (`150.164.214.40`).
No `~/.ssh/config` do seu computador:

```
Host dcc
    HostName login.dcc.ufmg.br
    User <seu usuário do DCC>

Host lab-kenia
    HostName 150.164.214.40
    User kenia
    ProxyJump dcc
```

No VS Code: **Ctrl+Shift+P → "Remote-SSH: Connect to Host…" → `lab-kenia`**. São pedidas
duas senhas: a do DCC e a da máquina.

**Swagger (a página onde se dispara a anotação):** a API roda na porta **8001** da máquina.
O VS Code redireciona essa porta para o seu computador; confira na aba **Ports** (ao lado do
Terminal) e abra **http://localhost:8001/docs**.

---

## 4. Conceitos que importam

### 4.1 Ollama

O Ollama é o **servidor que roda os modelos** na GPU. A API do projeto manda as perguntas
para ele em `localhost:11434`.

- Usamos uma instalação própria em `~/llm/ollama`, na **versão 0.17.1**. **Não atualize.**
  Nas versões 0.40.x, o pedido para desligar o raciocínio é ignorado pelo deepseek-r1: ele
  continua "pensando" escondido (respostas 10–100× mais lentas e às vezes sem sentido).
- A máquina tinha um Ollama do sistema (versão antiga) que ocupava a mesma porta. Ele foi
  desligado (`systemctl disable`). Se `ollama --version` mostrar `Warning: client version`,
  existe outro Ollama respondendo no lugar do nosso (ver seção 11).
- O Ollama roda dentro do `tmux` (seção 12), então continua funcionando depois que você fecha
  o VS Code.

### 4.2 Parâmetros (`~/llm/env.sh`)

| Variável | Valor | Para que serve |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | 15 | Máximo de perguntas que **cada modelo** atende ao mesmo tempo |
| `OLLAMA_CONTEXT_LENGTH` | 6000 | Tamanho máximo (em tokens) de cada pergunta + resposta. O maior prompt dos cenários tem ~2700 tokens |
| `OLLAMA_MAX_LOADED_MODELS` | 3 | Os 3 modelos ficam carregados na GPU ao mesmo tempo |
| `OLLAMA_KEEP_ALIVE` | 24h | Tempo que um modelo fica na GPU sem uso |
| `OLLAMA_FLASH_ATTENTION` / `OLLAMA_KV_CACHE_TYPE` | 1 / q8_0 | Economia de memória da GPU |
| `OLLAMA_MODELS` | `~/llm/ollama-models` | Onde os modelos ficam salvos |
| `ANNOTATION_RUN_ENV` | lab | Marca em cada resposta em qual máquina ela foi gerada |

Cada modelo também tem `num_thread` limitado (threads de CPU), para os 3 não disputarem a CPU.

### 4.3 Paralelismo

- As mensagens são anotadas **uma por vez**. Dentro de cada mensagem, cada modelo recebe até
  `max_concurrent_classes` perguntas ao mesmo tempo (uma por classe). Os 3 modelos trabalham juntos.
- **Regra:** `max_concurrent_classes` (no JSON) **≤** `OLLAMA_NUM_PARALLEL` (no `env.sh`).
- **Valor usado: 10** (o `OLLAMA_NUM_PARALLEL=15` é só o teto). A GPU fica 100% ocupada, então mexer no
  paralelismo quase não muda a velocidade: com 10, ~15,5 s por mensagem; com 5, ~16,3 s. Use o
  mesmo valor do início ao fim de um cenário, porque o tempo de cada resposta
  (`annotation_time_sec`) depende dele.
- Uso de GPU com os 3 modelos carregados: ~36 GB de 48 GB.

### 4.4 Modelos e raciocínio

O raciocínio do qwen3 e do deepseek-r1 está desligado na configuração do projeto
(`src/config/llms.py`, `"reasoning": False`). O mistral não tem raciocínio. As demais
configurações (temperatura etc.) são as padrão de cada modelo.

---

## 5. Ligar tudo (Ollama + modelos + API)

Abra um terminal no VS Code conectado à máquina e cole o bloco inteiro.
**Atenção:** ele derruba a API e o Ollama que estiverem rodando. Uma anotação em andamento
para, mas o que já foi anotado fica salvo e ela continua de onde parou ao ser disparada de novo.

```bash
# derruba o que estiver rodando
tmux kill-session -t api 2>/dev/null; tmux kill-session -t ollama 2>/dev/null
source ~/llm/env.sh

# sobe o Ollama
tmux new-session -d -s ollama "source ~/llm/env.sh; CUDA_VISIBLE_DEVICES=0 ~/llm/ollama/bin/ollama serve; exec bash"
until curl -s localhost:11434/api/version > /dev/null; do sleep 1; done
ollama --version                      # tem que ser 0.17.1, sem "Warning"

# carrega os 3 modelos na GPU
for m in qwen3:8b mistral:7b deepseek-r1:8b; do curl -s localhost:11434/api/generate -d "{\"model\":\"$m\"}" > /dev/null; done
ollama ps                             # os 3, "100% GPU", CONTEXT 6000

# sobe a API (na porta 8001) e já mostra o log dela
tmux new-session -s api "cd ~/llm/llm-annotation && source ~/llm/env.sh && OMP_NUM_THREADS=\$(( \$(nproc) / 5 )) TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001 --reload --reload-dir src; exec bash"
```

Quando aparecer `Uvicorn running on http://0.0.0.0:8001`, saia do log sem parar a API
(**Ctrl+B, depois d**) e abra **http://localhost:8001/docs**.

---

## 6. Preparar um dataset

Cada dataset é uma pasta em `data/datasets/`. **O nome da pasta é o `dataset_name`.**

```
data/datasets/CENARIO_X/
├── dataset.json          ← como ler o arquivo
├── mensagens.csv         ← as mensagens
└── class_prompts.json    ← o prompt de cada classe
```

### `dataset.json`

```json
{
    "path": "mensagens.csv",
    "text_column": "message_content",
    "label_column": null,
    "label_meanings": {
        "0": "Nome da classe 0",
        "1": "Nome da classe 1"
    },
    "prompt": "mensagem do chat ao vivo da Twitch",
    "description": "Texto livre, só documentação"
}
```

- `text_column`: a coluna com a mensagem.
- `label_column`: a coluna com o rótulo verdadeiro, ou `null` se não houver (sem rótulo, as métricas de acerto são puladas).
- `label_meanings`: o código e o nome de **todas** as classes.
- `prompt`: como a mensagem é chamada dentro do prompt.
- CSV separado por `;` ou com acentos quebrados: acrescente `"read_kwargs": {"sep": ";", "encoding": "latin-1"}`.

A pasta é reconhecida sozinha: não é preciso reiniciar a API.

### `class_prompts.json`

Toda classe do `label_meanings` precisa de um prompt (a chave pode ser o código ou o nome).
O jeito mais simples é dar **descrição + exemplos**, que entram no prompt padrão:

```json
{
    "0": {
        "description": "Descrição do rótulo, como no ClassInd.",
        "examples": [
            "Situação que ilustra o rótulo.",
            {"text": "mensagem de exemplo", "answer": "SIM"},
            {"text": "outra mensagem", "answer": "NAO"}
        ]
    }
}
```

A resposta é lida pela primeira palavra SIM/NÃO; qualquer outra coisa vira inválida (`-1`).
Se aumentar muito as descrições ou os exemplos, o prompt pode passar do contexto de 6000 tokens.

---

## 7. O JSON do experimento binário

Fica em `src/api/experiments/binary_annotation/` (um por cenário: `cenario_teste.json`,
`cenario_original_13.json` etc.). É o conteúdo que se cola no Swagger.

```json
{
    "dataset_name": "CENARIO_TESTE",
    "dataset_config": { "sample_size": null, "random_state": 42 },
    "models": ["qwen3-8b", "mistral-7b", "deepseek-r1-8b"],
    "keep_columns": ["message_id"],
    "class_prompts_file": "class_prompts.json",
    "custom_prompt": null,
    "annotation": {
        "use_alternative_params": false,
        "model_strategy": "parallel",
        "class_strategy": "parallel",
        "max_concurrent_classes": 10,
        "keep_alive": "24h"
    },
    "cache": { "dir": "data/.cache" },
    "results": { "intermediate": 100, "dir": "data/results" }
}
```

| Campo | O que mudar |
|---|---|
| `dataset_name` | O nome da pasta do dataset |
| `dataset_config.sample_size` | `null` = todas as mensagens; um número = amostra (bom para testar) |
| `models` | Os 3 modelos (não mude sem combinar) |
| `keep_columns` | Colunas do dataset copiadas para a planilha final (ex.: `message_id`) |
| `max_concurrent_classes` | Paralelismo (seção 4.3). Usado: 10, nunca acima do `OLLAMA_NUM_PARALLEL` |
| `results.intermediate` | Salva o progresso a cada N mensagens |
| `cache.dir` / `results.dir` | Deixe `data/.cache` e `data/results` |

---

## 8. Rodar a anotação

1. No Swagger (**http://localhost:8001/docs**), abra **`POST /binary-annotation/`** → **Try it out**.
2. Cole o conteúdo do JSON do cenário → **Execute**.
3. **Anote o `annotation_id`** da resposta. É com ele que se acompanha e cancela.

**Acompanhar:**
- Pelo Swagger: **`GET /annotation/{annotation_id}`** (status, progresso e mensagem).
- Pelo log: no terminal, `tmux attach -t api` (sair: **Ctrl+B, depois d**). No início aparece
  `Total de perguntas … | Já no checkpoint … | A fazer …`, e depois a barra `Anotando (binário)`
  com o tempo restante.

**Quanto demora:** ~15 s por mensagem com 23 classes. O CENARIO_TESTE (~700 mensagens) leva
cerca de 3 horas; os cenários completos (39–56 mil mensagens), vários dias.

---

## 9. Pausar, retomar e recomeçar

- **Pausar:** `POST /annotation/{annotation_id}/cancel`. O que já foi respondido fica salvo.
- **Retomar:** dispare **o mesmo JSON** de novo. Só são feitas as perguntas que faltam
  (o log mostra `Já no checkpoint: N`).
- **Recomeçar do zero:** renomeie o checkpoint do dataset antes de disparar:
  ```bash
  cd ~/llm/llm-annotation
  mv data/results/<NOME>/binary/intermediate.csv data/results/<NOME>/binary/intermediate_antigo.csv
  ```
  Faça isso sempre que mudar algo que altera as respostas (modelos, prompts, versão do Ollama).

**Não edite arquivos `.py` dentro de `src/` com uma anotação rodando:** a API reinicia
sozinha e a anotação em andamento é interrompida (os JSONs podem ser editados).

---

## 10. Consenso e resultados

Com a anotação concluída, rode o consenso: **`POST /binary-annotation/consensus`** com

```json
{
    "dataset_name": "CENARIO_TESTE",
    "results_dir": "data/results",
    "specific_date": "2026-10-22_16-35-47",
    "consensus_threshold": 0.8,
    "consensus_strategy": "majority_vote"
}
```

- `specific_date` é a pasta da execução dentro de `data/results/<NOME>/binary/`. Use a data
  explícita (com `"latest"` ele pega a pasta modificada por último, que pode ser outra).
- **Sempre informe `"results_dir": "data/results"`.** O valor padrão aponta para um caminho de
  outro computador.
- Status em **`GET /consensus/{job_id}`**.

### O que sai

```
data/results/<NOME>/binary/
├── intermediate.csv                ← checkpoint: uma linha por (mensagem, classe, modelo)
└── <data>/
    ├── binary_labels_<modelo>.csv  ← respostas de cada modelo: uma coluna por classe
    ├── annotations.csv             ← entrada do consenso
    ├── prompts.json                ← o prompt usado em cada classe
    ├── class_metrics.csv           ← só com rótulo verdadeiro: precisão/recall/F1 por classe
    ├── binary_labels.csv           ← (consenso) resultado final: uma coluna por classe
    ├── consensus/                  ← (consenso) concordância entre modelos, casos problemáticos
    └── summary/                    ← (consenso) alta confiança, revisão necessária, sumário
```

Nas planilhas: **1 = SIM, 0 = NÃO, -1 = resposta inválida**. Cada linha é uma linha do
dataset, com as `keep_columns` (ex.: `message_id`) na frente.

---

## 11. Problemas comuns

| Sintoma | Causa | O que fazer |
|---|---|---|
| `ollama --version` mostra `Warning: client version` | Outro Ollama respondendo na porta 11434 | `ps -eo pid,user,args \| grep "[o]llama serve"`; pare o que não for `~/llm/ollama` |
| `ollama ps` não mostra os 3 modelos | Faltou memória na GPU e o Ollama descarregou um modelo | Reduza `OLLAMA_NUM_PARALLEL` no `env.sh` e religue (seção 5) |
| Deepseek demora 10 s ou mais por pergunta, ou responde lixo | Ollama em versão nova, que não desliga o raciocínio | Confira `ollama --version` = 0.17.1 |
| Erro `peg-native` no log | Ollama antigo do sistema respondendo | Mesmo caso da primeira linha |
| `GET /annotation/` vem vazio | A API reiniciou, ou o Swagger está em outra porta | Confira a aba **Ports** do VS Code (porta 8001); veja o log da API |
| Muitas respostas `-1` | O modelo respondeu fora do formato | Veja os avisos `Resposta sem sim/não reconhecível` no log |

Para conferir as respostas inválidas por modelo:

```bash
cd ~/llm/llm-annotation
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('data/results/CENARIO_TESTE/binary/intermediate.csv')
print(df.groupby('model')['answer'].apply(lambda s: f'{(s == -1).sum()} inválidas de {len(s)}'))"
```

---

## 12. tmux (terminais que continuam rodando)

| Comando | O que faz |
|---|---|
| `tmux ls` | Lista as sessões (`api` e `ollama`) |
| `tmux attach -t api` | Entra na sessão (vê o log) |
| **Ctrl+B, depois d** | Sai da sessão **sem parar** o que está rodando |
| `tmux kill-session -t api` | Para a sessão |

Cuidado: **Ctrl+B, depois D maiúsculo** abre um menu de clientes; saia dele com `q`.

Depois que a máquina reinicia, o Ollama e a API não sobem sozinhos: rode o bloco da seção 5.
