# Roteiro do vídeo — Anotação binária na máquina do laboratório

**Duração estimada:** 20–25 minutos. **Material de apoio:** `docs/TUTORIAL_ANOTACAO_BINARIA.md`
(cite no começo: tudo o que aparece no vídeo está escrito lá).

**Antes de gravar:**
- VS Code conectado à máquina (`lab-kenia`), terminal aberto em `~/llm/llm-annotation`.
- Nenhuma anotação rodando (ou disposto a pará-la: ela retoma depois).
- Navegador com uma aba em branco para o Swagger.
- Separe um cenário pequeno para a demonstração: `cenario_teste.json` com `"sample_size": 5`
  (5 mensagens, poucos minutos), para mostrar o ciclo inteiro até o consenso.

Cada cena tem: **Tela** (o que mostrar), **Fale** (pontos a explicar) e **Faça** (ação).

---

## Cena 1 — Abertura (1 min)

**Tela:** o tutorial aberto no VS Code.

**Fale:**
- Objetivo: rodar a anotação binária sozinha, do início ao fim, sem mexer em código.
- O que é a anotação binária: cada modelo responde SIM ou NÃO para cada classe, uma pergunta por classe.
- Três modelos (qwen3, mistral, deepseek), sem raciocínio, e no final um consenso por maioria.
- Duas etapas: anotação e consenso.

---

## Cena 2 — Conectar na máquina (2 min)

**Tela:** VS Code → Ctrl+Shift+P → "Remote-SSH: Connect to Host" → `lab-kenia`.

**Fale:**
- O acesso passa pelo servidor do DCC e de lá vai para a máquina do laboratório; são duas senhas.
- A configuração do SSH está no tutorial (seção 3).

**Faça:** conecte, abra a pasta `~/llm/llm-annotation` e um terminal.

---

## Cena 3 — Tour das pastas (3 min)

**Tela:** Explorer do VS Code.

**Mostre, nesta ordem:**
1. `~/llm/` — a casa de tudo: o projeto, o `env.sh`, o Ollama e os modelos.
2. `data/datasets/CENARIO_TESTE/` — um dataset: o arquivo de mensagens, o `dataset.json` (como ler o arquivo, nomes das classes) e o `class_prompts.json` (descrição e exemplos de cada classe).
3. `src/api/experiments/binary_annotation/` — um JSON por cenário; é o que vai ser colado no Swagger.
4. `data/results/CENARIO_TESTE/binary/` — onde saem os resultados (volta aqui na cena 9).

**Fale:** para um cenário novo, basta criar a pasta do dataset e o JSON do experimento; nada de código.

---

## Cena 4 — O Ollama e os parâmetros (3 min)

**Tela:** `~/llm/env.sh` aberto.

**Fale:**
- O Ollama é o servidor que roda os modelos na GPU; a API manda as perguntas para ele.
- Usamos a **versão 0.17.1, que não pode ser atualizada**: nas versões novas, o deepseek ignora o raciocínio desligado e fica muito lento.
- Percorra a tabela de parâmetros (tutorial, seção 4.2), destacando:
  - `OLLAMA_NUM_PARALLEL=15`: quantas perguntas cada modelo atende ao mesmo tempo.
  - `OLLAMA_CONTEXT_LENGTH=6000`: cabe o maior prompt (~2700 tokens) com folga.
  - `OLLAMA_MAX_LOADED_MODELS=3`: os três modelos ficam juntos na GPU.

---

## Cena 5 — Ligar o Ollama, os modelos e a API (4 min)

**Tela:** terminal.

**Fale antes:** o bloco derruba o que estiver rodando e sobe tudo de novo; uma anotação em andamento para, mas retoma depois.

**Faça:** cole o bloco da seção 5 do tutorial e comente cada parte enquanto roda:
1. `ollama --version` → **0.17.1, sem "Warning"** (se aparecer Warning, há outro Ollama na porta).
2. `ollama ps` → **os 3 modelos, "100% GPU"**.
3. O log da API → `Uvicorn running on http://0.0.0.0:8001`.

**Mostre o tmux:** saia com **Ctrl+B, depois d**; `tmux ls` mostra `api` e `ollama`; `tmux attach -t api` volta ao log.
**Fale:** o tmux mantém tudo rodando mesmo fechando o VS Code.

---

## Cena 6 — O Swagger e o JSON do experimento (3 min)

**Tela:** aba **Ports** do VS Code (a 8001) e depois **http://localhost:8001/docs**.

**Fale:**
- O Swagger é a página para disparar e acompanhar, sem programar.
- Mostre as rotas: `POST /binary-annotation/`, `GET /annotation/{id}`, `POST /annotation/{id}/cancel`, `POST /binary-annotation/consensus`.
- Abra o `cenario_teste.json` e explique os campos que se mudam (tutorial, seção 7):
  `dataset_name`, `sample_size`, `keep_columns`, `max_concurrent_classes` (**10, e nunca acima do `OLLAMA_NUM_PARALLEL`**).
- Paralelismo: as mensagens vão uma por vez; dentro de cada uma, cada modelo responde até 10 classes ao mesmo tempo. A GPU fica 100% ocupada, por isso mexer no paralelismo quase não muda a velocidade (testamos 5 e 10: 10 foi um pouco melhor).

---

## Cena 7 — Disparar e acompanhar (3 min)

**Faça:**
1. `POST /binary-annotation/` → Try it out → cole o JSON (com `"sample_size": 5`) → Execute.
2. Copie o `annotation_id` (**fale: guardem esse id**).
3. `GET /annotation/{id}` → status `running`.
4. No terminal, `tmux attach -t api`: mostre `Já no checkpoint … | A fazer …`, os tempos de cada modelo e a barra de progresso com o tempo restante.

**Fale:** ~15 s por mensagem com 23 classes; o CENARIO_TESTE completo leva ~3 h, os cenários completos, dias.

---

## Cena 8 — Pausar e retomar (2 min)

**Faça:** `POST /annotation/{id}/cancel` → `GET /annotation/{id}` mostra `cancelled`. Dispare o mesmo JSON de novo e mostre no log `Já no checkpoint: N`.

**Fale:**
- O progresso fica no `intermediate.csv`; disparar o mesmo JSON continua de onde parou.
- Para recomeçar do zero (mudou prompt, modelo ou versão do Ollama): renomeie o `intermediate.csv` (tutorial, seção 9).
- Não editar arquivos `.py` com uma anotação rodando: a API reinicia e a anotação para.

---

## Cena 9 — Consenso e resultados (3 min)

**Faça (com a amostra de 5 mensagens concluída):**
1. Veja a pasta da execução em `data/results/CENARIO_TESTE/binary/<data>/`.
2. `POST /binary-annotation/consensus` com o corpo do tutorial (seção 10), usando essa data.
   **Fale: sempre `"results_dir": "data/results"`** (o padrão aponta para outro computador).
3. `GET /consensus/{job_id}` até `completed`.
4. Abra o `binary_labels.csv`: uma linha por mensagem, uma coluna por classe, **1 = SIM, 0 = NÃO, -1 = inválida**.
5. Mostre também `binary_labels_<modelo>.csv` (um por modelo) e a pasta `summary/`.

---

## Cena 10 — Quando algo dá errado (2 min)

**Tela:** tutorial, seção 11 (tabela de problemas).

**Fale dos três mais prováveis:**
1. `Warning: client version` → outro Ollama na porta.
2. `ollama ps` sem os 3 modelos → faltou memória na GPU.
3. Deepseek muito lento → conferir se o Ollama é o 0.17.1.

**Feche:** se a máquina reiniciar, rode o bloco da seção 5 de novo; o resto está no tutorial.

---

## Depois de gravar

- Volte o `cenario_teste.json` para `"sample_size": null`.
- Se a demonstração anotou o CENARIO_TESTE, renomeie o `intermediate.csv` antes da execução de verdade.
