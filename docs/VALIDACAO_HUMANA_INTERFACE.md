# Interface web da validação humana

Aplicação web (FastAPI + telas em HTML/JS) que conduz a validação humana em rodadas,
usando a mesma amostragem e a mesma estimação do `human_validation_system`.

- **Avaliadores** (`/avaliar`): entram com nome + código, veem só os documentos da rodada
  aberta, na mesma ordem para os três, respondem em duas etapas e podem revisar as próprias
  respostas até a rodada fechar. Cada resposta é gravada na hora.
- **Administrador** (`/admin`): acompanha o progresso de cada avaliador, fecha a rodada
  (só com os três completos), vê os resultados e dispara a próxima rodada.

O avaliador nunca recebe rótulo de referência, anotações das LLMs, grupo ou classe: as rotas
dele só leem a lista cega (`id_anonimo` + texto) e as próprias respostas.

## Estrutura

```
src/systems/human_validation_system/interface/
  api/
    core/        settings (HV_* do .env) e autenticação
    routes/      auth (login), evaluator (avaliação), admin (controle das rodadas)
    schemas/     contratos de entrada/saída das telas
    services/    ResponseStore (SQLite), RoundController, ConsolidatedWorkbookWriter, ...
    server.py    create_app / create_app_from_env
  web/           index.html, avaliar.html, admin.html, css/, js/
tests/human_validation_system/
```

## Configuração (.env)

```
HV_ADMIN_CODE=<código do administrador>
HV_CODE_AVALIADOR_1=<código>
HV_CODE_AVALIADOR_2=<código>
HV_CODE_AVALIADOR_3=<código>
# só para Docker/servidor (localmente ficam no main do run_human_validation.py)
HV_EXPERIMENTS=books:<date>,dblp:<date>
HV_RESULTS_DIR=/app/data/results
```

Nunca versione os códigos reais (o `.env` já está no `.gitignore`).

## Rodar localmente

Em `src/run_human_validation.py`, use `mode = "interface"` (as datas ficam em `experiments`) e rode
o script. A interface sobe em `http://localhost:8001`.

## Ciclo de uma rodada

1. Rode o consenso da pasta do experimento (`run_consensus.py`) antes da 1ª rodada.
   Depois disso, não rode de novo: o registro da amostragem confere o sha256 do CSV.
2. Em `/admin`, **Iniciar primeira rodada**. Cada dataset tem as próprias rodadas.
3. Avaliadores respondem em `/avaliar` (o relógio da rodada aparece para todos). Quem termina antes é
   avisado de que pode aguardar os demais.
4. Quando o último avaliador termina, todos têm 5 minutos para revisar (qualquer alteração reinicia o prazo).
   Depois a rodada fecha sozinha (ou pelo botão **Fechar rodada**): consolida por maioria, classifica os
   desfechos, calcula concordância e intervalos, aplica o critério de parada, guarda a duração e as métricas
   da rodada (só a rodada e acumulado) e atualiza
   `data/validacao_humana/<dataset>/<date>/validacao_consolidada_<dataset>.xlsx`.
5. Se algum grupo não parou, a próxima rodada abre sozinha (só os grupos pendentes são sorteados) e você
   recebe um email para avisar os avaliadores. As duas automações podem ser desligadas no bloco "Automação".

O **Painel por dataset** no `/admin` mostra, para cada rodada fechada, duração, documentos e as métricas
(só daquela rodada ou acumuladas até ela). Os dados ficam no banco (tabela `metricas_rodada`) e em
`estimativas/metricas_por_rodada.csv`; nunca são somados entre datasets.

## Aviso por email

Ao fechar uma rodada, o app envia um email com a duração, o critério de parada por grupo e se a próxima
rodada já foi aberta. Configure no `.env`:

```
HV_NOTIFY_EMAIL=<quem recebe>
HV_SMTP_USER=<conta Gmail que envia>
HV_SMTP_PASSWORD=<senha de app do Gmail>
```

A senha de app é criada em https://myaccount.google.com/apppasswords (exige verificação em duas etapas).
Sem essas variáveis, o aviso fica desativado e só aparece no log.

## Deploy em uma VM (Oracle Always Free, Google e2-micro ou a VM existente)

1. Clone o repositório na VM e crie o `.env` com as variáveis acima.
2. Copie para a VM apenas o necessário de cada experimento:
   `data/results/<dataset>/<date>/consensus/dataset_consenso.csv`
   (e, se já houver rodadas, a pasta `data/validacao_humana/` inteira, que inclui o banco).
3. Suba: `docker compose -f docker/docker-compose.validacao.yml up -d --build`.
4. HTTPS sem domínio próprio: na VM, `cloudflared tunnel --url http://localhost:8001`
   gera um endereço `https://<palavras>.trycloudflare.com` (muda se o túnel reiniciar).
   Com um domínio na Cloudflare, use um túnel nomeado para ter endereço fixo.
5. Envie o endereço e os códigos aos avaliadores (cada um só o próprio código).

## Backup

Tudo da validação humana fica em `data/validacao_humana/` (ao lado de `data/results/`): o banco
`validacao_humana.db` (SQLite) e uma pasta `<dataset>/<date>/` por experimento. Faça cópias
periódicas, de preferência com o servidor parado ou via `sqlite3 validacao_humana.db ".backup copia.db"`.

## Testes

```
poetry run pytest tests/human_validation_system
```

Cobrem: rodada não fecha com avaliador incompleto, maioria (inclusive 1x1x1), os quatro
desfechos com casos de fronteira, a planilha consolidada não perder rodadas anteriores e
nenhuma tela/rota do avaliador expor rótulo, grupo ou classe.
