// Tela do administrador: progresso, abertura/fechamento de rodadas e resultados.
Session.require("admin");

const GROUPS = {
  A: "3 LLMs concordam entre si e divergem da referência",
  B: "2 LLMs divergem da referência e 1 concorda",
  C: "3 LLMs concordam com a referência (controle)",
};
const SITUATIONS = {
  benchmark_correct: "Benchmark correto",
  benchmark_mislabeling: "Benchmark mislabeling",
  genuine_ambiguity: "Ambiguidade genuína",
  insufficient_information: "Informação insuficiente",
};
const PRIMARY = "acerto_referencia";
// Métricas do painel agrupadas por tema (as parecidas ficam lado a lado)
const METRIC_GROUPS = [
  ["Concordância com os rótulos", { acerto_referencia: "Humano = referência", acerto_llm: "Humano = LLMs" }],
  ["Desfechos (RQ4)", {
    benchmark_correct: "Benchmark correto",
    benchmark_mislabeling: "Benchmark mislabeling",
    genuine_ambiguity: "Ambiguidade genuína",
    insufficient_information: "Informação insuficiente",
  }],
  ["Ambiguidade declarada", { outro_rotulo_possivel: "Outro rótulo possível" }],
  ["Concordância entre avaliadores", {
    kappa_fleiss: "κ de Fleiss",
    acordo_unanime: "Acordo unânime",
    acordo_par_a_par: "Acordo par a par",
  }],
];
const AGREEMENT = new Set(["kappa_fleiss", "acordo_unanime", "acordo_par_a_par"]);
const busy = new Set();
const panelChoice = {};
let lastPanels = [];

function duration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const minutes = Math.max(0, Math.floor(seconds / 60));
  const h = Math.floor(minutes / 60);
  return h ? `${h}h ${String(minutes % 60).padStart(2, "0")}min` : `${minutes}min`;
}
const since = (iso) => duration((Date.now() - new Date(iso).getTime()) / 1000);
const when = (iso) => (iso ? new Date(iso).toLocaleString("pt-BR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : "—");

const pm = (m) => `${pct(m.theta)} ± ${pct(m.moe)}`;
const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${(100 * v).toFixed(1)}%`);
const num = (v) => (v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(2));

async function act(dataset, action, confirmText) {
  if (!window.confirm(confirmText)) return;
  busy.add(dataset);
  document.getElementById("erro").textContent = "";
  try {
    await api("POST", `/api/admin/${dataset}/${action}`);
  } catch (e) {
    document.getElementById("erro").textContent = `${dataset}: ${e.message}`;
  } finally {
    busy.delete(dataset);
    await refresh();
  }
}

async function download(dataset) {
  const response = await fetch(`/api/admin/${dataset}/planilha`, {
    headers: { Authorization: `Bearer ${Session.get("token")}` },
  });
  if (!response.ok) {
    document.getElementById("erro").textContent = "Planilha ainda não disponível.";
    return;
  }
  const url = URL.createObjectURL(await response.blob());
  const link = el("a", { href: url, download: `validacao_consolidada_${dataset}.xlsx` });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function progressBlock(progress) {
  return Object.entries(progress).map(([name, p]) =>
    el("div", { class: "evaluator" },
      el("div", { class: "row" }, el("span", {}, name), el("span", {}, `${p.respondidos}/${p.total} · faltam ${p.faltam}`)),
      el("div", { class: "progress" }, el("span", { style: `width:${p.total ? (100 * p.respondidos) / p.total : 0}%` })),
    ));
}

function resultBlock(result) {
  const groups = Object.entries(result.grupos);
  const primary = el("table", {},
    el("thead", {}, el("tr", {}, ...["Grupo", "n", "Humano = referência (± MoE)", "MoE", "κ Fleiss", "Acordo unânime", "Status"].map((h) => el("th", {}, h)))),
    el("tbody", {}, ...groups.map(([g, r]) => el("tr", {},
      el("td", { title: GROUPS[g] }, `${g} — ${GROUPS[g]}`), el("td", { class: "num" }, r.n),
      el("td", { class: "num" }, pm(r.metricas[PRIMARY])), el("td", { class: "num" }, pct(r.metricas[PRIMARY].moe)),
      el("td", { class: "num" }, num(r.kappa_fleiss)), el("td", { class: "num" }, pct(r.acordo_unanime)),
      el("td", { class: r.status === "parar" ? "stop" : "go" }, r.status)))));
  const situations = el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Desfecho"), ...groups.map(([g]) => el("th", { title: GROUPS[g] }, `Grupo ${g}`)))),
    el("tbody", {}, ...Object.entries(SITUATIONS).map(([key, label]) =>
      el("tr", {}, el("td", {}, label), ...groups.map(([, r]) => el("td", { class: "num" }, pm(r.metricas[key])))))));
  return el("div", {},
    el("h3", {}, `Resultado até a rodada ${result.rodada}`),
    primary,
    el("h3", {}, "Desfechos (proporção ± MoE, IC 95%)"),
    situations,
    el("p", { class: result.todos_pararam ? "stop" : "go" },
      result.todos_pararam ? "Critério de parada atingido em todos os grupos." : "Ainda há grupos sem atingir o critério de parada."));
}

function datasetCard(s) {
  const working = busy.has(s.dataset);
  const card = el("section", { class: "card" },
    el("h2", {}, s.dataset, " ",
      s.rodada ? el("span", { class: `badge ${s.estado}` }, `rodada ${s.rodada} · ${s.estado}`) : el("span", { class: "badge" }, "sem rodada")));

  if (s.estado === "aberta" && s.aberta_em) {
    card.append(el("p", { class: "muted" }, `Rodada aberta há ${since(s.aberta_em)} (desde ${when(s.aberta_em)}).`));
  } else if (s.estado === "fechada") {
    card.append(el("p", { class: "muted" }, `Rodada ${s.rodada} durou ${duration(s.duracao_segundos)}.`));
  }
  if (s.progresso) card.append(...progressBlock(s.progresso));

  const actions = el("div", { class: "actions" });
  if (s.estado === "aberta") {
    actions.append(el("button", {
      class: "primary", disabled: working || !s.pode_fechar,
      onclick: () => act(s.dataset, "fechar", `Fechar a rodada ${s.rodada} de ${s.dataset}? Os avaliadores não poderão mais editar.`),
    }, working ? "Processando…" : "Fechar rodada"));
    if (!s.pode_fechar) actions.append(el("span", { class: "muted" }, "Aguardando os três avaliadores terminarem."));
  }
  if (s.pode_iniciar) {
    const label = s.rodada ? "Gerar próxima rodada" : "Iniciar primeira rodada";
    actions.append(el("button", {
      class: "primary", disabled: working,
      onclick: () => act(s.dataset, "rodadas", `${label} de ${s.dataset}?`),
    }, working ? "Processando…" : label));
  }
  if (s.estado === "aberta" && s.fecha_automaticamente_em) {
    const hora = new Date(s.fecha_automaticamente_em).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
    card.append(el("p", { class: "go" }, `Todos terminaram. Fecha automaticamente às ${hora} (janela de revisão; reinicia se alguém editar).`));
  }
  if (s.estado === "aberta" && s.proxima_automatica) {
    card.append(el("p", { class: "muted" }, "Ao fechar, a próxima rodada abre sozinha para os grupos que ainda não pararam."));
  }
  if (s.planilha_disponivel) actions.append(el("button", { onclick: () => download(s.dataset) }, "Baixar planilha consolidada"));
  card.append(actions);

  if (s.resultado) card.append(resultBlock(s.resultado));
  if (s.rodada) {
    card.append(el("div", { class: "actions card-reset" },
      el("button", { class: "danger-outline", type: "button", onclick: () => openReset(datasetReset(s.dataset)) },
        `Reiniciar ${s.dataset}…`)));
  }
  return card;
}

function metricCell(rows, round, group, metric) {
  const agreement = AGREEMENT.has(metric);
  const row = rows.find((r) => r.rodada === round && r.grupo === group && r.metrica === (agreement ? PRIMARY : metric));
  if (!row) return el("td", { class: "muted" }, "—");
  if (agreement) return el("td", { class: "num" }, metric === "kappa_fleiss" ? num(row.kappa_fleiss) : pct(row[metric]));
  return el("td", { class: "num" }, pm(row),
    el("div", { class: "muted small" }, `n=${row.n}${row.status ? ` · ${row.status}` : ""}`));
}

function panelCard(p) {
  const choice = (panelChoice[p.dataset] ||= { metric: PRIMARY });
  const card = el("section", { class: "card" }, el("h2", {}, `Painel — ${p.dataset}`));
  if (!p.rodadas.length) {
    card.append(el("p", { class: "muted" }, "Nenhuma rodada finalizada ainda."));
    return card;
  }

  card.append(
    el("h3", {}, "Rodadas finalizadas"),
    el("table", {},
      el("thead", {}, el("tr", {}, ...["Rodada", "Abertura", "Último avaliador terminou", "Duração", "Documentos"].map((h) => el("th", {}, h)))),
      el("tbody", {}, ...p.rodadas.map((r) => el("tr", {},
        el("td", {}, r.rodada), el("td", {}, when(r.aberta_em)), el("td", {}, when(r.concluida_em)),
        el("td", { class: "num" }, duration(r.duracao_segundos)), el("td", { class: "num" }, r.documentos))))),
  );

  const metric = el("select", { onchange: (e) => { choice.metric = e.target.value; renderPanels(); } },
    ...METRIC_GROUPS.map(([title, items]) => el("optgroup", { label: title },
      ...Object.entries(items).map(([k, label]) => el("option", { value: k }, label)))));
  metric.value = choice.metric;

  const groups = Object.keys(GROUPS);
  card.append(
    el("h3", {}, "Métricas acumuladas até cada rodada"),
    el("div", { class: "panel-controls" }, el("label", {}, "Métrica ", metric)),
    el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "Rodada"), ...groups.map((g) => el("th", { title: GROUPS[g] }, `Grupo ${g}`)))),
      el("tbody", {}, ...p.rodadas.map((r) => el("tr", {},
        el("td", {}, r.rodada), ...groups.map((g) => metricCell(p.metricas, r.rodada, g, choice.metric)))))),
  );
  return card;
}

function renderPanels() {
  document.getElementById("paineis").replaceChildren(...lastPanels.map(panelCard));
}

async function refresh() {
  try {
    const status = await api("GET", "/api/admin/status");
    document.getElementById("datasets").replaceChildren(...status.map(datasetCard));
    lastPanels = await Promise.all(status.map((s) => api("GET", `/api/admin/${s.dataset}/painel`)));
    renderPanels();
  } catch (e) {
    document.getElementById("erro").textContent = e.message;
  }
}

async function setupAutomation() {
  const close = document.getElementById("auto-fechar");
  const next = document.getElementById("auto-proxima");
  const notice = document.getElementById("auto-aviso");

  const show = (a) => {
    close.checked = a.fechamento_automatico;
    next.checked = a.proxima_automatica;
    document.getElementById("auto-fechar-janela").textContent = `(${a.janela_revisao_minutos} min após a última resposta)`;
  };
  const save = async (body) => {
    try {
      show(await api("PUT", "/api/admin/automacao", body));
      notice.textContent = "Preferência salva.";
      setTimeout(() => (notice.textContent = ""), 2500);
      await refresh();
    } catch (e) {
      document.getElementById("erro").textContent = e.message;
    }
  };

  close.addEventListener("change", () => save({ fechamento_automatico: close.checked }));
  next.addEventListener("change", () => save({ proxima_automatica: next.checked }));
  show(await api("GET", "/api/admin/automacao"));
}

// Popup de reinício: o mesmo para "Reiniciar tudo" e para um dataset específico
const RESET_ALL = {
  scope: "Atenção: esta ação apaga o progresso de todos os datasets.",
  word: "REINICIAR",
  endpoint: "/api/admin/reiniciar",
  button: "Reiniciar tudo",
};
let resetTarget = RESET_ALL;

function datasetReset(dataset) {
  return {
    scope: `Atenção: esta ação apaga o progresso de ${dataset}. Os outros datasets não são afetados.`,
    word: dataset,
    endpoint: `/api/admin/${dataset}/reiniciar`,
    button: `Reiniciar ${dataset}`,
  };
}

function openReset(target) {
  resetTarget = target;
  document.getElementById("reinicio-escopo").textContent = target.scope;
  document.getElementById("reinicio-palavra").textContent = target.word;
  document.getElementById("reinicio-confirmar").textContent = target.button;
  document.getElementById("reinicio-confirmacao").value = "";
  document.getElementById("reinicio-erro").textContent = "";
  document.getElementById("reinicio-confirmar").disabled = true;
  document.getElementById("reinicio").showModal();
  document.getElementById("reinicio-confirmacao").focus();
}

function setupReset() {
  const dialog = document.getElementById("reinicio");
  const input = document.getElementById("reinicio-confirmacao");
  const confirmButton = document.getElementById("reinicio-confirmar");
  const error = document.getElementById("reinicio-erro");

  document.getElementById("abrir-reinicio").addEventListener("click", () => openReset(RESET_ALL));
  document.getElementById("reinicio-cancelar").addEventListener("click", () => dialog.close());
  input.addEventListener("input", () => (confirmButton.disabled = input.value !== resetTarget.word));
  confirmButton.addEventListener("click", async () => {
    confirmButton.disabled = true;
    try {
      const result = await api("POST", resetTarget.endpoint, { confirmacao: input.value });
      dialog.close();
      window.alert(`Validação reiniciada. Cópia de segurança em:\n${result.backups.join("\n")}`);
      await refresh();
    } catch (e) {
      error.textContent = e.message;
      confirmButton.disabled = false;
    }
  });
}

setupReset();
setupAutomation();
watchVersion(() => window.location.reload());
refresh();
setInterval(() => { if (!busy.size) refresh(); }, 20000);
