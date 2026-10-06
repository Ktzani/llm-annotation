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
const busy = new Set();

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
    el("thead", {}, el("tr", {}, ...["Grupo", "n", "Humano = referência", "IC 95%", "MoE", "κ Fleiss", "Unânime", "Status"].map((h) => el("th", {}, h)))),
    el("tbody", {}, ...groups.map(([g, r]) => {
      const m = r.metricas[PRIMARY];
      return el("tr", {},
        el("td", { title: GROUPS[g] }, g), el("td", { class: "num" }, r.n), el("td", { class: "num" }, pct(m.theta)),
        el("td", { class: "num" }, `${pct(m.ic_inferior)} – ${pct(m.ic_superior)}`), el("td", { class: "num" }, pct(m.moe)),
        el("td", { class: "num" }, num(r.kappa_fleiss)), el("td", { class: "num" }, pct(r.acordo_unanime)),
        el("td", { class: r.status === "parar" ? "stop" : "go" }, r.status));
    })));
  const situations = el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Desfecho"), ...groups.map(([g]) => el("th", {}, g)))),
    el("tbody", {}, ...Object.entries(SITUATIONS).map(([key, label]) =>
      el("tr", {}, el("td", {}, label), ...groups.map(([, r]) => {
        const m = r.metricas[key];
        return el("td", { class: "num" }, `${pct(m.theta)} [${pct(m.ic_inferior)}–${pct(m.ic_superior)}]`);
      })))));
  return el("div", {},
    el("h3", {}, `Resultado até a rodada ${result.rodada}`),
    primary,
    el("h3", { style: "margin-top:1rem" }, "Desfechos (proporção e IC 95%)"),
    situations,
    el("p", { class: result.todos_pararam ? "stop" : "go" },
      result.todos_pararam ? "Critério de parada atingido em todos os grupos." : "Ainda há grupos sem atingir o critério de parada."));
}

function datasetCard(s) {
  const working = busy.has(s.dataset);
  const card = el("section", { class: "card" },
    el("h2", {}, s.dataset, " ",
      s.rodada ? el("span", { class: `badge ${s.estado}` }, `rodada ${s.rodada} · ${s.estado}`) : el("span", { class: "badge" }, "sem rodada")));

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
  return card;
}

async function refresh() {
  try {
    const status = await api("GET", "/api/admin/status");
    document.getElementById("datasets").replaceChildren(...status.map(datasetCard));
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

function setupReset() {
  const dialog = document.getElementById("reinicio");
  const input = document.getElementById("reinicio-confirmacao");
  const confirmButton = document.getElementById("reinicio-confirmar");
  const error = document.getElementById("reinicio-erro");

  document.getElementById("abrir-reinicio").addEventListener("click", () => {
    input.value = "";
    error.textContent = "";
    confirmButton.disabled = true;
    dialog.showModal();
    input.focus();
  });
  document.getElementById("reinicio-cancelar").addEventListener("click", () => dialog.close());
  input.addEventListener("input", () => (confirmButton.disabled = input.value !== "REINICIAR"));
  confirmButton.addEventListener("click", async () => {
    confirmButton.disabled = true;
    try {
      const result = await api("POST", "/api/admin/reiniciar", { confirmacao: input.value });
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
