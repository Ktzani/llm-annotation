// Tela do avaliador: um documento por vez, duas etapas, revisão das próprias respostas.
Session.require("avaliador");

const state = { dataset: null, options: null, doc: null, editing: false };
const $ = (id) => document.getElementById(id);

function radio(name, value, label, extraClass = "") {
  return el("label", { class: `choice ${extraClass}` },
    el("input", { type: "radio", name, value, required: true }), label);
}

function checked(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || null;
}

function renderOptions() {
  const { classes, opcao_indecidivel, sim_nao } = state.options;
  $("rotulos").replaceChildren(...classes.map((c) => radio("rotulo", c.rotulo, c.rotulo)));
  // Fora da grade de classes: é uma resposta sobre o texto, não uma classe
  const undecidableLabel = opcao_indecidivel.charAt(0).toUpperCase() + opcao_indecidivel.slice(1);
  $("indecidivel").replaceChildren(radio("rotulo", opcao_indecidivel, undecidableLabel, "undecidable"));
  $("sim-nao").replaceChildren(...sim_nao.map((v) => radio("outro", v, v)));

  $("guia").replaceChildren(
    ...classes.map((c) =>
      el("details", {},
        el("summary", {}, c.rotulo),
        c.descricao ? el("p", {}, c.descricao) : null,
        ...c.exemplos.map((t) => el("p", { class: "example" }, t)),
      )),
    el("details", {},
      el("summary", {}, undecidableLabel),
      el("p", {}, "Use só quando o texto não traz informação suficiente para escolher qualquer rótulo: "
        + "está truncado, é genérico demais ou trata de um assunto fora de todos os rótulos."),
      el("p", {}, "Não use quando estiver em dúvida entre dois rótulos: escolha o mais adequado na etapa 1 "
        + "e indique o outro na etapa 2."),
    ),
  );
}

function refreshOtherOptions(selected) {
  const chosen = checked("rotulo");
  const select = $("qual-outro");
  select.replaceChildren(el("option", { value: "" }, "Selecione…"),
    ...state.options.classes.filter((c) => c.rotulo !== chosen).map((c) => el("option", { value: c.rotulo }, c.rotulo)));
  if (selected && selected !== chosen) select.value = selected;
}

function syncStep2() {
  $("etapa2").disabled = !checked("rotulo");
  $("bloco-outro").classList.toggle("hidden", checked("outro") !== "sim");
  $("qual-outro").required = checked("outro") === "sim";
}

function fillForm(answer) {
  $("form-resposta").reset();
  if (answer) {
    document.querySelector(`input[name="rotulo"][value="${CSS.escape(answer.rotulo_escolhido)}"]`).checked = true;
    document.querySelector(`input[name="outro"][value="${CSS.escape(answer.outro_rotulo_possivel)}"]`).checked = true;
  }
  refreshOtherOptions(answer?.qual_outro_rotulo);
  syncStep2();
}

function hhmm(iso) {
  return new Date(iso).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
}

function showProgress(respondidos, total) {
  $("contagem").textContent = `${respondidos} de ${total} respondidos`;
  $("barra").style.width = total ? `${(100 * respondidos) / total}%` : "0%";
}

function showDocument(doc, editing) {
  state.doc = doc;
  state.editing = editing;
  $("concluido").classList.add("hidden");
  $("documento").classList.remove("hidden");
  $("posicao").textContent = `Documento ${doc.posicao} de ${doc.total}`;
  $("modo").textContent = editing ? "Revisando uma resposta já enviada." : "";
  $("texto").textContent = doc.texto;
  $("salvar").textContent = editing ? "Salvar alteração" : "Salvar e próximo";
  $("voltar").classList.toggle("hidden", !editing);
  $("erro").textContent = "";
  fillForm(doc.resposta);
  $("texto").scrollTop = 0;
}

async function loadNext() {
  const next = await api("GET", `/api/avaliacao/${state.dataset}/proximo`);
  showProgress(next.respondidos, next.total);
  if (next.concluido) {
    state.doc = null;
    $("revisao").textContent = next.revisao_ate
      ? `Todos os avaliadores terminaram. Você pode revisar suas respostas em "Minhas respostas" até ${hhmm(next.revisao_ate)}; depois a rodada fecha automaticamente.`
      : `Obrigado! Você ainda pode revisar suas respostas em "Minhas respostas" até a rodada fechar.`;
    $("documento").classList.add("hidden");
    $("concluido").classList.remove("hidden");
    $("posicao").textContent = "Rodada concluída";
  } else {
    showDocument(next.documento, false);
  }
}

async function loadAnswers() {
  const answers = await api("GET", `/api/avaliacao/${state.dataset}/respostas`);
  $("respostas").replaceChildren(...answers.map((a) =>
    el("li", {},
      el("span", {}, `#${a.posicao} · ${a.rotulo_escolhido}`),
      el("button", { class: "link", type: "button", onclick: () => openAnswer(a.id_anonimo) }, "revisar"),
    )));
  if (!answers.length) $("respostas").append(el("li", { class: "muted" }, "Nenhuma resposta ainda."));
}

async function openAnswer(idAnonimo) {
  const doc = await api("GET", `/api/avaliacao/${state.dataset}/documentos/${encodeURIComponent(idAnonimo)}`);
  showDocument(doc, true);
}

async function save(event) {
  event.preventDefault();
  $("erro").textContent = "";
  const body = {
    rotulo_escolhido: checked("rotulo"),
    outro_rotulo_possivel: checked("outro"),
    qual_outro_rotulo: checked("outro") === "sim" ? $("qual-outro").value : null,
  };
  try {
    await api("PUT", `/api/avaliacao/${state.dataset}/respostas/${encodeURIComponent(state.doc.id_anonimo)}`, body);
    $("aviso").textContent = "Resposta salva.";
    setTimeout(() => ($("aviso").textContent = ""), 2000);
    await loadAnswers();
    await loadNext();
  } catch (e) {
    $("erro").textContent = e.message;
  }
}

async function selectDataset(dataset, datasets) {
  state.dataset = dataset;
  document.querySelectorAll("#datasets button").forEach((b) => b.setAttribute("aria-selected", b.dataset.name === dataset));
  const info = datasets.find((d) => d.dataset === dataset);
  if (!info.rodada_aberta) {
    $("area").classList.add("hidden");
    $("sem-rodada").textContent = "Não há rodada aberta para este dataset no momento.";
    return;
  }
  $("sem-rodada").textContent = "";
  $("area").classList.remove("hidden");
  state.options = await api("GET", `/api/avaliacao/${dataset}/opcoes`);
  renderOptions();
  await loadAnswers();
  await loadNext();
}

// Introdução: aparece na primeira visita do avaliador e pode ser reaberta em "Instruções"
const introKey = () => `hv_intro_vista_${Session.get("usuario")}`;

function showIntro(visible) {
  $("intro").classList.toggle("hidden", !visible);
  $("trabalho").classList.toggle("hidden", visible);
  window.scrollTo(0, 0);
}

function introSeen() {
  try { return localStorage.getItem(introKey()) === "1"; } catch { return false; }
}

function markIntroSeen() {
  try { localStorage.setItem(introKey(), "1"); } catch { /* sem armazenamento: mostra de novo na próxima visita */ }
}

async function init() {
  $("quem").textContent = Session.get("usuario");
  $("comecar").addEventListener("click", () => { markIntroSeen(); showIntro(false); });
  $("abrir-intro").addEventListener("click", () => showIntro(true));
  showIntro(!introSeen());
  $("form-resposta").addEventListener("submit", save);
  $("form-resposta").addEventListener("change", (e) => {
    if (e.target.name === "rotulo") refreshOtherOptions($("qual-outro").value);
    syncStep2();
  });
  $("voltar").addEventListener("click", loadNext);

  const datasets = await api("GET", "/api/avaliacao/datasets");
  $("datasets").replaceChildren(...datasets.map((d) =>
    el("button", { type: "button", "data-name": d.dataset, onclick: () => selectDataset(d.dataset, datasets) },
      d.dataset, d.rodada_aberta ? ` (${d.respondidos}/${d.total})` : " (sem rodada aberta)")));
  const first = datasets.find((d) => d.rodada_aberta) || datasets[0];
  if (first) await selectDataset(first.dataset, datasets);
}

function showUpdateBanner() {
  if ($("atualizacao")) return;
  document.body.prepend(el("div", { class: "update-banner", id: "atualizacao", role: "status" },
    "A página foi atualizada. Salve a resposta atual e clique em ",
    el("button", { type: "button", class: "primary", onclick: () => window.location.reload() }, "Recarregar"),
    "."));
}

watchVersion(showUpdateBanner);
init().catch((e) => ($("sem-rodada").textContent = e.message));
