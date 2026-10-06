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
  $("rotulos").replaceChildren(
    ...classes.map((c) => radio("rotulo", c.rotulo, c.rotulo)),
    radio("rotulo", opcao_indecidivel, opcao_indecidivel, "undecidable"),
  );
  $("sim-nao").replaceChildren(...sim_nao.map((v) => radio("outro", v, v)));

  $("guia").replaceChildren(...classes.map((c) =>
    el("details", {},
      el("summary", {}, c.rotulo),
      c.descricao ? el("p", {}, c.descricao) : null,
      ...c.exemplos.map((t) => el("p", { class: "example" }, t)),
    )));
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
    $("observacao").value = answer.observacao || "";
  }
  refreshOtherOptions(answer?.qual_outro_rotulo);
  syncStep2();
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
    observacao: $("observacao").value,
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

async function init() {
  $("quem").textContent = Session.get("usuario");
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

init().catch((e) => ($("sem-rodada").textContent = e.message));
