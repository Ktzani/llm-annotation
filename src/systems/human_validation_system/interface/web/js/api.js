// Cliente da API: sessão no navegador + requisições autenticadas.
const Session = {
  get(key) {
    try { return localStorage.getItem(`hv_${key}`); } catch { return null; }
  },
  save({ token, usuario, papel }) {
    try {
      localStorage.setItem("hv_token", token);
      localStorage.setItem("hv_usuario", usuario);
      localStorage.setItem("hv_papel", papel);
    } catch { /* navegação privada: sessão vale só nesta página */ }
  },
  clear() {
    try { ["token", "usuario", "papel"].forEach((k) => localStorage.removeItem(`hv_${k}`)); } catch {}
  },
  require(role) {
    if (!this.get("token") || this.get("papel") !== role) window.location.href = "/";
  },
};

async function api(method, path, body) {
  const headers = { Authorization: `Bearer ${Session.get("token") || ""}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (response.status === 401) {
    Session.clear();
    window.location.href = "/";
    throw new Error("Sessão expirada");
  }
  const data = response.headers.get("content-type")?.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const detail = data?.detail;
    throw new Error(typeof detail === "string" ? detail : "Erro ao comunicar com o servidor");
  }
  return data;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== false && value !== null && value !== undefined) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function logout() {
  Session.clear();
  window.location.href = "/";
}
