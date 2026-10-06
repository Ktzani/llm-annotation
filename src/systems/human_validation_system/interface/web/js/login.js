// Tela de login: avaliadores vão para /avaliar, administrador para /admin.
const ROUTES = { avaliador: "/avaliar", admin: "/admin" };

async function initLogin() {
  const select = document.getElementById("usuario");
  const users = await api("GET", "/api/usuarios");
  select.append(el("option", { value: "" }, "Selecione…"));
  users.forEach((u) => select.append(el("option", { value: u }, u)));
  select.append(el("option", { value: "admin" }, "Administrador"));

  document.getElementById("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const erro = document.getElementById("erro");
    erro.textContent = "";
    try {
      const session = await api("POST", "/api/login", {
        usuario: select.value,
        codigo: document.getElementById("codigo").value,
      });
      Session.save(session);
      window.location.href = ROUTES[session.papel];
    } catch (e) {
      erro.textContent = e.message;
    }
  });
}

initLogin();
