"""
Duração das rodadas, painel por dataset (nunca agregado), avisos ao avaliador e email ao fechar.
"""
import shutil

import pytest
from fastapi.testclient import TestClient

from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.interface.api.server import create_app
from src.systems.human_validation_system.interface.api.services.email_notifier import EmailNotifier
from tests.human_validation_system.conftest import ADMIN_CODE, CODES, DATASET, DATE, answer_all, login

OTHER = "dblp"


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, subject, body):
        self.sent.append((subject, body))


@pytest.fixture
def notifier(client):
    fake = FakeNotifier()
    client.app.state.controller.notifier = fake
    return fake


def state_of(client, admin, dataset):
    return next(s for s in client.get("/api/admin/status", headers=admin).json() if s["dataset"] == dataset)


def run_round(client, admin, evaluators, dataset=DATASET):
    if state_of(client, admin, dataset)["estado"] != "aberta":
        client.post(f"/api/admin/{dataset}/rodadas", headers=admin)
    for headers in evaluators.values():
        answer_all_for(client, headers, dataset)
    assert client.post(f"/api/admin/{dataset}/fechar", headers=admin).status_code == 200


def answer_all_for(client, headers, dataset):
    while True:
        nxt = client.get(f"/api/avaliacao/{dataset}/proximo", headers=headers).json()
        if nxt["concluido"]:
            return
        doc = nxt["documento"]["id_anonimo"]
        body = {"rotulo_escolhido": "children", "outro_rotulo_possivel": "não"}
        assert client.put(f"/api/avaliacao/{dataset}/respostas/{doc}", json=body, headers=headers).status_code == 200


# ------------------------------------------------------------------ duração
def test_round_duration_is_saved_when_closing(client, admin, evaluators, notifier):
    run_round(client, admin, evaluators)
    panel = client.get(f"/api/admin/{DATASET}/painel", headers=admin).json()
    closed = panel["rodadas"][0]
    assert closed["rodada"] == 1 and closed["documentos"] == 90
    assert closed["concluida_em"] is not None and closed["fechada_em"] is not None
    assert closed["duracao_segundos"] >= 0


def test_open_round_exposes_start_time(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    status = client.get("/api/admin/status", headers=admin).json()[0]
    info = client.get("/api/avaliacao/datasets", headers=evaluators["avaliador_1"]).json()[0]
    assert status["aberta_em"] is not None
    assert info["rodada_aberta_em"] == status["aberta_em"]
    assert info["janela_revisao_minutos"] == 5


# ------------------------------------------------------------------ avisos ao avaliador
def test_evaluator_knows_whether_others_finished(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    answer_all(client, evaluators["avaliador_1"])
    early = client.get(f"/api/avaliacao/{DATASET}/proximo", headers=evaluators["avaliador_1"]).json()
    assert early["concluido"] and not early["todos_terminaram"] and early["revisao_ate"] is None

    answer_all(client, evaluators["avaliador_2"])
    answer_all(client, evaluators["avaliador_3"])
    last = client.get(f"/api/avaliacao/{DATASET}/proximo", headers=evaluators["avaliador_3"]).json()
    assert last["todos_terminaram"] and last["revisao_ate"] is not None


# ------------------------------------------------------------------ painel
def test_dashboard_metrics_are_always_cumulative(client, admin, evaluators, notifier):
    """Como no procedimento sequencial: cada rodada mostra o acumulado até ela, nunca a rodada sozinha."""
    run_round(client, admin, evaluators)
    run_round(client, admin, evaluators)
    metrics = client.get(f"/api/admin/{DATASET}/painel", headers=admin).json()["metricas"]

    def n(round_number, group="A"):
        return next(m["n"] for m in metrics if m["rodada"] == round_number
                    and m["grupo"] == group and m["metrica"] == "acerto_referencia")

    assert "escopo" not in metrics[0]
    assert n(1) == 30 and n(2) == 50  # 30 da rodada 1 + 20 da rodada 2
    path = client.app.state.controller.validation_dir(DATASET) / "estimativas" / "metricas_por_rodada.csv"
    assert path.exists()


def test_dashboard_is_per_dataset_and_rounds_are_independent(results_dir, tmp_path):
    shutil.copytree(results_dir / DATASET, results_dir / OTHER)
    settings = InterfaceSettings({DATASET: DATE, OTHER: DATE}, str(results_dir), CODES, ADMIN_CODE)
    client = TestClient(create_app(settings))
    client.app.state.controller.notifier = FakeNotifier()
    admin = login(client, "admin", ADMIN_CODE)
    evaluators = {u: login(client, u, c) for u, c in CODES.items()}

    client.post(f"/api/admin/{OTHER}/rodadas", headers=admin)
    run_round(client, admin, evaluators, DATASET)

    status = {s["dataset"]: s for s in client.get("/api/admin/status", headers=admin).json()}
    assert (status[DATASET]["rodada"], status[DATASET]["estado"]) == (2, "aberta")  # books já está na rodada 2
    assert (status[OTHER]["rodada"], status[OTHER]["estado"]) == (1, "aberta")      # dblp segue na rodada 1
    assert client.get(f"/api/admin/{OTHER}/painel", headers=admin).json() == {"dataset": OTHER, "rodadas": [], "metricas": []}
    assert client.get(f"/api/admin/{DATASET}/painel", headers=admin).json()["rodadas"][0]["rodada"] == 1


def test_reset_also_clears_dashboard(client, admin, evaluators, notifier):
    run_round(client, admin, evaluators)
    client.post("/api/admin/reiniciar", json={"confirmacao": "REINICIAR"}, headers=admin)
    assert client.get(f"/api/admin/{DATASET}/painel", headers=admin).json()["metricas"] == []


# ------------------------------------------------------------------ email
def test_email_sent_when_round_closes_with_next_round_info(client, admin, evaluators, notifier):
    run_round(client, admin, evaluators)
    assert len(notifier.sent) == 1
    subject, body = notifier.sent[0]
    assert DATASET in subject and "rodada 1" in subject
    assert "rodada 2 já foi aberta" in body and "Duração" in body


def test_email_says_when_next_round_was_not_opened(client, admin, evaluators, notifier):
    client.put("/api/admin/automacao", json={"proxima_automatica": False}, headers=admin)
    run_round(client, admin, evaluators)
    assert "não foi aberta automaticamente" in notifier.sent[0][1]


def test_email_notifier_is_disabled_without_credentials():
    notifier = EmailNotifier("alguem@exemplo.com", "smtp.gmail.com", 587, None, None)
    assert not notifier.enabled
    notifier.send("assunto", "corpo")  # não envia nem levanta erro


def test_smtp_settings_from_env(monkeypatch):
    monkeypatch.setenv("HV_NOTIFY_EMAIL", "admin@exemplo.com")
    monkeypatch.setenv("HV_SMTP_USER", "remetente@gmail.com")
    monkeypatch.setenv("HV_SMTP_PASSWORD", "abcd efgh ijkl mnop")
    for user, code in CODES.items():
        monkeypatch.setenv(InterfaceSettings.code_variable(user), code)
    monkeypatch.setenv("HV_ADMIN_CODE", ADMIN_CODE)
    settings = InterfaceSettings.from_env({DATASET: DATE}, "data/results")
    assert settings.smtp_password == "abcdefghijklmnop"
    assert settings.notify_email == "admin@exemplo.com"


def test_experiments_come_from_one_variable_per_dataset(monkeypatch):
    monkeypatch.setenv("HV_EXPERIMENT_BOOKS", "2026-09-13_07-19-54")
    monkeypatch.delenv("HV_EXPERIMENT_DBLP", raising=False)
    monkeypatch.setenv("HV_EXPERIMENT_AGNEWS", "ignorado")  # fora de VALIDATION_DATASETS
    assert InterfaceSettings.experiments_from_env() == {"books": "2026-09-13_07-19-54"}

    monkeypatch.setenv("HV_EXPERIMENT_DBLP", "2026-09-13_18-46-30")
    assert InterfaceSettings.experiments_from_env() == {"books": "2026-09-13_07-19-54", "dblp": "2026-09-13_18-46-30"}
