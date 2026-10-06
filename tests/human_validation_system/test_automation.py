"""
Automação ligada/desligada pelo administrador: próxima rodada e fechamento automático.
"""
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.interface.api.server import create_app
from tests.human_validation_system.conftest import ADMIN_CODE, CODES, DATASET, DATE, answer_all, login


def complete_round(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    for headers in evaluators.values():
        answer_all(client, headers)


def test_defaults_come_from_config(client, admin):
    automation = client.get("/api/admin/automacao", headers=admin).json()
    assert automation["proxima_automatica"] is True
    assert automation["fechamento_automatico"] is True


def test_disabling_next_round_keeps_round_closed(client, admin, evaluators):
    client.put("/api/admin/automacao", json={"proxima_automatica": False}, headers=admin)
    complete_round(client, admin, evaluators)

    status = client.post(f"/api/admin/{DATASET}/fechar", headers=admin).json()
    assert (status["rodada"], status["estado"]) == (1, "fechada")
    assert status["pode_iniciar"] is True
    assert status["proxima_automatica"] is False


def test_disabling_auto_close_never_closes_by_itself(client, admin, evaluators):
    client.put("/api/admin/automacao", json={"fechamento_automatico": False}, headers=admin)
    complete_round(client, admin, evaluators)

    controller = client.app.state.controller
    assert controller.review_deadline(DATASET) is None
    assert controller.close_due_rounds(now=datetime.now() + timedelta(days=1)) == []
    assert client.get("/api/admin/status", headers=admin).json()[0]["fecha_automaticamente_em"] is None
    # o botão manual continua funcionando
    assert client.post(f"/api/admin/{DATASET}/fechar", headers=admin).status_code == 200


def test_choice_survives_server_restart(client, admin):
    client.put("/api/admin/automacao", json={"proxima_automatica": False, "fechamento_automatico": False}, headers=admin)
    settings = client.app.state.settings
    restarted = TestClient(create_app(InterfaceSettings({DATASET: DATE}, settings.results_dir, CODES, ADMIN_CODE, settings.db_path)))
    automation = restarted.get("/api/admin/automacao", headers=login(restarted, "admin", ADMIN_CODE)).json()
    assert automation["proxima_automatica"] is False
    assert automation["fechamento_automatico"] is False


def test_only_admin_changes_automation(client, evaluators):
    headers = evaluators["avaliador_1"]
    assert client.get("/api/admin/automacao", headers=headers).status_code == 403
    assert client.put("/api/admin/automacao", json={"proxima_automatica": False}, headers=headers).status_code == 403
