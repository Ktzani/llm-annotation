"""
Janela de revisão: com os três completos, a rodada fecha sozinha após o prazo (e abre a próxima).
"""
import time
from datetime import timedelta

from tests.human_validation_system.conftest import DATASET, answer_all


def open_and_answer(client, admin, evaluators, leave_last: int = 0):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    users = list(evaluators)
    for user in users:
        answer_all(client, evaluators[user], leave=leave_last if user == users[-1] else 0)


def test_incomplete_round_never_closes_automatically(client, admin, evaluators):
    open_and_answer(client, admin, evaluators, leave_last=1)
    controller = client.app.state.controller
    assert controller.review_deadline(DATASET) is None
    assert controller.close_due_rounds() == []
    assert client.get("/api/admin/status", headers=admin).json()[0]["estado"] == "aberta"


def test_round_closes_only_after_review_window(client, admin, evaluators):
    open_and_answer(client, admin, evaluators)
    controller = client.app.state.controller
    deadline = controller.review_deadline(DATASET)
    assert deadline is not None

    assert controller.close_due_rounds(now=deadline - timedelta(seconds=1)) == []
    assert client.get("/api/admin/status", headers=admin).json()[0]["rodada"] == 1

    assert controller.close_due_rounds(now=deadline + timedelta(seconds=1)) == [DATASET]
    status = client.get("/api/admin/status", headers=admin).json()[0]
    assert status["resultado"]["rodada"] == 1
    assert (status["rodada"], status["estado"]) == (2, "aberta")  # próxima rodada aberta automaticamente


def test_edit_during_window_postpones_closing(client, admin, evaluators):
    open_and_answer(client, admin, evaluators)
    controller = client.app.state.controller
    before = controller.review_deadline(DATASET)

    time.sleep(1.1)
    headers = evaluators["avaliador_1"]
    first = client.get(f"/api/avaliacao/{DATASET}/respostas", headers=headers).json()[0]["id_anonimo"]
    body = {"rotulo_escolhido": "poetry", "outro_rotulo_possivel": "não"}
    assert client.put(f"/api/avaliacao/{DATASET}/respostas/{first}", json=body, headers=headers).status_code == 200

    after = controller.review_deadline(DATASET)
    assert after > before
    assert controller.close_due_rounds(now=before + timedelta(seconds=0.5)) == []


def test_deadline_is_shown_to_admin_and_evaluators(client, admin, evaluators):
    open_and_answer(client, admin, evaluators)
    status = client.get("/api/admin/status", headers=admin).json()[0]
    nxt = client.get(f"/api/avaliacao/{DATASET}/proximo", headers=evaluators["avaliador_2"]).json()
    assert status["fecha_automaticamente_em"] is not None
    assert nxt["concluido"] and nxt["revisao_ate"] == status["fecha_automaticamente_em"]


def test_zero_window_disables_automatic_closing(client, admin, evaluators):
    client.app.state.settings.review_window_minutes = 0
    open_and_answer(client, admin, evaluators)
    controller = client.app.state.controller
    assert controller.review_deadline(DATASET) is None
    assert controller.close_due_rounds() == []
