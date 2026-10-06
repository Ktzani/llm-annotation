"""
Ciclo da rodada: não fecha com avaliador incompleto, ordem igual para os três, travas.
"""
import pytest

from src.systems.human_validation_system.interface.api.services.round_controller import RoundIncompleteError
from tests.human_validation_system.conftest import DATASET, answer_all


def test_round_does_not_close_with_incomplete_evaluator(client, admin, evaluators):
    assert client.post(f"/api/admin/{DATASET}/rodadas", headers=admin).status_code == 200
    answer_all(client, evaluators["avaliador_1"])
    answer_all(client, evaluators["avaliador_2"])
    answer_all(client, evaluators["avaliador_3"], leave=1)

    status = client.get("/api/admin/status", headers=admin).json()[0]
    assert status["pode_fechar"] is False
    assert status["progresso"]["avaliador_3"]["faltam"] == 1

    response = client.post(f"/api/admin/{DATASET}/fechar", headers=admin)
    assert response.status_code == 409
    assert "avaliador_3" in response.json()["detail"]
    assert client.get("/api/admin/status", headers=admin).json()[0]["estado"] == "aberta"


def test_controller_refuses_to_close_incomplete_round(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    answer_all(client, evaluators["avaliador_1"])
    with pytest.raises(RoundIncompleteError):
        client.app.state.controller.close_round(DATASET)


def test_round_closes_when_all_complete_and_locks_answers(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    for headers in evaluators.values():
        answer_all(client, headers)

    response = client.post(f"/api/admin/{DATASET}/fechar", headers=admin)
    assert response.status_code == 200
    assert response.json()["resultado"]["rodada"] == 1
    assert set(response.json()["resultado"]["grupos"]) == {"A", "B", "C"}

    # Documento da rodada 1 não pode mais ser editado (não pertence à rodada aberta)
    key = client.app.state.controller.store_key(DATASET)
    any_id = client.app.state.store.all_responses(key, 1)["id_anonimo"].iloc[0]
    edit = client.put(
        f"/api/avaliacao/{DATASET}/respostas/{any_id}",
        json={"rotulo_escolhido": "romance", "outro_rotulo_possivel": "não"},
        headers=evaluators["avaliador_1"],
    )
    assert edit.status_code in (404, 409)


def test_next_round_opens_automatically_when_groups_pending(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    for headers in evaluators.values():
        answer_all(client, headers)

    status = client.post(f"/api/admin/{DATASET}/fechar", headers=admin).json()
    assert not status["resultado"]["todos_pararam"]
    assert (status["rodada"], status["estado"]) == (2, "aberta")
    assert status["resultado"]["rodada"] == 1
    nxt = client.get(f"/api/avaliacao/{DATASET}/proximo", headers=evaluators["avaliador_1"]).json()
    assert not nxt["concluido"] and nxt["respondidos"] == 0


def test_no_automatic_round_when_disabled(client, admin, evaluators):
    client.app.state.settings.auto_next_round = False
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    for headers in evaluators.values():
        answer_all(client, headers)

    status = client.post(f"/api/admin/{DATASET}/fechar", headers=admin).json()
    assert (status["rodada"], status["estado"]) == (1, "fechada")
    assert status["pode_iniciar"] is True


def test_same_order_for_all_evaluators(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    orders = [answer_all(client, headers) for headers in evaluators.values()]
    assert orders[0] == orders[1] == orders[2]
    assert len(orders[0]) == 90


def test_evaluator_can_revise_own_answer_while_open(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    headers = evaluators["avaliador_1"]
    first = answer_all(client, headers)[0]
    body = {"rotulo_escolhido": "poetry", "outro_rotulo_possivel": "sim", "qual_outro_rotulo": "romance"}
    assert client.put(f"/api/avaliacao/{DATASET}/respostas/{first}", json=body, headers=headers).status_code == 200

    doc = client.get(f"/api/avaliacao/{DATASET}/documentos/{first}", headers=headers).json()
    assert doc["resposta"]["rotulo_escolhido"] == "poetry"
    assert doc["resposta"]["qual_outro_rotulo"] == "romance"


@pytest.mark.parametrize("body", [
    {"rotulo_escolhido": "inexistente", "outro_rotulo_possivel": "não"},
    {"rotulo_escolhido": "poetry", "outro_rotulo_possivel": "talvez"},
    {"rotulo_escolhido": "poetry", "outro_rotulo_possivel": "sim"},
    {"rotulo_escolhido": "poetry", "outro_rotulo_possivel": "sim", "qual_outro_rotulo": "poetry"},
])
def test_invalid_answers_are_rejected(client, admin, evaluators, body):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    headers = evaluators["avaliador_1"]
    doc = client.get(f"/api/avaliacao/{DATASET}/proximo", headers=headers).json()["documento"]
    assert client.put(f"/api/avaliacao/{DATASET}/respostas/{doc['id_anonimo']}", json=body, headers=headers).status_code == 422


def test_other_experiment_date_does_not_share_rounds(client, admin, results_dir):
    """Mesmo banco, outra data: as rodadas do primeiro experimento (ex.: teste) não aparecem."""
    import shutil

    from fastapi.testclient import TestClient

    from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
    from src.systems.human_validation_system.interface.api.server import create_app
    from tests.human_validation_system.conftest import ADMIN_CODE, CODES, DATE, login

    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    shutil.copytree(results_dir / DATASET / DATE / "consensus", results_dir / DATASET / "outra_data" / "consensus")

    settings = client.app.state.settings
    other = TestClient(create_app(InterfaceSettings({DATASET: "outra_data"}, str(results_dir), CODES, ADMIN_CODE, settings.db_path)))
    status = other.get("/api/admin/status", headers=login(other, "admin", ADMIN_CODE)).json()[0]
    assert status["estado"] == "sem_rodada"


def test_outputs_stay_out_of_experiment_folder(client, admin, evaluators, results_dir):
    """Tudo da validação humana fica em data/validacao_humana (irmã de results); a pasta do experimento não muda."""
    from tests.human_validation_system.conftest import DATE

    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    for headers in evaluators.values():
        answer_all(client, headers)
    client.post(f"/api/admin/{DATASET}/fechar", headers=admin)

    validation = results_dir.parent / "validacao_humana"
    assert not (results_dir / "validacao_humana").exists()
    assert client.app.state.controller.validation_dir(DATASET) == validation / DATASET / DATE
    assert client.app.state.settings.db_path == validation / "validacao_humana.db"
    assert (validation / DATASET / DATE / f"validacao_consolidada_{DATASET}.xlsx").exists()
    assert sorted(p.name for p in (results_dir / DATASET / DATE).iterdir()) == ["consensus"]
