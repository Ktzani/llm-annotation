"""
Três avaliadores por dataset: cada um vê só o seu dataset e a rodada fecha com os três dele.
"""
import shutil

import pytest
from fastapi.testclient import TestClient

from src.config.human_validation import EVALUATOR_NAMES, EVALUATORS, EVALUATORS_BY_DATASET
from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.interface.api.server import create_app
from tests.human_validation_system.conftest import ADMIN_CODE, DATASET, DATE, login

OTHER = "dblp"
TEAMS = {DATASET: ["livro_1", "livro_2", "livro_3"], OTHER: ["artigo_1", "artigo_2", "artigo_3"]}
CODES = {name: f"code-{name}" for names in TEAMS.values() for name in names}


@pytest.fixture
def two_teams(results_dir):
    shutil.copytree(results_dir / DATASET, results_dir / OTHER)
    settings = InterfaceSettings({DATASET: DATE, OTHER: DATE}, str(results_dir), CODES, ADMIN_CODE,
                                 evaluators_by_dataset=TEAMS)
    client = TestClient(create_app(settings))
    admin = login(client, "admin", ADMIN_CODE)
    for dataset in TEAMS:
        client.post(f"/api/admin/{dataset}/rodadas", headers=admin)
    return client, admin, {name: login(client, name, code) for name, code in CODES.items()}


def answer_all(client, dataset, headers):
    while True:
        nxt = client.get(f"/api/avaliacao/{dataset}/proximo", headers=headers).json()
        if nxt["concluido"]:
            return
        doc = nxt["documento"]["id_anonimo"]
        body = {"rotulo_escolhido": "children", "outro_rotulo_possivel": "não"}
        assert client.put(f"/api/avaliacao/{dataset}/respostas/{doc}", json=body, headers=headers).status_code == 200


def test_evaluator_sees_only_own_dataset(two_teams):
    client, _, evaluators = two_teams
    books_reader = evaluators["livro_1"]
    assert [d["dataset"] for d in client.get("/api/avaliacao/datasets", headers=books_reader).json()] == [DATASET]
    for path in (f"/api/avaliacao/{OTHER}/proximo", f"/api/avaliacao/{OTHER}/opcoes", f"/api/avaliacao/{OTHER}/respostas"):
        assert client.get(path, headers=books_reader).status_code == 404


def test_progress_and_closing_use_only_the_dataset_team(two_teams):
    client, admin, evaluators = two_teams
    status = {s["dataset"]: s for s in client.get("/api/admin/status", headers=admin).json()}
    assert set(status[DATASET]["progresso"]) == set(TEAMS[DATASET])
    assert set(status[OTHER]["progresso"]) == set(TEAMS[OTHER])

    for name in TEAMS[DATASET]:
        answer_all(client, DATASET, evaluators[name])
    # books fecha com os 3 dele, sem esperar ninguém do dblp
    assert client.post(f"/api/admin/{DATASET}/fechar", headers=admin).status_code == 200
    assert client.post(f"/api/admin/{OTHER}/fechar", headers=admin).status_code == 409


def test_login_lists_all_six_evaluators(two_teams):
    client, _, _ = two_teams
    assert set(client.get("/api/usuarios").json()) == set(CODES)


def test_real_configuration_has_three_per_dataset():
    assert {d: len(names) for d, names in EVALUATORS_BY_DATASET.items()} == {"books": 3, "dblp": 3}
    assert len(set(EVALUATORS)) == 6
    assert InterfaceSettings.code_variable("books_1") == "HV_EVALUATOR_BOOKS_1_CODE"


def test_logins_are_anonymous_and_names_only_in_admin(two_teams):
    client, admin, evaluators = two_teams
    assert not set(EVALUATOR_NAMES.values()) & set(EVALUATORS)
    assert not set(EVALUATOR_NAMES.values()) & set(client.get("/api/usuarios").json())
    status = {s["dataset"]: s for s in client.get("/api/admin/status", headers=admin).json()}
    assert set(status[DATASET]["nomes"]) == set(TEAMS[DATASET])
    assert client.get("/api/admin/status", headers=evaluators["livro_1"]).status_code == 403


def test_missing_code_names_the_variable():
    codes = {name: "x" for name in CODES if name != "artigo_2"}
    with pytest.raises(ValueError, match="HV_EVALUATOR_ARTIGO_2_CODE"):
        InterfaceSettings({DATASET: DATE, OTHER: DATE}, "data/results", codes, ADMIN_CODE, evaluators_by_dataset=TEAMS)
