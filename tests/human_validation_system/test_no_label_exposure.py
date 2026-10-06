"""
Nenhuma tela ou rota do avaliador expõe rótulo de referência, anotação de LLM, grupo ou classe.
"""
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.systems.human_validation_system.interface.api.server import WEB_DIR
from tests.human_validation_system.conftest import DATASET, answer_all

FORBIDDEN_KEYS = {
    "grupo", "classe", "classe_referencia", "rotulo_referencia", "ground_truth", "id_original", "text_id",
    "anotacao_llm_1", "anotacao_llm_2", "anotacao_llm_3", "rotulo_consolidado", "resolved_annotation",
}
EVALUATOR_FILES = ["index.html", "avaliar.html", "js/api.js", "js/login.js", "js/avaliar.js"]


def all_keys(value) -> set:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in all_keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in all_keys(v)}
    return set()


def answer_key(client) -> pd.DataFrame:
    path = client.app.state.controller.validation_dir(DATASET) / "rodada_01" / "gabarito.csv"
    return pd.read_csv(path)


@pytest.fixture
def evaluator_payloads(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    headers = evaluators["avaliador_1"]
    seen = answer_all(client, headers, leave=5)
    payloads = [
        client.get("/api/avaliacao/datasets", headers=headers).json(),
        client.get(f"/api/avaliacao/{DATASET}/proximo", headers=headers).json(),
        client.get(f"/api/avaliacao/{DATASET}/respostas", headers=headers).json(),
        *[client.get(f"/api/avaliacao/{DATASET}/documentos/{i}", headers=headers).json() for i in seen[:10]],
    ]
    return payloads


def test_evaluator_payloads_have_no_forbidden_keys(evaluator_payloads):
    leaked = all_keys(evaluator_payloads) & FORBIDDEN_KEYS
    assert not leaked, leaked


def test_evaluator_payloads_have_no_answer_key_values(client, evaluator_payloads):
    dumped = json.dumps(evaluator_payloads, ensure_ascii=False)
    key = answer_key(client)
    assert not any(text_id in dumped for text_id in key["id_original"])


def test_document_payload_shape_does_not_depend_on_group(client, admin, evaluators):
    """Documentos de grupos diferentes chegam com exatamente os mesmos campos."""
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    key = answer_key(client)
    headers = evaluators["avaliador_1"]
    shapes = {
        group: tuple(sorted(client.get(f"/api/avaliacao/{DATASET}/documentos/{doc}", headers=headers).json()))
        for group, doc in key.groupby("grupo")["id_anonimo"].first().items()
    }
    assert len(set(shapes.values())) == 1


def test_evaluator_cannot_reach_admin_routes(client, admin, evaluators):
    headers = evaluators["avaliador_1"]
    assert client.get("/api/admin/status", headers=headers).status_code == 403
    assert client.post(f"/api/admin/{DATASET}/rodadas", headers=headers).status_code == 403
    assert client.get(f"/api/admin/{DATASET}/planilha", headers=headers).status_code == 403
    assert client.get("/api/admin/status").status_code == 401


def test_evaluator_sees_only_own_answers(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    answer_all(client, evaluators["avaliador_2"])
    assert client.get(f"/api/avaliacao/{DATASET}/respostas", headers=evaluators["avaliador_1"]).json() == []


@pytest.mark.parametrize("name", EVALUATOR_FILES)
def test_evaluator_screens_do_not_reference_answer_key(client, name):
    source = (Path(WEB_DIR) / name).read_text(encoding="utf-8")
    assert client.get("/" if name == "index.html" else f"/static/{name}").status_code == 200
    for forbidden in FORBIDDEN_KEYS | {"gabarito"}:
        assert not re.search(rf"{forbidden}", source), f"{name} menciona '{forbidden}'"


def test_interface_database_stores_no_answer_key(client, admin):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    with sqlite3.connect(client.app.state.settings.db_path) as con:
        columns = {
            col[1] for table in ("rodadas", "documentos", "respostas", "sessoes")
            for col in con.execute(f"PRAGMA table_info({table})")
        }
    assert not columns & FORBIDDEN_KEYS
