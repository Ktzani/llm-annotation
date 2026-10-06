"""
Fixtures da validação humana: dataset de consenso sintético e cliente da interface.
"""
import hashlib
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from loguru import logger

from src.config.human_validation import INSUFFICIENT_INFO_OPTION
from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.interface.api.server import create_app

DATASET = "books"
DATE = "sintetico"
N_CLASSES = 8
DOCS_PER_STRATUM = 40
CODES = {"avaliador_1": "c1", "avaliador_2": "c2", "avaliador_3": "c3"}
ADMIN_CODE = "adm"
MODELS = ("m1", "m2", "m3")


@pytest.fixture(autouse=True)
def quiet_logs():
    logger.remove()
    yield


def synthetic_consensus() -> pd.DataFrame:
    """Grupos A (3x0 contra a referência), B (2x1) e C (3x0 a favor) em todas as classes."""
    rows = []
    for group in "ABC":
        for gt in range(N_CLASSES):
            other = (gt + 1) % N_CLASSES
            votes = {"A": (other, other, other), "B": (other, other, gt), "C": (gt, gt, gt)}[group]
            for i in range(DOCS_PER_STRATUM):
                text = f"documento {group}{gt}-{i} " + "palavra " * (5 + i)
                row = {"text_id": hashlib.md5(text.encode()).hexdigest(), "text": text, "ground_truth": gt}
                for model, vote in zip(MODELS, votes):
                    row[f"{model}_consensus"] = vote
                    row[f"{model}_consensus_score"] = 1.0
                row["resolved_annotation"] = max(set(votes), key=votes.count)
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """`<tmp>/results`; a validação humana vai para a irmã `<tmp>/validacao_humana`."""
    results = tmp_path / "results"
    consensus_dir = results / DATASET / DATE / "consensus"
    consensus_dir.mkdir(parents=True)
    synthetic_consensus().to_csv(consensus_dir / "dataset_consenso.csv", index=False)
    return results


@pytest.fixture
def client(results_dir: Path) -> TestClient:
    settings = InterfaceSettings({DATASET: DATE}, str(results_dir), CODES, ADMIN_CODE)
    return TestClient(create_app(settings))


def login(client: TestClient, user: str, code: str) -> dict:
    token = client.post("/api/login", json={"usuario": user, "codigo": code}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin(client: TestClient) -> dict:
    return login(client, "admin", ADMIN_CODE)


@pytest.fixture
def evaluators(client: TestClient) -> dict:
    return {user: login(client, user, code) for user, code in CODES.items()}


def answer_all(client: TestClient, headers: dict, label_for=None, leave: int = 0) -> list:
    """Responde a rodada na ordem; `leave` documentos ficam sem resposta. Retorna a ordem vista."""
    options = client.get(f"/api/avaliacao/{DATASET}/opcoes", headers=headers).json()
    first_label = options["classes"][0]["rotulo"]
    seen = []
    while True:
        nxt = client.get(f"/api/avaliacao/{DATASET}/proximo", headers=headers).json()
        if nxt["concluido"] or nxt["total"] - nxt["respondidos"] <= leave:
            return seen
        doc = nxt["documento"]
        seen.append(doc["id_anonimo"])
        label = label_for(doc) if label_for else first_label
        body = {"rotulo_escolhido": label, "outro_rotulo_possivel": "não"}
        response = client.put(f"/api/avaliacao/{DATASET}/respostas/{doc['id_anonimo']}", json=body, headers=headers)
        assert response.status_code == 200, response.text


UNDECIDABLE = INSUFFICIENT_INFO_OPTION
