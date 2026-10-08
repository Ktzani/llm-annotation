"""
Envio do dataset_consenso.csv pela tela do administrador (deploy sem acesso ao disco do servidor).
"""
import pytest
from fastapi.testclient import TestClient

from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.interface.api.server import create_app
from tests.human_validation_system.conftest import ADMIN_CODE, CODES, DATASET, DATE, login, synthetic_consensus

NEW_DATE = "sem_csv"


@pytest.fixture
def empty_client(results_dir):
    """Experimento cujo CSV ainda não está no servidor."""
    client = TestClient(create_app(InterfaceSettings({DATASET: NEW_DATE}, str(results_dir), CODES, ADMIN_CODE)))
    return client, login(client, "admin", ADMIN_CODE)


def csv_bytes(df=None) -> bytes:
    return (synthetic_consensus() if df is None else df).to_csv(index=False).encode()


def put(client, headers, content):
    return client.put(f"/api/admin/{DATASET}/consenso", content=content, headers={**headers, "Content-Type": "text/csv"})


def test_round_needs_consensus_first(empty_client):
    client, admin = empty_client
    status = client.get("/api/admin/status", headers=admin).json()[0]
    assert status["consenso_disponivel"] is False and status["pode_iniciar"] is False
    response = client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    assert response.status_code == 409 and "CSV de consenso" in response.json()["detail"]


def test_admin_uploads_consensus_and_starts_round(empty_client):
    client, admin = empty_client
    response = put(client, admin, csv_bytes())
    assert response.status_code == 200 and response.json()["linhas"] == len(synthetic_consensus())

    status = client.get("/api/admin/status", headers=admin).json()[0]
    assert status["consenso_disponivel"] and status["pode_iniciar"] and not status["consenso_bloqueado"]
    assert client.post(f"/api/admin/{DATASET}/rodadas", headers=admin).status_code == 200


@pytest.mark.parametrize("content", [
    b"isto nao e um csv\x00\x01",
    b"text_id,text\n1,abc\n",
    csv_bytes(synthetic_consensus().drop(columns=["m3_consensus", "m3_consensus_score"])),
], ids=["nao_csv", "sem_colunas", "so_duas_llms"])
def test_invalid_consensus_is_rejected(empty_client, content):
    client, admin = empty_client
    assert put(client, admin, content).status_code == 422
    assert client.get("/api/admin/status", headers=admin).json()[0]["consenso_disponivel"] is False


def test_only_admin_uploads(empty_client):
    client, _ = empty_client
    evaluator = login(client, "avaliador_1", CODES["avaliador_1"])
    assert put(client, evaluator, csv_bytes()).status_code == 403


def test_consensus_is_locked_after_first_draw(client, admin, results_dir):
    """Depois do sorteio, só o MESMO arquivo é aceito (outro quebraria o registro da amostragem)."""
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    assert client.get("/api/admin/status", headers=admin).json()[0]["consenso_bloqueado"] is True

    original = (results_dir / DATASET / DATE / "consensus" / "dataset_consenso.csv").read_bytes()
    assert put(client, admin, original).status_code == 200
    changed = csv_bytes(synthetic_consensus().iloc[:-5])
    assert put(client, admin, changed).status_code == 409
