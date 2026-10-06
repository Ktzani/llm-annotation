"""
Reiniciar tudo: exige confirmação, é só do administrador e guarda backup antes de limpar.
"""
from pathlib import Path

import pandas as pd

from tests.human_validation_system.conftest import DATASET, answer_all


def test_reset_requires_exact_confirmation(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    for word in ("", "reiniciar", "sim"):
        assert client.post("/api/admin/reiniciar", json={"confirmacao": word}, headers=admin).status_code == 400
    assert client.get("/api/admin/status", headers=admin).json()[0]["estado"] == "aberta"


def test_only_admin_can_reset(client, evaluators):
    response = client.post("/api/admin/reiniciar", json={"confirmacao": "REINICIAR"}, headers=evaluators["avaliador_1"])
    assert response.status_code == 403


def test_reset_clears_everything_and_keeps_backup(client, admin, evaluators):
    client.post(f"/api/admin/{DATASET}/rodadas", headers=admin)
    answer_all(client, evaluators["avaliador_1"])
    first_round = answer_all(client, evaluators["avaliador_2"])
    validation_dir = client.app.state.controller.validation_dir(DATASET)

    response = client.post("/api/admin/reiniciar", json={"confirmacao": "REINICIAR"}, headers=admin)
    assert response.status_code == 200
    backup = Path(response.json()["backups"][0])

    assert client.get("/api/admin/status", headers=admin).json()[0]["estado"] == "sem_rodada"
    assert not validation_dir.exists()
    saved = pd.read_csv(backup / "banco_respostas.csv")
    assert set(saved["avaliador"]) == {"avaliador_1", "avaliador_2"}
    assert (backup / "arquivos" / "amostragem.json").exists()

    # Recomeça do zero: mesma semente, mesmos documentos na rodada 1
    assert client.post(f"/api/admin/{DATASET}/rodadas", headers=admin).status_code == 200
    assert answer_all(client, evaluators["avaliador_2"]) == first_round
