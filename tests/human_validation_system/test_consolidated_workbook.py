"""
Planilha consolidada: quatro abas e nenhuma rodada anterior perdida.
"""
import pandas as pd

from src.systems.human_validation_system.interface.api.services.consolidated_workbook import ConsolidatedWorkbookWriter
from tests.human_validation_system.conftest import DATASET, answer_all


def run_round(client, admin, evaluators):
    """Abre a rodada se ainda não houver uma aberta (após o 1º fechamento ela abre sozinha)."""
    if client.get("/api/admin/status", headers=admin).json()[0]["estado"] != "aberta":
        assert client.post(f"/api/admin/{DATASET}/rodadas", headers=admin).status_code == 200
    for headers in evaluators.values():
        answer_all(client, headers)
    assert client.post(f"/api/admin/{DATASET}/fechar", headers=admin).status_code == 200


def read(client):
    return pd.read_excel(client.app.state.controller.workbook_path(DATASET), sheet_name=None)


def test_workbook_has_four_sheets(client, admin, evaluators):
    run_round(client, admin, evaluators)
    assert list(read(client)) == list(ConsolidatedWorkbookWriter.SHEETS)


def test_workbook_keeps_previous_rounds(client, admin, evaluators):
    run_round(client, admin, evaluators)
    round_1 = read(client)["respostas_individuais"]

    run_round(client, admin, evaluators)
    sheets = read(client)
    responses = sheets["respostas_individuais"]

    assert sorted(responses["rodada"].unique()) == [1, 2]
    kept = responses[responses["rodada"] == 1].drop(columns="atualizado_em").reset_index(drop=True)
    assert kept.equals(round_1.drop(columns="atualizado_em").reset_index(drop=True))
    assert list(sheets["historico"]["rodada"]) == [1, 2]
    assert sorted(sheets["consolidado_documentos"]["rodada"].unique()) == [1, 2]


def test_history_has_one_row_per_round(client, admin, evaluators):
    run_round(client, admin, evaluators)
    run_round(client, admin, evaluators)
    history = read(client)["historico"]
    assert len(history) == 2
    assert {"A_humano_ref", "A_mislabeling_ic_inf", "C_status"} <= set(history.columns)
