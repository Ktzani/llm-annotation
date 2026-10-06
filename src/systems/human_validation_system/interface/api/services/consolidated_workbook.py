"""
Consolidated Workbook - Planilha Excel cumulativa da validação humana (uma por dataset)
"""
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config.human_validation import PRIMARY_METRIC


class ConsolidatedWorkbookWriter:
    """
    Reescreve a planilha consolidada a cada fechamento de rodada.

    É sempre regerada a partir das fontes cumulativas (banco com todas as
    respostas, estimativa até a última rodada e histórico), então nenhuma rodada
    anterior se perde.

    Abas: respostas_individuais, consolidado_documentos, resumo, historico.
    """

    SHEETS = ("respostas_individuais", "consolidado_documentos", "resumo", "historico")
    HISTORY_METRICS = {PRIMARY_METRIC: "humano_ref", "benchmark_mislabeling": "mislabeling"}

    def __init__(self):
        logger.debug("ConsolidatedWorkbookWriter inicializado")

    def history_by_round(self, history: pd.DataFrame) -> pd.DataFrame:
        """Histórico longo (rodada x grupo x métrica) -> uma linha por rodada."""
        rows = []
        for rodada, by_round in history.groupby("rodadas"):
            row = {"rodada": int(rodada), "calculado_em": by_round["calculado_em"].max()}
            for group, by_group in by_round.groupby("grupo"):
                primary = by_group[by_group["metrica"] == PRIMARY_METRIC].iloc[0]
                row[f"{group}_n"] = int(primary["n"])
                row[f"{group}_kappa_fleiss"] = primary["kappa_fleiss"]
                row[f"{group}_acordo_unanime"] = primary["acordo_unanime"]
                for metric, alias in self.HISTORY_METRICS.items():
                    m = by_group[by_group["metrica"] == metric].iloc[0]
                    row[f"{group}_{alias}"] = m["theta"]
                    row[f"{group}_{alias}_ic_inf"] = m["ic_inferior"]
                    row[f"{group}_{alias}_ic_sup"] = m["ic_superior"]
                    row[f"{group}_{alias}_moe"] = m["moe"]
                row[f"{group}_status"] = primary["status"]
            rows.append(row)
        return pd.DataFrame(rows)

    def write(
        self,
        path: Path,
        responses: pd.DataFrame,
        documents: pd.DataFrame,
        summary: pd.DataFrame,
        history: pd.DataFrame,
    ) -> Path:
        frames = (responses, documents, summary, self.history_by_round(history))
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in zip(self.SHEETS, frames):
                frame.to_excel(writer, sheet_name=name, index=False)
                writer.sheets[name].freeze_panes = "A2"
        logger.success(f"Planilha consolidada salva: {path}")
        return Path(path)
