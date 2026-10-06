"""
Round Controller - Abre, acompanha e fecha rodadas da validação humana via interface
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.config.datasets_collected import LABEL_MEANINGS
from src.config.human_validation import (
    CLASS_DEFINITIONS,
    EXAMPLE_MAX_CHARS,
    INSUFFICIENT_INFO_OPTION,
    PRIMARY_METRIC,
)
from src.systems.human_validation_system.estimation.pipeline import (
    HumanValidationEstimationConfig,
    HumanValidationEstimationPipeline,
)
from src.systems.human_validation_system.estimation.situation_classifier import SituationClassifier
from src.systems.human_validation_system.estimation.stopping_status import StoppingStatusReader
from src.systems.human_validation_system.interface.api.services.consolidated_workbook import ConsolidatedWorkbookWriter
from src.systems.human_validation_system.interface.api.services.database_response_source import DatabaseResponseSource
from src.systems.human_validation_system.interface.api.services.response_store import ResponseStore
from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.pipeline import HumanValidationConfig, HumanValidationPipeline
from src.systems.human_validation_system.sampling.sampling_ledger import SamplingLedger


class RoundStateError(RuntimeError):
    """Operação incompatível com o estado da rodada (ex.: abrir com rodada aberta)."""


class RoundIncompleteError(RoundStateError):
    """Tentativa de fechar a rodada com avaliador incompleto."""


class RoundController:
    """
    Conduz o ciclo de rodadas de cada dataset.

    Responsabilidades:
    - Abrir a próxima rodada chamando a amostragem e publicando só id_anonimo + texto
    - Informar o progresso de cada avaliador
    - Fechar a rodada somente com todos completos: estimação (respostas do banco),
      critério de parada e planilha consolidada
    """

    def __init__(self, settings: InterfaceSettings, store: ResponseStore):
        self.settings = settings
        self.store = store
        self.workbook_writer = ConsolidatedWorkbookWriter()
        logger.debug(f"RoundController inicializado ({list(settings.experiments)})")

    # ------------------------------------------------------------- caminhos
    def _check_dataset(self, dataset: str) -> None:
        if dataset not in self.settings.experiments:
            raise KeyError(f"Dataset não configurado: {dataset}")

    def store_key(self, dataset: str) -> str:
        """Chave do experimento no banco (dataset + data): rodadas de testes ou de outra data nunca se misturam."""
        return f"{dataset}/{self.settings.experiments[dataset]}"

    def validation_dir(self, dataset: str) -> Path:
        return HumanValidationPipeline.validation_dir(self.settings.results_dir, dataset, self.settings.experiments[dataset])

    def workbook_path(self, dataset: str) -> Path:
        return self.validation_dir(dataset) / f"validacao_consolidada_{dataset}.xlsx"

    def class_names(self, dataset: str) -> List[str]:
        meanings = LABEL_MEANINGS[dataset]
        return [meanings[k] for k in sorted(meanings, key=int)]

    def _sampling_config(self, dataset: str) -> HumanValidationConfig:
        return HumanValidationConfig(
            dataset_name=dataset,
            specific_date=self.settings.experiments[dataset],
            results_dir=self.settings.results_dir,
            increment_size=self.settings.increment_size,
            evaluators=self.settings.evaluators,
            export_sheets=False,
        )

    # ------------------------------------------------------------- guia
    def guide(self, dataset: str) -> Dict:
        """Classes (nome + descrição de apoio) e exemplos reservados do guia."""
        self._check_dataset(dataset)
        meanings = LABEL_MEANINGS[dataset]
        definitions = CLASS_DEFINITIONS.get(dataset, {})
        examples: Dict[str, List[str]] = {name: [] for name in meanings.values()}

        path = self.validation_dir(dataset) / "exemplos_guia.csv"
        if path.exists():
            for _, row in pd.read_csv(path).iterrows():
                text = " ".join(str(row["text"]).split())
                if len(text) > EXAMPLE_MAX_CHARS:
                    text = text[:EXAMPLE_MAX_CHARS].rstrip() + "…"
                examples[meanings[str(int(row["ground_truth"]))]].append(text)

        return {
            "classes": [
                {"rotulo": meanings[k], "descricao": definitions.get(k, ""), "exemplos": examples[meanings[k]]}
                for k in sorted(meanings, key=int)
            ],
            "opcao_indecidivel": INSUFFICIENT_INFO_OPTION,
        }

    # ------------------------------------------------------------- status
    def progress(self, dataset: str, round_number: int) -> Dict[str, Dict[str, int]]:
        total = self.store.round_size(self.store_key(dataset), round_number)
        answered = self.store.progress(self.store_key(dataset), round_number, self.settings.evaluators)
        return {e: {"respondidos": n, "total": total, "faltam": total - n} for e, n in answered.items()}

    def last_result(self, dataset: str, round_number: int) -> Optional[Dict]:
        """Resumo da estimativa da rodada (se já calculada)."""
        path = StoppingStatusReader.summary_path(self.validation_dir(dataset), round_number)
        if not path.exists():
            return None
        summary = pd.read_csv(path)
        groups = {}
        for group, rows in summary.groupby("grupo"):
            by_metric = rows.set_index("metrica")
            primary = by_metric.loc[PRIMARY_METRIC]
            groups[group] = {
                "n": int(primary["n"]),
                "kappa_fleiss": None if pd.isna(primary["kappa_fleiss"]) else float(primary["kappa_fleiss"]),
                "acordo_unanime": None if pd.isna(primary["acordo_unanime"]) else float(primary["acordo_unanime"]),
                "acordo_par_a_par": None if pd.isna(primary["acordo_par_a_par"]) else float(primary["acordo_par_a_par"]),
                "status": primary["status"],
                "metricas": {
                    m: {k: float(by_metric.loc[m, k]) for k in ("theta", "ic_inferior", "ic_superior", "moe")}
                    for m in (PRIMARY_METRIC, *SituationClassifier.SITUATIONS)
                },
            }
        stopped = all(g["status"] == StoppingStatusReader.STOP for g in groups.values())
        return {"rodada": round_number, "grupos": groups, "todos_pararam": stopped}

    def status(self, dataset: str) -> Dict:
        self._check_dataset(dataset)
        current = self.store.current_round(self.store_key(dataset))
        if current is None:
            return {"dataset": dataset, "rodada": None, "estado": "sem_rodada", "pode_iniciar": True}

        round_number, state = current["rodada"], current["estado"]
        progress = self.progress(dataset, round_number)
        complete = all(p["faltam"] == 0 for p in progress.values())
        result = self.last_result(dataset, round_number) if state == "fechada" else None
        return {
            "dataset": dataset,
            "rodada": round_number,
            "estado": state,
            "progresso": progress,
            "pode_fechar": state == "aberta" and complete,
            "pode_iniciar": state == "fechada" and not (result and result["todos_pararam"]),
            "resultado": result,
            "planilha_disponivel": self.workbook_path(dataset).exists(),
        }

    # ------------------------------------------------------------- ciclo
    def start_round(self, dataset: str) -> Dict:
        """Sorteia (ou republica) a próxima rodada e a abre para os avaliadores."""
        self._check_dataset(dataset)
        if self.store.open_round(self.store_key(dataset)) is not None:
            raise RoundStateError("Já existe uma rodada aberta para este dataset")

        current = self.store.current_round(self.store_key(dataset))
        published = current["rodada"] if current else 0
        ledger = SamplingLedger(self.validation_dir(dataset))
        sampled = 0
        if ledger.exists():
            ledger.load()
            sampled = ledger.last_round

        pipeline = HumanValidationPipeline(self._sampling_config(dataset))
        # Rodada já sorteada mas não publicada (ex.: falha anterior): republica a mesma
        round_dir = pipeline.run(round_number=published + 1 if sampled > published else None)
        if round_dir is None:
            raise RoundStateError("Todos os grupos atingiram o critério de parada: nenhuma rodada nova")

        round_number = int(round_dir.name.split("_")[-1])
        documents = pd.read_csv(round_dir / HumanValidationPipeline.DOCUMENTS_FILE)
        self.store.publish_round(self.store_key(dataset), round_number, zip(documents["id_anonimo"], documents["texto"]))
        logger.success(f"{dataset}: rodada {round_number} aberta com {len(documents)} documentos")
        return self.status(dataset)

    def close_round(self, dataset: str) -> Dict:
        """Fecha a rodada aberta se todos terminaram; roda a estimação e a planilha consolidada."""
        self._check_dataset(dataset)
        round_number = self.store.open_round(self.store_key(dataset))
        if round_number is None:
            raise RoundStateError("Não há rodada aberta para este dataset")

        missing = {e: p["faltam"] for e, p in self.progress(dataset, round_number).items() if p["faltam"]}
        if missing:
            raise RoundIncompleteError(f"Avaliadores com documentos pendentes: {missing}")

        # Trava antes de estimar (nenhuma edição durante a análise); reabre se a análise falhar
        self.store.close_round(self.store_key(dataset), round_number)
        try:
            estimation = HumanValidationEstimationPipeline(
                HumanValidationEstimationConfig(
                    dataset_name=dataset,
                    specific_date=self.settings.experiments[dataset],
                    results_dir=self.settings.results_dir,
                    evaluators=self.settings.evaluators,
                ),
                response_source=DatabaseResponseSource(self.store, self.store_key(dataset)),
            )
            summary = estimation.run()
            self._write_workbook(dataset, round_number, summary)
        except Exception:
            self.store.reopen_round(self.store_key(dataset), round_number)
            raise
        logger.success(f"{dataset}: rodada {round_number} fechada")
        return self.status(dataset)

    def _write_workbook(self, dataset: str, round_number: int, summary: pd.DataFrame) -> None:
        estimates = self.validation_dir(dataset) / StoppingStatusReader.OUTPUT_SUBDIR
        self.workbook_writer.write(
            self.workbook_path(dataset),
            responses=self.store.all_responses(self.store_key(dataset), round_number),
            documents=pd.read_csv(estimates / f"documentos_ate_rodada_{round_number:02d}.csv"),
            summary=summary,
            history=pd.read_csv(estimates / "historico.csv"),
        )
