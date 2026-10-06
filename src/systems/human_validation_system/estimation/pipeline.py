"""
Pipeline de ESTIMAÇÃO da validação humana (margem de erro e critério de parada).

Lê as planilhas preenchidas de todas as rodadas entregues, consolida os
avaliadores por maioria, classifica cada documento nas quatro situações do RQ4
(SituationClassifier) e estima, por grupo de concordância, cada métrica com o
estimador estratificado por classe (pesos N_h do `amostragem.json`) e IC de
Wald. O grupo pode parar quando a métrica principal tem MoE <= ε e ao menos
INITIAL_ROUND_SIZE documentos válidos (regra do TLC, Merlo et al., CIKM'25).
Como a MoE é verificada a cada rodada (parada sequencial), a cobertura real do
IC fica um pouco abaixo da nominal; vale registrar isso ao reportar.

Estrutura de saída (em ``<results>/validacao_humana/<dataset>/<date>/estimativas/``):
    estimativa_ate_rodada_XX.csv    θ̂, IC, MoE e status por grupo e métrica
    estratos_ate_rodada_XX.csv      Detalhe por estrato (N_h, n_h, θ̂_h, W_h)
    documentos_ate_rodada_XX.csv    Gabarito + rótulo humano, situação, observações e métricas por documento
    historico.csv                   Uma linha por grupo/métrica/rodada (recalcular a rodada substitui as linhas dela)
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import pandas as pd
from loguru import logger

from src.config.datasets_collected import LABEL_MEANINGS
from src.config.human_validation import (
    CONFIDENCE_LEVEL,
    ESTIMATION_METRICS,
    EVALUATORS,
    INITIAL_ROUND_SIZE,
    INSUFFICIENT_INFO_OPTION,
    MOE_THRESHOLD,
    PRIMARY_METRIC,
    VARIANCE_PSEUDO_COUNT,
)
from src.systems.human_validation_system.estimation.human_label_aggregator import HumanLabelAggregator
from src.systems.human_validation_system.estimation.inter_rater_agreement import InterRaterAgreement
from src.systems.human_validation_system.estimation.response_loader import EvaluationResponseLoader
from src.systems.human_validation_system.estimation.situation_classifier import SituationClassifier
from src.systems.human_validation_system.estimation.stopping_status import StoppingStatusReader
from src.systems.human_validation_system.estimation.stratified_estimator import StratifiedEstimator
from src.systems.human_validation_system.pipeline import DEFAULT_RESULTS_DIR, HumanValidationPipeline
from src.systems.human_validation_system.sampling.sampling_ledger import SamplingLedger


class ResponseSource(Protocol):
    """Origem das respostas: planilhas (EvaluationResponseLoader) ou banco da interface web."""

    def load(self, validation_dir: Path, n_rounds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...


class HumanValidationEstimationConfig:
    """Configurações da estimação da validação humana."""

    def __init__(
        self,
        dataset_name: str,
        specific_date: str,
        results_dir: str = DEFAULT_RESULTS_DIR,
        evaluators: Optional[List[str]] = None,
        moe_threshold: float = MOE_THRESHOLD,
        confidence_level: float = CONFIDENCE_LEVEL,
        primary_metric: str = PRIMARY_METRIC,
        min_sample: int = INITIAL_ROUND_SIZE,
        pseudo_count: float = VARIANCE_PSEUDO_COUNT,
    ):
        self.dataset_name = dataset_name
        self.specific_date = specific_date
        self.results_dir = results_dir
        self.evaluators = evaluators or EVALUATORS
        self.moe_threshold = moe_threshold
        self.confidence_level = confidence_level
        self.primary_metric = primary_metric
        # n mínimo antes de ativar o critério de parada
        self.min_sample = min_sample
        self.pseudo_count = pseudo_count


class HumanValidationEstimationPipeline:
    """Pipeline de estimação da validação humana."""

    GROUPS = ("A", "B", "C")

    def __init__(self, config: HumanValidationEstimationConfig, response_source: Optional[ResponseSource] = None):
        self.config = config
        self.validation_dir = HumanValidationPipeline.validation_dir(
            config.results_dir, config.dataset_name, config.specific_date
        )
        self.output_dir = self.validation_dir / StoppingStatusReader.OUTPUT_SUBDIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        label_names = LABEL_MEANINGS[config.dataset_name]
        self.class_names = {int(k): v for k, v in label_names.items()}

        # Inicializar componentes
        n = len(config.evaluators)
        self.ledger = SamplingLedger(self.validation_dir)
        self.loader = response_source or EvaluationResponseLoader(
            config.evaluators, [self.class_names[c] for c in sorted(self.class_names)], INSUFFICIENT_INFO_OPTION
        )
        self.aggregator = HumanLabelAggregator(n)
        self.situation_classifier = SituationClassifier(n, INSUFFICIENT_INFO_OPTION)
        self.agreement = InterRaterAgreement(n)
        self.estimator = StratifiedEstimator(config.confidence_level, config.pseudo_count)
        logger.success(f"✓ Setup completo — saída em: {self.output_dir}")

    def _status(self, metric: str, summary: dict) -> str:
        if metric != self.config.primary_metric:
            return ""
        if summary["n"] < self.config.min_sample:
            return "continuar (n insuficiente)"
        return StoppingStatusReader.STOP if summary["moe"] <= self.config.moe_threshold else "continuar"

    def _estimate_group(self, group: str, docs: pd.DataFrame, responses: pd.DataFrame):
        group_docs = docs[docs["grupo"] == group]
        sizes = {int(c): n for c, n in self.ledger.data["strata_sizes"][group].items()}
        complete_ids = group_docs.loc[group_docs["completo"], "id_anonimo"]
        kappa = self.agreement.fleiss(responses, complete_ids)
        raw_agreement = self.agreement.raw(responses, complete_ids)

        summaries, details = [], []
        for metric in ESTIMATION_METRICS:
            values = (
                group_docs[["rotulo_referencia", metric]]
                .dropna(subset=[metric])
                .rename(columns={"rotulo_referencia": "estrato", metric: "valor"})
            )
            summary, detail = self.estimator.estimate(values, sizes, where=f"Grupo {group} / {metric}")
            summaries.append({
                "grupo": group,
                "metrica": metric,
                "principal": metric == self.config.primary_metric,
                "n_entregues": len(group_docs),
                "n_completos": int(group_docs["completo"].sum()),
                **summary,
                "kappa_fleiss": kappa,
                **raw_agreement,
                "status": self._status(metric, summary),
            })
            detail.insert(0, "metrica", metric)
            detail.insert(0, "grupo", group)
            detail.insert(3, "classe", detail["estrato"].map(self.class_names))
            details.append(detail)
        return summaries, details

    def _log_primary(self, result: pd.DataFrame) -> None:
        logger.info(f"Métrica principal: {ESTIMATION_METRICS[self.config.primary_metric]}")
        logger.info(f"Critério: MoE <= {self.config.moe_threshold} ({self.config.confidence_level:.0%})")
        for _, r in result[result["principal"]].iterrows():
            logger.info(
                f"  Grupo {r['grupo']}: θ̂={r['theta']:.3f}  IC=[{r['ic_inferior']:.3f}, {r['ic_superior']:.3f}]  "
                f"MoE={r['moe']:.3f}  n={r['n']}/{r['n_entregues']}  κ={r['kappa_fleiss']:.2f}  "
                f"unânime={r['acordo_unanime']:.2f}  -> {r['status']}"
            )

    def _log_situations(self, result: pd.DataFrame) -> None:
        logger.info("Situações do RQ4 (proporção estimada ± MoE):")
        situations = result[result["metrica"].isin(SituationClassifier.SITUATIONS)]
        for group, rows in situations.groupby("grupo"):
            parts = [f"{r['metrica']}={r['theta']:.2f}±{r['moe']:.2f}" for _, r in rows.iterrows()]
            logger.info(f"  Grupo {group}: " + "  ".join(parts))

    def _save(self, result: pd.DataFrame, strata: pd.DataFrame, docs: pd.DataFrame, n_rounds: int) -> None:
        tag = f"ate_rodada_{n_rounds:02d}"
        result.to_csv(StoppingStatusReader.summary_path(self.validation_dir, n_rounds), index=False)
        strata.to_csv(self.output_dir / f"estratos_{tag}.csv", index=False)
        docs.to_csv(self.output_dir / f"documentos_{tag}.csv", index=False)

        history_path = self.output_dir / "historico.csv"
        history = result.assign(calculado_em=datetime.now().isoformat(timespec="seconds"), rodadas=n_rounds)
        if history_path.exists():
            previous = pd.read_csv(history_path)
            history = pd.concat([previous[previous["rodadas"] != n_rounds], history], ignore_index=True)
        history.sort_values(["rodadas", "grupo"], kind="stable").to_csv(history_path, index=False)
        logger.success(f"Estimativas salvas em: {self.output_dir}")

    def run(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info(f"Estimação da validação humana — {self.config.dataset_name}")
        logger.info("=" * 60)

        if not self.ledger.exists():
            raise FileNotFoundError(f"Registro de amostragem não encontrado: {self.ledger.path}")
        self.ledger.load()
        n_rounds = self.ledger.last_round

        responses, answer_key = self.loader.load(self.validation_dir, n_rounds)
        docs = self.aggregator.aggregate(responses, answer_key)
        docs = self.situation_classifier.classify(responses, docs)

        summaries, details = [], []
        for group in self.GROUPS:
            s, d = self._estimate_group(group, docs, responses)
            summaries += s
            details += d

        result = pd.DataFrame(summaries)
        self._log_primary(result)
        self._log_situations(result)
        self._save(result, pd.concat(details, ignore_index=True), docs, n_rounds)

        pending = result.loc[result["principal"] & (result["status"] != StoppingStatusReader.STOP), "grupo"].tolist()
        if pending:
            logger.info(f"Grupos que ainda precisam de rodada: {pending}")
        else:
            logger.success("Todos os grupos atingiram a margem de erro desejada")
        return result
