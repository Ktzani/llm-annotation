"""
Pipeline de aplicação do CONSENSO entre LLMs.

Aplica o consenso sobre as anotações de um experimento (`annotations.csv`),
de forma independente dos notebooks de análise. Reúne o que antes estava
disperso no notebook de análise de consenso:

    1. Calcula o consenso (`ConsensusCalculator` / `ConsensusEvaluator`).
    2. Gera o relatório de concordância (pairwise / Cohen / Fleiss) e os casos
       problemáticos em `<results>/<dataset>/<date>/consensus/`.
    3. (Opcional) Valida o consenso final contra o ground truth.
    4. Exporta o dataset de consenso em
       `<results>/<dataset>/<date>/consensus/dataset_consenso.csv` e os demais
       artefatos consolidados em `<results>/<dataset>/<date>/summary/`:
           consensus/dataset_consenso.csv   Dataset com consenso (resolved_annotation)
           summary/alta_confianca.csv       Subconjunto com score >= threshold
           summary/necessita_revisao.csv    Subconjunto com score < threshold
           summary/sumario_experimento.json Métricas resumidas

IMPORTANTE: o consenso é calculado e exportado SEM filtragem — o summary e o
relatório refletem o dataset completo (inclusive instâncias inválidas/`-1` e
problemáticas/baixo consenso, que precisam ser identificadas e validadas). A
filtragem é responsabilidade dos consumidores downstream (ex.: o fine-tuning,
via `remove_invalid_instances` / `remove_problematic_instances` / IS).

A parte de gráficos (calibração, ECE/BBS, heatmaps) permanece em notebook para
análise posterior — este pipeline materializa apenas os artefatos de dados.
"""
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.utils.get_latest_results_date import get_latest_results_date
from src.llm_annotation_system.consensus.consensus_calculator import ConsensusCalculator
from src.llm_annotation_system.consensus.consensus_evaluator import ConsensusEvaluator

DEFAULT_RESULTS_DIR = "C:\\Users\\gabri\\Documents\\GitHub\\llm-annotation\\data\\results"


class ConsensusConfig:
    """Configurações da aplicação de consenso."""

    def __init__(
        self,
        dataset_name: str = "movie_review",
        results_dir: str = DEFAULT_RESULTS_DIR,
        specific_date: str = "latest",
        consensus_threshold: float = 0.8,
        consensus_strategy: str = "majority_vote",
        categories: Optional[List[int]] = None,
    ):
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        self.specific_date = specific_date
        self.consensus_threshold = consensus_threshold
        self.consensus_strategy = consensus_strategy
        # Categorias válidas; se None, são derivadas dos próprios dados.
        self.categories = categories


class ConsensusPipeline:
    """Pipeline principal de aplicação de consenso."""

    #: Nome do dataset de consenso, salvo dentro da pasta `consensus/`.
    DATASET_FILENAME = "dataset_consenso.csv"

    @classmethod
    def dataset_path(cls, results_dataset_path: Path) -> Path:
        """Caminho canônico do dataset de consenso (`consensus/dataset_consenso.csv`)."""
        return Path(results_dataset_path) / "consensus" / cls.DATASET_FILENAME

    def __init__(self, config: ConsensusConfig):
        self.config = config
        self.results_dataset_path = self._get_results_path()
        self.summary_dir = self.results_dataset_path / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        logger.success(f"✓ Setup completo — saída em: {self.results_dataset_path}")

    def _get_results_path(self) -> Path:
        date = self.config.specific_date
        if date == "latest":
            date = get_latest_results_date(self.config.results_dir, self.config.dataset_name)
        return Path(self.config.results_dir) / self.config.dataset_name / date

    # -------------------------------------------------------------------------
    # CARGA
    # -------------------------------------------------------------------------
    def load_annotations(self) -> pd.DataFrame:
        path = self.results_dataset_path / "annotations.csv"
        if not path.exists():
            raise FileNotFoundError(f"Anotações não encontradas: {path}")

        df = pd.read_csv(path)
        before = len(df)
        df = df.drop_duplicates(subset=["text_id"]).reset_index(drop=True)
        if before != len(df):
            logger.info(f"Removidas {before - len(df)} duplicatas por text_id.")

        logger.info(f"Carregado: {len(df)} anotações de {path}")
        return df

    def _resolve_categories(
        self,
        df: pd.DataFrame,
        consensus_cols: List[str],
    ) -> List[int]:
        """Define as categorias válidas (config explícita ou derivadas dos dados)."""
        if self.config.categories is not None:
            return list(self.config.categories)

        labels = set()
        for col in consensus_cols:
            labels.update(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
        if "ground_truth" in df.columns:
            labels.update(pd.to_numeric(df["ground_truth"], errors="coerce").dropna().tolist())

        labels.discard(-1)
        categories = sorted(int(v) for v in labels)
        logger.info(f"Categorias derivadas dos dados: {categories}")
        return categories

    # -------------------------------------------------------------------------
    # EXPORTAÇÃO
    # -------------------------------------------------------------------------
    def _export(
        self,
        df: pd.DataFrame,
        report: dict,
        categories: List[int],
        models: List[str],
        cls_report: Optional[dict],
    ) -> None:
        consensus_dataset_path = self.dataset_path(self.results_dataset_path)
        consensus_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(consensus_dataset_path, index=False)
        logger.info(f"✓ {consensus_dataset_path.name}: {len(df)} registros")

        thr = self.config.consensus_threshold
        high_conf = df[df["consensus_score"] >= thr]
        high_conf.to_csv(self.summary_dir / "alta_confianca.csv", index=False)
        logger.info(f"✓ alta_confianca.csv: {len(high_conf)} registros")

        low_conf = df[df["consensus_score"] < thr]
        low_conf.to_csv(self.summary_dir / "necessita_revisao.csv", index=False)
        logger.info(f"✓ necessita_revisao.csv: {len(low_conf)} registros")

        summary = {
            "dataset": {
                "name": self.config.dataset_name,
                "total_texts": int(len(df)),
                "categories": categories,
                "has_ground_truth": "ground_truth" in df.columns,
            },
            "config": {
                "models": models,
                "total_models": len(models),
                "consensus_threshold": thr,
                "consensus_strategy": self.config.consensus_strategy,
            },
            "results": {
                "consensus_mean": float(df["consensus_score"].mean()),
                "consensus_median": float(df["consensus_score"].median()),
                "high_consensus": int((df["consensus_level"] == "high").sum()),
                "medium_consensus": int((df["consensus_level"] == "medium").sum()),
                "low_consensus": int((df["consensus_level"] == "low").sum()),
            },
            "metrics": {
                "fleiss_kappa": float(report["fleiss_kappa"]),
                "fleiss_interpretation": report["fleiss_interpretation"],
            },
        }
        if cls_report is not None:
            summary["validation"] = {"classification_report": cls_report}

        with open(self.summary_dir / "sumario_experimento.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.success("✓ Resultados exportados com sucesso!")

    # -------------------------------------------------------------------------
    # ORQUESTRAÇÃO
    # -------------------------------------------------------------------------
    def run(self) -> dict:
        logger.info("=" * 60)
        logger.info(f"Aplicação de consenso — {self.config.dataset_name}")
        logger.info("=" * 60)

        df_annotations = self.load_annotations()

        calculator = ConsensusCalculator(
            consensus_threshold=self.config.consensus_threshold,
            default_strategy=self.config.consensus_strategy,
        )

        consensus_cols = calculator._extract_consensus_columns(df_annotations)
        models = [col.replace("_consensus", "") for col in consensus_cols]
        categories = self._resolve_categories(df_annotations, consensus_cols)

        evaluator = ConsensusEvaluator(
            categories=categories,
            calculator=calculator,
            output_dir=self.results_dataset_path,
        )

        # 1. Consenso sobre TODAS as anotações — SEM filtragem. O summary e o
        #    relatório precisam refletir o dataset completo (inclusive os casos
        #    inválidos/problemáticos). A filtragem (-1, problemáticos, IS) é feita
        #    downstream, no fine-tuning. Índice resetado p/ o cálculo posicional
        #    do Fleiss' Kappa.
        df = evaluator.compute_consensus(df_annotations).reset_index(drop=True)

        # 2. Relatório de concordância sobre o df completo → identifica e salva
        #    os casos problemáticos (consensus/problematic_cases.csv).
        report = evaluator.generate_consensus_report(df=df)

        # 3. Validação com ground truth (opcional). Aqui SIM filtramos os
        #    inválidos (resolved_annotation == -1): eles não são uma classe real
        #    e poluiriam as métricas (accuracy / classification_report). A
        #    filtragem é aplicada APENAS na validação — o relatório e o export
        #    permanecem sobre o df completo.
        cls_report = None
        if "ground_truth" in df.columns and df["ground_truth"].notna().any():
            df_valid = df[df["resolved_annotation"] != -1]
            _, cls_report, _ = evaluator.evaluate_ground_truth(df_valid)

        # 4. Exportação do dataset de consenso completo.
        self._export(df, report, categories, models, cls_report)

        return {
            "df_with_consensus": df,
            "report": report,
            "categories": categories,
            "models": models,
        }
