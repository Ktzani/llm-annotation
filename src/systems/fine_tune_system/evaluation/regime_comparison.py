"""Regime Comparison - Compara por fold os quatro regimes de treino (RQ6) entre si e com o benchmark"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.systems.fine_tune_system.core.run_versioner import FineTuningRunVersioner
from src.systems.fine_tune_system.evaluation.paired_comparator import PairedSystemComparator
from src.utils.get_latest_results_date import get_latest_results_date

REGIMES = ["aggregated", "aggregated_replicated", "perspectivism", "soft_labels"]
BENCHMARK = "benchmark"


class RegimeComparison:
    """
    Reúne as execuções dos regimes de cada dataset e as compara pareadas por fold
    Responsabilidades: localizar a última execução de cada regime, montar as métricas por fold (geral e por estrato de concordância), ler o benchmark e aplicar o PairedSystemComparator
    """

    REGIMES = REGIMES
    BENCHMARK = BENCHMARK
    # Cada contraste isola uma parte da RQ6 (diff = A − B)
    REGIME_CONTRASTS = [
        ("aggregated_replicated", "aggregated"),     # efeito da repetição
        ("perspectivism", "aggregated_replicated"),  # preservar a divergência, mesmo nº de linhas
        ("soft_labels", "aggregated"),               # preservar a divergência, uma linha por texto
        ("soft_labels", "perspectivism"),            # duas formas de preservar
    ]
    BENCHMARK_CONTRASTS = [(BENCHMARK, regime) for regime in REGIMES]
    CALIBRATION_METRICS = ["ece", "brier"]
    STRATA = ["3x0", "2x1"]

    def __init__(
        self,
        results_dir: Path,
        annotation_dates: Dict[str, str],
        benchmark_xlsx: Optional[Path] = None,
        conf: float = 0.95,
    ):
        self.results_dir = Path(results_dir)
        self.annotation_dates = annotation_dates  # dataset → pasta da anotação (ou "latest")
        self.benchmark_xlsx = Path(benchmark_xlsx) if benchmark_xlsx else None
        self.comparator = PairedSystemComparator(conf=conf)
        logger.debug(f"RegimeComparison: {annotation_dates}")

    def annotation_dir(self, dataset: str) -> Path:
        date = self.annotation_dates[dataset]
        if date == "latest":
            date = get_latest_results_date(self.results_dir, dataset)
        return self.results_dir / dataset / date

    def locate_runs(self, dataset: str) -> Dict[str, Path]:
        """Última execução (maior vN) concluída, sem seleção de instâncias, de cada regime"""
        runs = {}
        for config_path in (self.annotation_dir(dataset) / "finetuning").glob("v*/config.json"):
            run_dir = config_path.parent
            config = json.loads(config_path.read_text(encoding="utf-8"))
            mode = config.get("training_mode")
            if mode not in self.REGIMES or config.get("instance_selection", {}).get("enabled"):
                continue
            if not any(run_dir.glob("*_fine_tuning_results.json")):
                continue
            if mode not in runs or self._version(run_dir) > self._version(runs[mode]):
                runs[mode] = run_dir

        missing = set(self.REGIMES) - set(runs)
        if missing:
            logger.warning(f"[{dataset}] regimes sem execução concluída: {sorted(missing)}")
        return runs

    @staticmethod
    def _version(run_dir: Path) -> int:
        match = FineTuningRunVersioner.VERSION_PATTERN.match(run_dir.name)
        return int(match.group(1)) if match else -1

    def load_folds(self) -> pd.DataFrame:
        """Formato longo: dataset, system, metric, fold, value (métricas gerais e <métrica>_<estrato>)"""
        rows = []
        for dataset in self.annotation_dates:
            for regime, run_dir in self.locate_runs(dataset).items():
                rows += self._run_rows(dataset, regime, run_dir)
            rows += self._benchmark_rows(dataset)
        return pd.DataFrame(rows)

    def _run_rows(self, dataset: str, regime: str, run_dir: Path) -> list:
        results = json.loads(next(run_dir.glob("*_fine_tuning_results.json")).read_text(encoding="utf-8"))
        rows = []
        for fold, metrics in results["folds"].items():
            for metric in ["f1_macro", "accuracy", *self.CALIBRATION_METRICS]:
                if f"eval_{metric}" in metrics:
                    rows.append(self._row(dataset, regime, f"{metric}", fold, metrics[f"eval_{metric}"]))

        by_stratum = results.get("calibration_by_agreement", {}).get("folds", {})
        for fold, strata in by_stratum.items():
            for stratum in self.STRATA:
                for metric in ["f1_macro", *self.CALIBRATION_METRICS]:
                    if stratum in strata:
                        rows.append(self._row(dataset, regime, f"{metric}_{stratum}", fold, strata[stratum][metric]))
        return rows

    def _benchmark_rows(self, dataset: str) -> list:
        """macro-F1 por fold do RoBERTa com benchmark labels (planilha `effect-fold`: colunas mac0…mac9)"""
        if self.benchmark_xlsx is None:
            return []
        sheet = pd.read_excel(self.benchmark_xlsx, sheet_name="effect-fold")
        match = sheet[(sheet["dataset"] == dataset) & sheet["method"].isna()]
        if match.empty:
            logger.warning(f"[{dataset}] benchmark não encontrado em {self.benchmark_xlsx.name}")
            return []
        row = match.iloc[0]
        return [
            self._row(dataset, self.BENCHMARK, "f1_macro", fold, float(row[f"mac{fold}"]))
            for fold in range(10) if f"mac{fold}" in row and pd.notna(row[f"mac{fold}"])
        ]

    @staticmethod
    def _row(dataset: str, system: str, metric: str, fold, value) -> dict:
        return {"dataset": dataset, "system": system, "metric": metric, "fold": int(fold), "value": float(value)}

    def run(self) -> Dict[str, pd.DataFrame]:
        """Tabelas: folds (longo), individual (IC por sistema) e paired (IC da diferença + t pareado corrigido)"""
        folds = self.load_folds()
        paired = [self.comparator.paired(folds, "f1_macro", self.BENCHMARK_CONTRASTS + self.REGIME_CONTRASTS)]
        for metric in self.CALIBRATION_METRICS:
            for name in [metric, *(f"{metric}_{s}" for s in self.STRATA)]:
                paired.append(self.comparator.paired(folds, name, self.REGIME_CONTRASTS, lower_is_better=True))
        for stratum in self.STRATA:
            paired.append(self.comparator.paired(folds, f"f1_macro_{stratum}", self.REGIME_CONTRASTS))

        return {
            "folds": folds,
            "individual": self.comparator.individual(folds),
            "paired": pd.concat([p for p in paired if not p.empty], ignore_index=True),
        }

    def export(self, tables: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
        """Salva comparacao_regimes.json e um CSV por tabela"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")

        payload = {
            "descricao": "RQ6 — regimes de treino pareados por fold (IC 95% t de Student; t pareado com Bonferroni/BH por métrica)",
            "nivel_confianca": self.comparator.conf,
            "convencao_diff": "system_a − system_b (ECE/Brier: negativo ⇒ system_a melhor)",
            "anotacoes": {ds: self.annotation_dir(ds).name for ds in self.annotation_dates},
            **{name: json.loads(table.to_json(orient="records")) for name, table in tables.items()},
        }
        path = output_dir / "comparacao_regimes.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.success(f"Comparação dos regimes salva em: {output_dir}")
        return path
