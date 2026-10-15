"""Supervision Comparison - Compara por fold a supervisão consolidada (com e sem biO-IS) entre si e com o benchmark (RQ1/RQ2)"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.systems.fine_tune_system.evaluation.regime_comparison import RegimeComparison

AGGREGATED = "aggregated"
AGGREGATED_IS = "aggregated_bio-is"
BENCHMARK = "benchmark"


class SupervisionComparison(RegimeComparison):
    """
    Reúne as execuções do voto majoritário (sem e com seleção de instâncias) e as compara pareadas por fold
    Responsabilidades: localizar a última execução de cada sistema (com ou sem config.json) e aplicar uma correção de múltiplas comparações por família de contraste
    """

    REGIMES = [AGGREGATED, AGGREGATED_IS]
    # Cada família é corrigida separadamente (Bonferroni/BH entre datasets), diff = A − B
    FAMILIES = {
        "rq1_benchmark_vs_consolidado": ([(BENCHMARK, AGGREGATED)], ["f1_macro"]),
        "rq2_com_is_vs_sem_is": ([(AGGREGATED_IS, AGGREGATED)], ["f1_macro", "accuracy"]),
        "rq2_benchmark_vs_com_is": ([(BENCHMARK, AGGREGATED_IS)], ["f1_macro"]),
    }

    def __init__(self, results_dir: Path, annotation_dates: Dict[str, str], benchmark_xlsx: Optional[Path] = None, conf: float = 0.95):
        super().__init__(results_dir, annotation_dates, benchmark_xlsx, conf)
        logger.debug(f"SupervisionComparison: {annotation_dates}")

    def locate_runs(self, dataset: str) -> Dict[str, Path]:
        """Última execução concluída do voto majoritário, sem e com biO-IS"""
        runs = {}
        for results_path in (self.annotation_dir(dataset) / "finetuning").glob("v*/*_fine_tuning_results.json"):
            run_dir = results_path.parent
            system = self._system(run_dir)
            if system is None:
                continue
            if system not in runs or self._version(run_dir) > self._version(runs[system]):
                runs[system] = run_dir

        missing = set(self.REGIMES) - set(runs)
        if missing:
            logger.warning(f"[{dataset}] sistemas sem execução concluída: {sorted(missing)}")
        return runs

    @staticmethod
    def _system(run_dir: Path) -> Optional[str]:
        """aggregated / aggregated_bio-is a partir do config.json ou, nas execuções antigas sem ele, do nome da pasta"""
        config_path = run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            mode = config.get("training_mode", AGGREGATED)
            with_is = bool(config.get("instance_selection", {}).get("enabled"))
            curve = bool(config.get("instance_selection", {}).get("sweep"))
        else:
            name = run_dir.name
            mode = next((m for m in ["aggregated_replicated", "perspectivism", "soft_labels"] if m in name), AGGREGATED)
            with_is, curve = "bio-is" in name, "curva" in name
        if mode != AGGREGATED or curve:
            return None
        return AGGREGATED_IS if with_is else AGGREGATED

    def run(self) -> Dict[str, pd.DataFrame]:
        """Tabelas: folds (longo), individual (IC por sistema) e paired (IC da diferença + t pareado corrigido por família)"""
        folds = self.load_folds()
        paired = []
        for family, (contrasts, metrics) in self.FAMILIES.items():
            for metric in metrics:
                table = self.comparator.paired(folds, metric, contrasts)
                if not table.empty:
                    paired.append(table.assign(family=family))

        return {
            "folds": folds,
            "individual": self.comparator.individual(folds),
            "paired": pd.concat(paired, ignore_index=True),
        }

    def export(self, tables: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
        """Salva comparacao_supervisao.json e um CSV por tabela"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")

        payload = {
            "descricao": "RQ1/RQ2 — voto majoritário sem e com biO-IS × benchmark, pareado por fold (IC 95% t de Student; t pareado com Bonferroni/BH por família)",
            "nivel_confianca": self.comparator.conf,
            "convencao_diff": "system_a − system_b",
            "anotacoes": {ds: self.annotation_dir(ds).name for ds in self.annotation_dates},
            "execucoes": {ds: {s: p.name for s, p in self.locate_runs(ds).items()} for ds in self.annotation_dates},
            **{name: json.loads(table.to_json(orient="records")) for name, table in tables.items()},
        }
        path = output_dir / "comparacao_supervisao.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.success(f"Comparação da supervisão salva em: {output_dir}")
        return path
