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
    BASE_BETA, BASE_THETA = 0.25, 0.5  # parâmetros da execução biO-IS principal
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

    def locate_curve_points(self, dataset: str) -> Dict[str, Path]:
        """Pontos concluídos da curva de biO-IS (subpastas <param><valor> da última execução de curva), como aggregated_bio-is_<param><valor>"""
        points = {}
        for results_path in (self.annotation_dir(dataset) / "finetuning").glob("v*/*/*_fine_tuning_results.json"):
            point_dir, run_dir = results_path.parent, results_path.parent.parent
            if not self._is_curve(run_dir):
                continue
            system = f"{AGGREGATED_IS}_{point_dir.name}"
            if system not in points or self._version(run_dir) > self._version(points[system].parent):
                points[system] = point_dir
        return points

    @staticmethod
    def _is_curve(run_dir: Path) -> bool:
        config_path = run_dir / "config.json"
        if config_path.exists():
            return bool(json.loads(config_path.read_text(encoding="utf-8")).get("instance_selection", {}).get("sweep"))
        return "curva" in run_dir.name

    def load_folds(self) -> pd.DataFrame:
        """Folds dos sistemas principais, do benchmark e dos pontos da curva"""
        rows = [super().load_folds()]
        for dataset in self.annotation_dates:
            for system, point_dir in self.locate_curve_points(dataset).items():
                rows.append(pd.DataFrame(self._run_rows(dataset, system, point_dir)))
        return pd.concat(rows, ignore_index=True)

    def curve(self, individual: pd.DataFrame) -> pd.DataFrame:
        """Curva % mantido × macro-F1: sem filtro (β=0), execução biO-IS principal e pontos da curva"""
        rows = []
        for dataset in self.annotation_dates:
            is_dir = self.annotation_dir(dataset) / "instance_selection"
            points = {AGGREGATED: (0.0, 1.0), AGGREGATED_IS: (self.BASE_BETA, self._retention(is_dir, self.BASE_BETA))}
            for system in self.locate_curve_points(dataset):
                beta = float(system.rsplit("beta", 1)[1])
                points[system] = (beta, self._retention(is_dir, beta))
            for system, (beta, retention) in points.items():
                f1 = individual[(individual.dataset == dataset) & (individual.system == system) & (individual.metric == "f1_macro")]
                if f1.empty:
                    continue
                r = f1.iloc[0]
                rows.append({"dataset": dataset, "system": system, "beta": beta, "retencao": retention,
                             "f1_macro": r["mean"], "ci_low": r.ci_low, "ci_high": r.ci_high, "n_folds": r.n_folds})
        return pd.DataFrame(rows).sort_values(["dataset", "beta"]).reset_index(drop=True)

    @staticmethod
    def _retention(is_dir: Path, beta: float) -> Optional[float]:
        """Fração mantida pela seleção (kept / rotuladas), da seleção versionada ou do relatório legado (β=0,25)"""
        candidates = [is_dir / f"bio-is_beta{beta:g}_theta{SupervisionComparison.BASE_THETA:g}_seed42" / "instance_selection_report.json"]
        if beta == SupervisionComparison.BASE_BETA:
            candidates.append(is_dir / "instance_selection_report.json")
        for path in candidates:
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
                return report["kept_instances"] / report["labeled_instances"]
        return None

    @staticmethod
    def _system(run_dir: Path) -> Optional[str]:
        """aggregated / aggregated_bio-is a partir do config.json ou, nas execuções antigas sem ele, do nome da pasta"""
        mode, with_is, curve = RegimeComparison.run_info(run_dir)
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

        # Curva: cada ponto contra o sem filtro; família = pontos × datasets
        curve_systems = sorted({s for s in folds["system"].unique() if s.startswith(f"{AGGREGATED_IS}_")})
        for metric in ["f1_macro", "accuracy"]:
            table = self.comparator.paired(folds, metric, [(s, AGGREGATED) for s in curve_systems])
            if not table.empty:
                paired.append(table.assign(family="rq2_curva_vs_sem_is"))

        individual = self.comparator.individual(folds)
        return {
            "folds": folds,
            "individual": individual,
            "paired": pd.concat(paired, ignore_index=True),
            "curve": self.curve(individual),
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
