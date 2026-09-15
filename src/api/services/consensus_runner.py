import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.api.core.state import consensus_jobs
from src.api.schemas.consensus.consensus import ConsensusRequest
from src.systems.llm_annotation_system.consensus.pipeline import ConsensusConfig, ConsensusPipeline


def _jsonify(obj: Any) -> Any:
    """Converte DataFrames / tipos numpy do relatório em estruturas JSON-safe."""
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json())
    return obj


async def run_consensus_background(
    job_id: str,
    config: ConsensusRequest,
) -> None:
    """Executa o mesmo fluxo do entry point `src/run_consensus.py`, em background."""
    try:
        consensus_jobs[job_id].status = "running"
        consensus_jobs[job_id].started_at = datetime.now()
        consensus_jobs[job_id].message = "Inicializando pipeline de consenso..."

        logger.info(
            f"[{job_id}] Iniciando consenso — dataset={config.dataset_name}, "
            f"date={config.specific_date}, threshold={config.consensus_threshold}, "
            f"strategy={config.consensus_strategy}"
        )

        consensus_config = ConsensusConfig(
            dataset_name=config.dataset_name,
            results_dir=config.results_dir,
            specific_date=config.specific_date,
            consensus_threshold=config.consensus_threshold,
            consensus_strategy=config.consensus_strategy,
            categories=config.categories,
        )

        consensus_jobs[job_id].progress = 0.1
        consensus_jobs[job_id].message = "Configuração montada. Localizando anotações..."

        pipeline = ConsensusPipeline(consensus_config)

        consensus_jobs[job_id].progress = 0.2
        consensus_jobs[job_id].message = "Pipeline criado. Aplicando consenso..."

        # pipeline.run() é síncrono — rodamos direto (já estamos numa background task)
        result = pipeline.run()

        df = result["df_with_consensus"]
        report = result["report"]
        problematic = report.get("problematic_cases")

        results_path = pipeline.results_dataset_path
        dataset_path = ConsensusPipeline.dataset_path(results_path)

        # O sumário completo já é materializado pelo pipeline — devolvemos junto
        # para evitar uma segunda chamada só para ler o JSON no servidor.
        summary_path = pipeline.summary_dir / "sumario_experimento.json"
        summary = None
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)

        consensus_jobs[job_id].status = "completed"
        consensus_jobs[job_id].completed_at = datetime.now()
        consensus_jobs[job_id].progress = 1.0
        consensus_jobs[job_id].message = "Consenso aplicado com sucesso"
        consensus_jobs[job_id].results = {
            "dataset_name": config.dataset_name,
            "specific_date": results_path.name,
            "consensus_threshold": config.consensus_threshold,
            "consensus_strategy": config.consensus_strategy,
            "models": result["models"],
            "categories": result["categories"],
            "total_records": int(len(df)),
            "consensus_mean": float(df["consensus_score"].mean()),
            "consensus_median": float(df["consensus_score"].median()),
            "invalid_instances": int((df["resolved_annotation"] == -1).sum()),
            "problematic_cases": int(len(problematic)) if problematic is not None else 0,
            "metrics": {
                "fleiss_kappa": float(report["fleiss_kappa"]),
                "fleiss_interpretation": report["fleiss_interpretation"],
                "pairwise_agreement": _jsonify(report["pairwise_agreement"]),
                "cohens_kappa": _jsonify(report["cohens_kappa"]),
            },
            "summary": summary,
            "output": {
                "results_dir": str(results_path),
                "dataset_consenso": str(dataset_path),
                "summary_dir": str(pipeline.summary_dir),
                "consensus_dir": str(Path(results_path) / "consensus"),
            },
        }

        logger.success(
            f"[{job_id}] Consenso concluído — {len(df)} registros | "
            f"Fleiss' Kappa: {report['fleiss_kappa']:.3f}"
        )

    except Exception as e:
        logger.exception(f"[{job_id}] Erro no consenso: {e}")
        consensus_jobs[job_id].status = "failed"
        consensus_jobs[job_id].completed_at = datetime.now()
        consensus_jobs[job_id].message = f"Erro: {str(e)}"
