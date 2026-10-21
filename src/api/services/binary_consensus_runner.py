from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.api.core.state import consensus_jobs
from src.api.schemas.consensus.consensus import ConsensusRequest
from src.api.services.consensus_runner import _jsonify
from src.systems.llm_annotation_system.consensus.pipeline import ConsensusConfig, ConsensusPipeline
from src.systems.llm_annotation_system.consensus.pipeline_binary import BinaryConsensusPipeline


async def run_binary_consensus_background(
    job_id: str,
    config: ConsensusRequest,
) -> None:
    """Mesmo fluxo do run_consensus_background, sobre uma execução da anotação binária (binary/<data>)."""
    try:
        consensus_jobs[job_id].status = "running"
        consensus_jobs[job_id].started_at = datetime.now()
        consensus_jobs[job_id].message = "Inicializando consenso da anotação binária..."

        consensus_config = ConsensusConfig(
            dataset_name=config.dataset_name,
            results_dir=config.results_dir,
            specific_date=config.specific_date,
            consensus_threshold=config.consensus_threshold,
            consensus_strategy=config.consensus_strategy,
            categories=config.categories,
        )

        consensus_jobs[job_id].progress = 0.1
        consensus_jobs[job_id].message = "Configuração montada. Localizando anotações binárias..."

        pipeline = BinaryConsensusPipeline(consensus_config)

        consensus_jobs[job_id].progress = 0.2
        consensus_jobs[job_id].message = "Pipeline criado. Aplicando consenso..."

        result = pipeline.run()

        df = result["df_with_consensus"]
        report = result["report"]
        problematic = report.get("problematic_cases")
        results_path = pipeline.consensus.results_dataset_path

        consensus_jobs[job_id].status = "completed"
        consensus_jobs[job_id].completed_at = datetime.now()
        consensus_jobs[job_id].progress = 1.0
        consensus_jobs[job_id].message = "Consenso da anotação binária aplicado com sucesso"
        consensus_jobs[job_id].results = {
            "dataset_name": config.dataset_name,
            "specific_date": consensus_config.specific_date,
            "consensus_threshold": config.consensus_threshold,
            "consensus_strategy": config.consensus_strategy,
            "models": result["models"],
            "total_items": int(len(df)),
            # resolved_annotation pode vir None (ex.: unanimous_only): conta como inválido
            "invalid_items": int((pd.to_numeric(df["resolved_annotation"], errors="coerce").fillna(-1) == -1).sum()),
            "problematic_cases": int(len(problematic)) if problematic is not None else 0,
            "metrics": {
                "fleiss_kappa": float(report["fleiss_kappa"]),
                "fleiss_interpretation": report["fleiss_interpretation"],
                "pairwise_agreement": _jsonify(report["pairwise_agreement"]),
                "cohens_kappa": _jsonify(report["cohens_kappa"]),
            },
            "output": {
                "results_dir": str(results_path),
                "binary_labels": str(pipeline.labels_path),
                "dataset_consenso": str(ConsensusPipeline.dataset_path(results_path)),
                "summary_dir": str(pipeline.consensus.summary_dir),
                "consensus_dir": str(Path(results_path) / "consensus"),
            },
        }

        logger.success(
            f"[{job_id}] Consenso binário concluído — {len(df)} itens (texto × classe) | "
            f"Fleiss' Kappa: {report['fleiss_kappa']:.3f}"
        )

    except Exception as e:
        logger.exception(f"[{job_id}] Erro no consenso binário: {e}")
        consensus_jobs[job_id].status = "failed"
        consensus_jobs[job_id].completed_at = datetime.now()
        consensus_jobs[job_id].message = f"Erro: {str(e)}"

