"""
Entry point: aplicação do CONSENSO entre LLMs.

Aplica o consenso sobre as anotações (`annotations.csv`) de um experimento,
limpa instâncias inválidas/problemáticas, gera o relatório de concordância
(pairwise / Cohen / Fleiss) e exporta o dataset consolidado em
`<results>/<dataset>/<date>/consensus/dataset_consenso.csv`.

A análise gráfica (calibração, ECE/BBS, heatmaps) permanece no notebook
`src/notebooks/analise_consenso_llms_extra.ipynb` para análise posterior.

As configurações são definidas estaticamente abaixo.
"""
import sys
from loguru import logger

# Garante UTF-8 no console do Windows (evita UnicodeEncodeError com emojis/acentos).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

from src.llm_annotation_system.consensus.pipeline import ConsensusConfig, ConsensusPipeline


def main() -> None:
    # Configuração estática
    dataset_name = "movie_review"
    specific_date = "2026-04-09_13-17-23"
    consensus_threshold = 0.8
    consensus_strategy = "majority_vote"

    config = ConsensusConfig(
        dataset_name=dataset_name,
        specific_date=specific_date,
        consensus_threshold=consensus_threshold,
        consensus_strategy=consensus_strategy,
    )

    result = ConsensusPipeline(config).run()

    df = result["df_with_consensus"]
    report = result["report"]
    logger.success("Consenso aplicado.")
    logger.info(
        f"Registros: {len(df)} | Consenso médio: {df['consensus_score'].mean():.2%} | "
        f"Fleiss' Kappa: {report['fleiss_kappa']:.3f} ({report['fleiss_interpretation']})"
    )


if __name__ == "__main__":
    main()
