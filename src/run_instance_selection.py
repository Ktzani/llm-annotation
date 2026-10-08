"""
Entry point: filtragem por Seleção de Instâncias (biO-IS).

Aplica a técnica biO-IS (waashk/bio-is) para remover instâncias redundantes e
ruidosas do dataset anotado pelas LLMs (coluna de consenso `resolved_annotation`),
gerando um conjunto filtrado para o fine-tuning.

Modos: "single" (uma seleção) ou "sweep" (varia um parâmetro, ex.: beta, e salva a retenção de cada valor
em instance_selection/sweep_<param>.csv; as seleções ficam versionadas e o fine-tuning as reutiliza).

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

from src.systems.instance_selection_system.pipeline import InstanceSelectionConfig, InstanceSelectionPipeline


def run_single(config: InstanceSelectionConfig) -> None:
    """Uma seleção com os params da config"""
    result = InstanceSelectionPipeline(config).run()

    s = result.stats
    logger.success("Filtragem concluída.")
    logger.info(
        f"Mantidas: {s['kept_instances']} | Removidas: {s['removed_instances']} "
        f"(redundantes: {s['removed_redundant']}, ruidosas: {s['removed_noise']}) | "
        f"Redução: {s['reduction_rate']:.2%}"
    )


def run_sweep(config: InstanceSelectionConfig, param: str, values: list) -> None:
    """Varia um parâmetro (demais fixos na config) e mede a retenção sobre o conjunto de treino do fine-tuning"""
    summary = InstanceSelectionPipeline(config).sweep(param, values)
    columns = [param, "kept_instances", "retention", "removed_redundant", "removed_noise"]
    logger.info(f"Retenção por {param}:\n{summary[columns].to_string(index=False)}")


def main() -> None:
    # Configuração estática
    mode = "sweep"  # "single" | "sweep"
    dataset_name = "books"
    specific_date = "latest"
    method = "bio-is"
    params = {"beta": 0.25, "theta": 0.5}

    # Sweep: varia `sweep_param`, mantendo os demais params acima
    sweep_param = "beta"
    sweep_values = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

    config = InstanceSelectionConfig(
        dataset_name=dataset_name,
        specific_date=specific_date,
        method=method,
        params=params,
    )

    if mode == "sweep":
        run_sweep(config, sweep_param, sweep_values)
    else:
        run_single(config)


if __name__ == "__main__":
    main()
