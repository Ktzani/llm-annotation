"""
🤖 Anotação sem LLM Hacking

Uso:
    # Anotação de um único texto (índice 0 por padrão)
    python annotation_llms.py --mode single

    # Anotação de um único texto específico por índice
    python annotation_llms.py --mode single --index 3

    # Anotação do dataset completo (padrão)
    python annotation_llms.py --mode dataset
    python annotation_llms.py
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from datetime import date

from loguru import logger

from src.api.schemas.experiment import ExperimentRequest
from src.api.services.prompt_factory import get_prompt_template
from src.utils.data_loader import load_hf_dataset, load_hf_dataset_as_dataframe, list_available_datasets
from src.llm_annotation_system.annotation.llm_annotator import LLMAnnotator


# =============================================================================
# 1) Setup e Configuração
# =============================================================================

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

logger.success("✓ Setup completo")

experiment = "time_experiment_local"

config_path = Path(f"C:\\Users\\gabri\\Documents\\GitHub\\llm-annotation\\src\\experiments\\{experiment}.json")

with open(config_path, "r") as f:
    config_dict = json.load(f)

EXPERIMENT_CONFIG = ExperimentRequest(**config_dict)

# --- Modelos e prompt ---
DEFAULT_MODELS = EXPERIMENT_CONFIG.models
PROMPT_TEMPLATE = get_prompt_template(
    EXPERIMENT_CONFIG.prompt_type,
    EXPERIMENT_CONFIG.custom_prompt,
)

# --- Configurações de anotação ---
annotation_cfg = EXPERIMENT_CONFIG.annotation

num_repetitions = annotation_cfg.num_repetitions_per_llm
use_alternative_params = annotation_cfg.use_alternative_params

model_strategy = annotation_cfg.model_strategy
rep_strategy = annotation_cfg.rep_strategy

# --- Configurações de dataset ---
dataset_cfg = EXPERIMENT_CONFIG.dataset_config

dataset_split = dataset_cfg.split
combine_splits = dataset_cfg.combine_splits
sample_size = dataset_cfg.sample_size
random_state = dataset_cfg.random_state

# --- Configurações de cache ---
cache_cfg = EXPERIMENT_CONFIG.cache

cache_enabled = cache_cfg.enabled
cache_dir = cache_cfg.dir

# --- Resultados ---
results_cfg = EXPERIMENT_CONFIG.results

save_intermediate = results_cfg.save_intermediate
intermediate = results_cfg.intermediate
results_dir = results_cfg.dir


# =============================================================================
# 2) Carregar Dataset
# =============================================================================

logger.info("Datasets disponíveis:")
for dataset in list_available_datasets():
    logger.info(f"  - {dataset}")

dataset_name = "sst2"  # Ajuste conforme necessário

texts, categories, ground_truth = load_hf_dataset(
    dataset_name=dataset_name,
    cache_dir=cache_dir,
    dataset_global_config=dataset_cfg
)

logger.info(f"Textos: {len(texts)}")
logger.info(f"Categorias: {categories}")
logger.info(f"Ground truth: {'Sim' if ground_truth else 'Não'}")

# Visualizar amostra
logger.info("Amostra dos textos:")
for i, text in enumerate(texts[:3]):
    logger.info(f"{i+1}. {text[:100]}...")
    if ground_truth:
        logger.info(f"   Label: {ground_truth[i]}")

# =============================================================================
# 3) Configurar Modelos LLM
# =============================================================================

annotator = LLMAnnotator(
    dataset_name=dataset_name,
    models=DEFAULT_MODELS,
    categories=categories,
    cache_dir=cache_dir,
    results_dir=results_dir,
    prompt_template=PROMPT_TEMPLATE,
    use_langchain_cache=cache_enabled,
    use_alternative_params=use_alternative_params
)

logger.success(f"✓ Annotator inicializado com {len(annotator.models)} modelos")


# =============================================================================
# 4) Executar Anotação
# =============================================================================

async def run_single(index: int = 0):
    """Anota um único texto do dataset pelo índice."""
    if index >= len(texts):
        logger.error(f"Índice {index} fora do range. Dataset tem {len(texts)} textos.")
        return

    text = texts[index]
    model = DEFAULT_MODELS[2] if DEFAULT_MODELS else "deepseek-r1-8b"

    logger.warning(f"  Modo: anotação única")
    logger.warning(f"  Índice: {index}")
    logger.warning(f"  Modelo: {model}")
    logger.warning(f"  Texto ({len(text)} chars): {text[:100]}...")
    logger.warning(f"  Repetições: {3}")

    annotations = await annotator.annotate_single(
        text=text,
        model=model,
        num_repetitions=3,
        use_cache=cache_enabled,
        rep_strategy=rep_strategy
    )

    logger.success("✓ Anotação única completa")
    logger.info(f"Resultado: {annotations}")

    if ground_truth:
        logger.info(f"Ground truth: {ground_truth[index]}")


async def run_dataset():
    """Anota o dataset completo e salva os resultados."""
    logger.info("  Modo: anotação do dataset completo")

    df_annotations = await annotator.annotate_dataset(
        texts=texts,
        num_repetitions=num_repetitions,
        use_cache=cache_enabled,
        save_intermediate=save_intermediate,
        intermediate=intermediate,
        model_strategy=model_strategy,
        rep_strategy=rep_strategy
    )

    df_annotations["ground_truth"] = ground_truth if ground_truth else None

    logger.success("✓ Anotações completas")

    # -------------------------------------------------------------------------
    # 5) Salvando resultados
    # -------------------------------------------------------------------------
    date_path = date.today().strftime("%Y-%m-%d")
    experiment_id = f"{date_path}"

    output_dir = annotator.results_dir or f"results/{experiment_id}"
    output_dir = Path(output_dir).joinpath(experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Anotações ---
    annotations_path = Path(output_dir) / "annotations.csv"
    df_annotations.to_csv(annotations_path, index=False)
    logger.success(f"✓ Anotações salvas em: {annotations_path}")

    # --- Métricas ---
    df_metrics = annotator.evaluate_model_metrics(
        df_annotations,
        ground_truth_col="ground_truth",
        output_csv=True,
        output_dir=output_dir
    )
    logger.success(f"✓ Métricas salvas em: {output_dir}")


# =============================================================================
# Controle de execução — altere aqui para debug
# =============================================================================
DEBUG_SINGLE = False  # True = anota apenas um texto; False = anota o dataset completo

async def main():
    if DEBUG_SINGLE:
        await run_single(index=0)
    else:
        await run_dataset()


if __name__ == "__main__":
    asyncio.run(main())
