"""
🤖 Anotação sem LLM Hacking

"""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from loguru import logger

from src.api.schemas.experiment import ExperimentRequest
from src.api.services.prompt_factory import get_prompt_template
from src.utils.data_loader import load_hf_dataset, list_available_datasets
from src.llm_annotation_system.annotation.llm_annotator import LLMAnnotator
from src.llm_annotation_system.core.evaluate_model_metrics import evaluate_model_metrics
from src.utils.get_text_id_from_text import get_text_id_from_text


# =============================================================================
# 1) Setup e Configuração
# =============================================================================

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


def load_experiment_config(config_path: Path) -> tuple:
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    EXPERIMENT_CONFIG = ExperimentRequest(**config_dict)

    DEFAULT_MODELS = EXPERIMENT_CONFIG.models
    PROMPT_TEMPLATE = get_prompt_template(
        EXPERIMENT_CONFIG.prompt_type,
        EXPERIMENT_CONFIG.custom_prompt,
    )

    annotation_cfg = EXPERIMENT_CONFIG.annotation
    dataset_cfg = EXPERIMENT_CONFIG.dataset_config
    cache_cfg = EXPERIMENT_CONFIG.cache
    results_cfg = EXPERIMENT_CONFIG.results

    return DEFAULT_MODELS, PROMPT_TEMPLATE, annotation_cfg, dataset_cfg, cache_cfg, results_cfg


def load_texts(dataset_name: str, cache_dir: str, dataset_cfg, remove_annotated_texts: bool, annotated_texts_path: str) -> tuple:
    texts, categories, ground_truth = load_hf_dataset(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        dataset_global_config=dataset_cfg
    )

    if remove_annotated_texts:
        logger.warning(f"Removendo textos já anotados...")
        logger.info(f"Textos antes: {len(texts)}")
        
        df = pd.read_csv(annotated_texts_path)
        annotated_texts = set(df["text"])
        
        texts = [t for t in texts if t not in annotated_texts]
        
        logger.info(f"Textos anotados encontrados: {len(annotated_texts)}")
        logger.info(f"Textos após remoção: {len(texts)}")

    return texts, categories, ground_truth


async def run_single(
    annotator: LLMAnnotator, 
    texts: list, 
    ground_truth: list, 
    models: list, 
    rep_strategy: str, 
    index: int = 0
):
    """Anota um único texto do dataset pelo índice."""
    if index >= len(texts):
        logger.error(f"Índice {index} fora do range. Dataset tem {len(texts)} textos.")
        return

    text = texts[index]
    model = models[2] if models else "deepseek-r1-8b"

    logger.warning(f"  Modo: anotação única")
    logger.warning(f"  Índice: {index}")
    logger.warning(f"  Modelo: {model}")
    logger.warning(f"  Texto ({len(text)} chars): {text[:100]}...")
    logger.warning(f"  Repetições: {3}")

    annotations = await annotator.annotate_single(
        text=text,
        model=model,
        num_repetitions=3,
        rep_strategy=rep_strategy
    )

    logger.success("✓ Anotação única completa")
    logger.info(f"Resultado: {annotations}")

    if ground_truth:
        logger.info(f"Ground truth: {ground_truth[index]}")


async def run_dataset(
    annotator: LLMAnnotator, 
    texts: list, 
    ground_truth: list, 
    num_repetitions: int, 
    intermediate: str, 
    model_strategy: str, 
    rep_strategy: str, 
    max_concurrent_texts: int
):
    """Anota o dataset completo e salva os resultados."""
    logger.info("  Modo: anotação do dataset completo")

    df_annotations = await annotator.annotate_dataset(
        texts=texts,
        num_repetitions=num_repetitions,
        intermediate=intermediate,
        model_strategy=model_strategy,
        rep_strategy=rep_strategy,
        max_concurrent_texts=max_concurrent_texts
    )
    
    # Remover duplicatas de texto, mantendo a primeira ocorrência
    df_annotations = df_annotations.drop_duplicates(subset=["text"])
    
    if texts:
        df_gt = pd.DataFrame({
            "text": texts,
            "ground_truth": ground_truth
        })

        df_gt["text_id"] = df_gt["text"].apply(get_text_id_from_text)

        df_annotations = df_annotations.merge(
            df_gt[["text_id", "ground_truth"]],
            on="text_id",
            how="left"
        )

    logger.success("✓ Anotações completas")

    date_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = annotator.results_dir or f"results/{date_path}"
    output_dir = Path(output_dir).joinpath(date_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_annotations.to_csv(
        Path(output_dir) / "annotations.csv",
        index=False,
    )
    logger.success(f"✓ Anotações salvas em: {output_dir}")

    evaluate_model_metrics(
        df_annotations,
        models=annotator.models,
        ground_truth_col="ground_truth",
        output_dir=output_dir
    )
    logger.success(f"✓ Métricas salvas em: {output_dir}")


# =============================================================================
# Main
# =============================================================================

async def main():
    # =========================================================================
    # 🔧 CONFIGURE AQUI
    # =========================================================================
    DEBUG_SINGLE = False  # True = anota apenas um texto; False = anota o dataset completo

    experiment = "large_experiment"
    config_path = Path(f"C:\\Users\\gabri\\Documents\\GitHub\\llm-annotation\\src\\experiments\\{experiment}.json")

    dataset_name = "dblp"  # Exemplo: "agnews", "yelp_review_full", "imdb", "amazon_polarity", etc.
    annotated_texts_path = rf"data\results\{dataset_name}\intermediate.csv"

    logger.success("✓ Setup completo")

    DEFAULT_MODELS, PROMPT_TEMPLATE, annotation_cfg, dataset_cfg, cache_cfg, results_cfg = load_experiment_config(config_path)

    num_repetitions = annotation_cfg.num_repetitions_per_llm
    use_alternative_params = annotation_cfg.use_alternative_params
    model_strategy = annotation_cfg.model_strategy
    rep_strategy = annotation_cfg.rep_strategy
    max_concurrent_texts = annotation_cfg.max_concurrent_texts
    keep_alive = annotation_cfg.keep_alive

    cache_enabled = cache_cfg.enabled
    cache_dir = cache_cfg.dir

    intermediate = results_cfg.intermediate
    results_dir = results_cfg.dir

    # -------------------------------------------------------------------------
    # Carregar Dataset
    # -------------------------------------------------------------------------
    logger.info("Datasets disponíveis:")
    for dataset in list_available_datasets():
        logger.info(f"  - {dataset}")

    texts, categories, ground_truth = load_texts(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        dataset_cfg=dataset_cfg,
        remove_annotated_texts=dataset_cfg.remove_texts,
        annotated_texts_path=annotated_texts_path
    )
    

    logger.info(f"Textos: {len(texts)}")
    logger.info(f"Categorias: {categories}")
    logger.info(f"Ground truth: {'Sim' if ground_truth else 'Não'}")

    logger.info("Amostra dos textos:")
    for i, text in enumerate(texts[:3]):
        logger.info(f"{i+1}. {text[:100]}...")
        if ground_truth:
            logger.info(f"   Label: {ground_truth[i]}")

    # -------------------------------------------------------------------------
    # Configurar Annotator
    # -------------------------------------------------------------------------
    annotator = LLMAnnotator(
        dataset_name=dataset_name,
        models=DEFAULT_MODELS,
        categories=categories,
        cache_dir=cache_dir,
        results_dir=results_dir,
        prompt_template=PROMPT_TEMPLATE,
        use_cache=cache_enabled,
        use_alternative_params=use_alternative_params,
        keep_alive=keep_alive
    )

    logger.success(f"✓ Annotator inicializado com {len(annotator.models)} modelos")

    # -------------------------------------------------------------------------
    # Executar
    # -------------------------------------------------------------------------
    if DEBUG_SINGLE:
        await run_single(
            annotator=annotator,
            texts=texts,
            ground_truth=ground_truth,
            models=DEFAULT_MODELS,
            rep_strategy=rep_strategy,
            index=0
        )
    else:
        await run_dataset(
            annotator=annotator,
            texts=texts,
            ground_truth=ground_truth,
            num_repetitions=num_repetitions,
            intermediate=intermediate,
            model_strategy=model_strategy,
            rep_strategy=rep_strategy,
            max_concurrent_texts=max_concurrent_texts
        )


if __name__ == "__main__":
    asyncio.run(main())