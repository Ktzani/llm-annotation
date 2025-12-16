from datetime import datetime
from pathlib import Path
from loguru import logger

from src.api.core.state import experiments
from src.api.services.prompt_factory import get_prompt_template
from src.api.schemas.experiment import ExperimentRequest

from src.utils.data_loader import load_hf_dataset
from src.llm_annotation_system.annotation.llm_annotator import LLMAnnotator


async def run_experiment_background(
    experiment_id: str,
    config: ExperimentRequest,
):
    try:
        exp = experiments[experiment_id]
        exp.status = "running"
        exp.started_at = datetime.now()
        exp.message = "Carregando dataset..."

        texts, categories, ground_truth = load_hf_dataset(
            config.dataset_name,
            split=config.dataset_config.split,
            combine_splits=config.dataset_config.combine_splits,
            sample_size=config.dataset_config.sample_size,
            random_state=config.dataset_config.random_state,
        )

        exp.progress = 0.3

        prompt = get_prompt_template(
            config.prompt_type,
            config.custom_prompt,
        )

        annotator = LLMAnnotator(
            dataset_name=config.dataset_name,
            categories=categories,
            models=config.models,
            prompt_template=prompt,
            use_langchain_cache=config.cache.enabled,
            use_alternative_params=config.use_alternative_params,
        )

        df_annotations = annotator.annotate_dataset(
            texts=texts,
            num_repetitions=config.num_repetitions_per_llm,
            save_intermediate=config.save_intermediate,
            use_cache=config.cache.enabled,
            model_strategy=config.model_strategy,
            rep_strategy=config.rep_strategy,
        )

        exp.progress = 0.8

        output_dir = config.results_dir or f"results/{experiment_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df_annotations.to_csv(
            f"{output_dir}/annotations.csv",
            index=False,
        )

        exp.status = "completed"
        exp.completed_at = datetime.now()
        exp.progress = 1.0
        exp.message = "Experimento concluído com sucesso"

    except Exception as e:
        logger.exception(e)
        exp.status = "failed"
        exp.completed_at = datetime.now()
        exp.message = str(e)
