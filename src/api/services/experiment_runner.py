from datetime import date, datetime
from pathlib import Path
from loguru import logger

from src.api.core.state import experiments
from src.api.services.prompt_factory import get_prompt_template
from src.api.schemas.experiment import ExperimentRequest

from src.utils.data_loader import load_hf_dataset
from src.llm_annotation_system.annotation.llm_annotator import LLMAnnotator

from src.api.core.state import experiments


async def run_experiment_background(
    experiment_id: str,
    config: ExperimentRequest,
):
    try:
        experiments[experiment_id].status = "running"
        experiments[experiment_id].started_at = datetime.now()
        experiments[experiment_id].message = "Carregando dataset..."

        # 1. Carregar dataset
        texts, categories, ground_truth = load_hf_dataset(
            dataset_name=config.dataset_name,
            cache_dir=config.cache.dir,
            dataset_global_config=config.dataset_config,
        )

        experiments[experiment_id].progress = 0.2
        experiments[experiment_id].message = f"Dataset carregado: {len(texts)} textos"

        # 2. Inicializar annotator
        logger.info(f"[{experiment_id}] Inicializando annotator...")
        prompt_template  = get_prompt_template(
            config.prompt_type,
            config.custom_prompt,
        )

        annotator = LLMAnnotator(
            dataset_name=config.dataset_name,
            models=config.models,
            categories=categories,
            cache_dir=config.cache.dir,
            results_dir=config.results.dir,
            prompt_template=prompt_template,
            use_langchain_cache=config.cache.enabled,
            use_alternative_params=config.annotation.use_alternative_params,
        )
        
        experiments[experiment_id].progress = 0.3
        experiments[experiment_id].message = f"Annotator inicializado com {len(config.models)} modelos"

        # 3. Executar anotação
        logger.info(f"[{experiment_id}] Iniciando anotações...")
        experiments[experiment_id].message = "Executando anotações..."
        
        df_annotations = await annotator.annotate_dataset(
            texts=texts,
            num_repetitions=config.annotation.num_repetitions_per_llm,
            save_intermediate=config.results.save_intermediate,
            intermediate=config.results.intermediate,
            use_cache=config.cache.enabled,
            model_strategy=config.annotation.model_strategy,
            rep_strategy=config.annotation.rep_strategy,
        )
        
        df_annotations["ground_truth"] = ground_truth

        experiments[experiment_id].progress = 0.7
        experiments[experiment_id].message = "Anotações completas. Calculando métricas..."
        
        results = {
            "num_texts": len(texts),
            "num_models": len(config.models),
            "num_repetitions": config.annotation.num_repetitions_per_llm,
            "total_annotations": len(df_annotations),
            "categories": categories,
            "has_ground_truth": ground_truth is not None
        }

        date_path = date.today()
        date_path = date_path.strftime("%Y-%m-%d")

        # 4. Salvar resultados
        output_dir = annotator.results_dir or f"results/{date_path}"
        output_dir = Path(output_dir).joinpath(date_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        df_annotations.to_csv(
            Path(output_dir) / "annotations.csv",
            index=False,
        )
        
        if config.results.save_model_metrics:
            df_metrics = annotator.evaluate_model_metrics(
                df_annotations,
                ground_truth_col="ground_truth",
                output_csv=config.results.save_model_metrics,
                output_dir=output_dir
            )
            results["metrics"] = df_metrics.to_dict(orient="records")


        experiments[experiment_id].status = "completed"
        experiments[experiment_id].completed_at = datetime.now()
        experiments[experiment_id].progress = 1.0
        experiments[experiment_id].message = "Experimento concluído com sucesso"
        experiments[experiment_id].results = results
        
        logger.success(f"[{experiment_id}] Experimento concluído!")

    except Exception as e:
        logger.exception(f"[{experiment_id}] Erro no experimento: {e}")
        experiments[experiment_id].status = "failed"
        experiments[experiment_id].completed_at = datetime.now()
        experiments[experiment_id].message = f"Erro: {str(e)}"
