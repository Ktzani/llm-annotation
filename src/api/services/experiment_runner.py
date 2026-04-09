from datetime import date, datetime
from pathlib import Path
from loguru import logger

from src.api.core.state import experiments
from src.api.services.prompt_factory import get_prompt_template
from src.api.schemas.experiment import ExperimentRequest

from src.utils.data_loader import load_hf_dataset
from src.llm_annotation_system.annotation.llm_annotator import LLMAnnotator

from src.api.core.state import experiments

import pandas as pd

from src.llm_annotation_system.core.evaluate_model_metrics import evaluate_model_metrics
from src.utils.get_text_id_from_text import get_text_id_from_text


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

        if config.dataset_config.remove_texts:
            logger.warning(f"Removendo textos já anotados...")
            logger.info(f"Textos antes: {len(texts)}")
            df = pd.read_csv(config.dataset_config.remove_texts.annotated_texts_path)

            annotated_texts = set(df["text"])

            texts = [t for t in texts if t not in annotated_texts]

            logger.info(f"Textos anotados encontrados: {len(annotated_texts)}")
            logger.info(f"Textos após remoção: {len(texts)}")

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
            use_cache=config.cache.enabled,
            use_alternative_params=config.annotation.use_alternative_params,
            keep_alive=config.annotation.keep_alive
        )
        
        experiments[experiment_id].progress = 0.3
        experiments[experiment_id].message = f"Annotator inicializado com {len(config.models)} modelos"

        # 3. Executar anotação
        logger.info(f"[{experiment_id}] Iniciando anotações...")
        experiments[experiment_id].message = "Executando anotações..."
        
        df_annotations = await annotator.annotate_dataset(
            texts=texts,
            num_repetitions=config.annotation.num_repetitions_per_llm,
            intermediate=config.results.intermediate,
            model_strategy=config.annotation.model_strategy,
            rep_strategy=config.annotation.rep_strategy,
            max_concurrent_texts=config.annotation.max_concurrent_texts
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

        date_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 4. Salvar resultados
        output_dir = annotator.results_dir or f"results/{date_path}"
        output_dir = Path(output_dir).joinpath(date_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        df_annotations.to_csv(
            Path(output_dir) / "annotations.csv",
            index=False,
        )

        df_metrics = evaluate_model_metrics(
            df_annotations,
            models=annotator.models,
            ground_truth_col="ground_truth",
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
