"""
LLM Annotator - Classe principal refatorada
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter
from loguru import logger
import asyncio
import time

from src.llm_annotation_system.core.llm_provider import LLMProvider
from src.llm_annotation_system.core.cache_manager import CacheManager, LangChainCacheManager
from src.llm_annotation_system.core.response_processor import ResponseProcessor
from src.llm_annotation_system.annotation.annotation_engine import AnnotationEngine
from src.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config.prompts import BASE_ANNOTATION_PROMPT

class LLMAnnotator:
    """
    Classe principal para anotações automáticas
    
    Coordena componentes:
    - LLMProvider: gerencia LLMs
    - CacheManager: gerencia cache
    - ResponseProcessor: processa respostas
    - AnnotationEngine: realiza anotações
    """
    
    def __init__(
        self,
        dataset_name: str,
        models: List[str],
        categories: List[str],
        cache_dir: str,
        results_dir: str,
        prompt_template = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        use_langchain_cache: bool = True,
        use_alternative_params: bool = False
    ):
        """
        Args:
            models: Lista de modelos
            categories: Lista de categorias
            api_keys: Chaves de API
            cache_dir: Diretório de cache
            results_dir: Diretório de resultados
            use_langchain_cache: Se True, usa cache do LangChain
            use_alternative_params: Se True, usa alternative_params dos modelos
        """
        self.models = models
        self.categories = categories
        self.cache_dir = Path(cache_dir)
        
        self.results_dir = Path(results_dir)
        self.results_dir = self.results_dir.joinpath(dataset_name)
        
        self.use_alternative_params = use_alternative_params
        
        # Criar diretórios
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Inicializar componentes
        self.llm_provider = LLMProvider(api_keys)
        self.cache_manager = CacheManager(cache_dir)
        self.langchain_cache = LangChainCacheManager(cache_dir, use_langchain_cache)
        self.response_processor = ResponseProcessor(categories)
        self.annotation_engine = AnnotationEngine(
            llm_provider=self.llm_provider,
            cache_manager=self.cache_manager,
            response_processor=self.response_processor,
            dataset_name=dataset_name,
            prompt_template=prompt_template,
            examples=examples
        )
        
        # Expandir modelos com alternative_params se necessário
        if use_alternative_params:
            self.models = self._expand_models(models)
            logger.info(f"Alternative params ativado: {len(self.models)} variações")
        
        # Inicializar LLMs
        self.llms = self._initialize_llms()
        
        logger.info(f"LLMAnnotator inicializado")
        logger.info(f"Modelos: {len(self.models)} | Categorias: {len(categories)}")
        
    @staticmethod
    def _expand_models(models: list[str]) -> list[str]:
        from src.config.llms import LLM_CONFIGS
        expanded = []

        for model in models:
            if model not in LLM_CONFIGS:
                logger.warning(f"Modelo {model} não encontrado em configs")
                expanded.append(model)
                continue
            
            config = LLM_CONFIGS[model]
            expanded.append(model)

            # Adicionando variações de parametros do modelo
            if "alternative_params" in config:
                for idx, alt in enumerate(config["alternative_params"]):
                    alt_name = f"{model}_alt{idx+1}"

                    LLM_CONFIGS[alt_name] = {
                        "provider": config["provider"],
                        "model_name": config["model_name"],
                        "description": f"{config['description']} (variação {idx+1})",
                        "default_params": alt,
                    }

                    logger.debug(f"Criada variação: {alt_name}")
                    expanded.append(alt_name)

        return expanded
    
    def _initialize_llms(self) -> Dict[str, Any]:
        """Inicializa todas as LLMs"""
        llms = {}
        for model in self.models:
            llms[model] = self.llm_provider.initialize_llm(model)
        return llms
    
    async def annotate_single(
        self,
        text: str,
        model: str,
        num_repetitions: int = 1,
        use_cache: bool = True,
        rep_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    ) -> List[str]:
        """
        Anota um texto único
        
        Args:
            text: Texto para anotar
            model: Modelo a usar
            num_repetitions: Número de repetições
            prompt_template: Template do prompt
            examples: Exemplos para few-shot
            use_cache: Se True, usa cache
            rep_strategy: Rodar as repeticoes dentro de cada modelo sequencialmente ou paralelo
            
        Returns:
            Lista de classificações
        """
        return await self.annotation_engine.annotate(
            text=text,
            model=model,
            llm=self.llms[model],
            num_repetitions=num_repetitions,
            use_cache=use_cache,
            rep_strategy=rep_strategy
        )
    
    async def annotate_dataset(
        self,
        texts: List[str],
        num_repetitions: Optional[int] = None,
        save_intermediate: bool = True,
        intermediate: int = 10,
        use_cache: bool = True,
        model_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        rep_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    ) -> pd.DataFrame:
        """
        Anota dataset completo
        
        Args:
            texts: Lista de textos
            num_repetitions: Número de repetições (usa config se None)
            prompt_template: Template do prompt
            examples: Exemplos para few-shot
            save_intermediate: Se True, salva resultados intermediários
            model_strategy: Rodar os modelos sequencialmente ou paralelo
            rep_strategy: Rodar as repeticoes dentro de cada modelo sequencialmente ou paralelo
            
        Returns:
            DataFrame com anotações
        """
            
        total_annotations = len(texts) * len(self.models) * num_repetitions
        
        logger.info(f"Iniciando anotação")
        logger.info(f"Textos: {len(texts)} | Modelos: {len(self.models)} | Repetições: {num_repetitions}")
        logger.info(f"Total de anotações: {total_annotations}")

        results = []
    
        for idx, text in enumerate(tqdm(texts, desc="Anotando")):
            text_results = {
                "text_id": idx,
                "text": text[:200],
            }
    
            # ===============================
            # 🔁 MODELOS SEQUENCIAL
            # ===============================
            if model_strategy == ExecutionStrategy.SEQUENTIAL:
                for model in self.models:
                    result = await self._annotate_model(
                        text=text,
                        model=model,
                        num_repetitions=num_repetitions,
                        use_cache=use_cache,
                        rep_strategy=rep_strategy
                    )
                    text_results.update(result)
    
            # ===============================
            # 🚀 MODELOS PARALELO
            # ===============================
            else:
                tasks = [
                    self._annotate_model(
                        text=text,
                        model=model,
                        num_repetitions=num_repetitions,
                        use_cache=use_cache,
                        rep_strategy=rep_strategy
                    )
                    for model in self.models
                ]
    
                model_results = await asyncio.gather(*tasks)
    
                for result in model_results:
                    text_results.update(result)
    
            results.append(text_results)

            # Salvar resultados intermediários
            if save_intermediate and (idx + 1) % intermediate == 0:
                pd.DataFrame(results).to_csv(
                    self.results_dir / f"intermediate_{idx+1}.csv",
                    index=False,
                    encoding="utf-8"
                )
                logger.debug(f"Salvos {idx+1} textos")
    
        self.cache_manager.save()
    
        df = pd.DataFrame(results)
    
        return df

    async def _annotate_model(
        self,
        text: str,
        model: str,
        num_repetitions: int,
        use_cache: bool,
        rep_strategy: ExecutionStrategy
    ) -> dict:
        start = time.perf_counter()
        # logger.warning(f"[START] {model} @ {start:.3f}")

        annotations = await self.annotate_single(
            text=text,
            model=model,
            num_repetitions=num_repetitions,
            use_cache=use_cache,
            rep_strategy=rep_strategy
        )
        
        end = time.perf_counter()
        # logger.warning(f"[END]   {model} @ {end:.3f} | Δ={end-start:.2f}s")
        elapsed = end - start

        result = {}

        for rep_idx, annotation in enumerate(annotations):
            result[f"{model}_rep{rep_idx+1}"] = annotation

        most_common = Counter(annotations).most_common(1)[0]
        result[f"{model}_consensus"] = int(most_common[0])
        result[f"{model}_consensus_score"] = float(
            most_common[1] / len(annotations)
        )
        
        result[f"{model}_annotation_time_sec"] = elapsed

        return result
    
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        return self.cache_manager.stats()
    
    
    def evaluate_model_metrics(
        self,
        df: pd.DataFrame,
        ground_truth_col: str = "ground_truth",
        output_csv: bool = False,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Calcula métricas por modelo, considerando -1 como classe de erro válida.
        Não remove as linhas com -1, pois isso faz parte da avaliação.
        """

        logger.info("Calculando métricas por modelo...")

        model_consensus_cols = {
            model: f"{model}_consensus"
            for model in self.models
            if f"{model}_consensus" in df.columns
        }

        if len(model_consensus_cols) == 0:
            logger.error("Nenhuma coluna *_consensus encontrada no DataFrame.")
            return pd.DataFrame()

        df_clean = df.copy()

        for col in model_consensus_cols.values():
            df_clean[col] = df_clean[col].replace(
                {"ERROR": -1, None: -1, "": -1, "N/A": -1}
            )

        df_clean = df_clean[df_clean[ground_truth_col].notna()]

        for col in model_consensus_cols.values():
            df_clean[col] = df_clean[col].astype(int)

        df_clean[ground_truth_col] = df_clean[ground_truth_col].astype(int)

        logger.info(f"Total de linhas avaliadas: {len(df_clean)}")


        # Calcular métricas
        results = []

        for model_name, col in model_consensus_cols.items():

            y_true = df_clean[ground_truth_col]
            y_pred = df_clean[col]

            # Métricas considerando -1 como classe válida
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

            # Coverage: % de predições != -1
            coverage = (y_pred != -1).mean()

            results.append({
                "model": model_name,
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": prec,
                "recall_weighted": rec,
                "coverage": coverage,
                "error_rate": 1 - acc,
                "invalid_predictions_rate": 1 - coverage
            })
            
            logger.info(f"Métricas para {model_name}: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Cov={coverage:.4f}")

        df_metrics = pd.DataFrame(results)
        df_metrics = df_metrics.sort_values("f1_weighted", ascending=False)

        if output_csv:
            if output_dir is None:
                output_dir = self.results_dir
            output_path = output_dir / "model_metrics.csv"
            df_metrics.to_csv(output_path, index=False)
            logger.success(f"Métricas por modelo salvas em: {output_path}")

        logger.success("✓ Métricas calculadas com sucesso")

        return df_metrics
