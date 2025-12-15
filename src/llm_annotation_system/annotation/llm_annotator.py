"""
LLM Annotator - Classe principal refatorada
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter
from loguru import logger

from src.llm_annotation_system.core.llm_provider import LLMProvider
from src.llm_annotation_system.core.cache_manager import CacheManager, LangChainCacheManager
from src.llm_annotation_system.core.response_processor import ResponseProcessor
from src.llm_annotation_system.annotation.annotation_engine import AnnotationEngine

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config.prompts import BASE_ANNOTATION_PROMPT
from src.experiments.base_experiment import EXPERIMENT_CONFIG

if EXPERIMENT_CONFIG["cache"].get("enabled", False):
    CACHE_DIR = EXPERIMENT_CONFIG["cache"].get("dir", "../../data/.cache")

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
        prompt_template = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        cache_dir: str = CACHE_DIR,
        results_dir: str = "../../results",
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

            # Adicionando variações
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
    
    def annotate_single(
        self,
        text: str,
        model: str,
        num_repetitions: int = 1,
        use_cache: bool = True
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
            
        Returns:
            Lista de classificações
        """
        return self.annotation_engine.annotate(
            text=text,
            model=model,
            llm=self.llms[model],
            num_repetitions=num_repetitions,
            use_cache=use_cache
        )
    
    def annotate_dataset(
        self,
        texts: List[str],
        num_repetitions: Optional[int] = None,
        save_intermediate: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Anota dataset completo
        
        Args:
            texts: Lista de textos
            num_repetitions: Número de repetições (usa config se None)
            prompt_template: Template do prompt
            examples: Exemplos para few-shot
            save_intermediate: Se True, salva resultados intermediários
            
        Returns:
            DataFrame com anotações
        """
        if num_repetitions is None:
            num_repetitions = EXPERIMENT_CONFIG.get("num_repetitions_per_llm", 3)
        
        total_annotations = len(texts) * len(self.models) * num_repetitions
        
        logger.info(f"Iniciando anotação")
        logger.info(f"Textos: {len(texts)} | Modelos: {len(self.models)} | Repetições: {num_repetitions}")
        logger.info(f"Total de anotações: {total_annotations}")
        
        results = []
        
        for idx, text in enumerate(tqdm(texts, desc="Anotando")):
            text_results = {
                'text_id': idx,
                'text': text[:200],
            }
            
            # Anotar com cada modelo
            for model in self.models:
                annotations = self.annotate_single(
                    text=text,
                    model=model,
                    num_repetitions=num_repetitions,
                    use_cache=use_cache
                )
                
                # Salvar repetições
                for rep_idx, annotation in enumerate(annotations):
                    text_results[f"{model}_rep{rep_idx+1}"] = annotation
                
                # Consenso interno
                most_common = Counter(annotations).most_common(1)[0]
                text_results[f"{model}_consensus"] = int(most_common[0])
                text_results[f"{model}_consensus_score"] = float(most_common[1] / len(annotations))
            
            results.append(text_results)
            
            # Salvar intermediários
            if save_intermediate and (idx + 1) % 10 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(
                    self.results_dir / f"intermediate_{idx+1}.csv",
                    index=False,
                    encoding='utf-8'
                )
                logger.debug(f"Salvos {idx+1} textos")
        
        self.cache_manager.save()
        
        df = pd.DataFrame(results)
        
        output_file = self.results_dir / "annotations_complete.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.success(f"Anotações completas salvas: {output_file}")
        
        return df
    
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        return self.cache_manager.stats()
    
    
    def evaluate_model_metrics(
        self,
        df: pd.DataFrame,
        ground_truth_col: str = "ground_truth",
        output_csv: bool = False
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
            output_path = self.results_dir / "model_metrics.csv"
            df_metrics.to_csv(output_path, index=False)
            logger.success(f"Métricas por modelo salvas em: {output_path}")

        logger.success("✓ Métricas calculadas com sucesso")

        return df_metrics
