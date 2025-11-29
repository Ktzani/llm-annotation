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

from src.config.prompts import BASE_ANNOTATION_PROMPT
from src.experiments.base_experiment import EXPERIMENT_CONFIG
from src.config.datasets_collected import CACHE_DIR


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
        models: List[str],
        categories: List[str],
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
            self.llm_provider,
            self.cache_manager,
            self.response_processor
        )
        
        # Expandir modelos com alternative_params se necessário
        if use_alternative_params:
            self.models = self._expand_models_with_alternatives(models)
            logger.info(f"Alternative params ativado: {len(self.models)} variações")
        
        # Inicializar LLMs
        self.llms = self._initialize_llms()
        
        logger.info(f"LLMAnnotator inicializado")
        logger.info(f"Modelos: {len(self.models)} | Categorias: {len(categories)}")
    
    def _expand_models_with_alternatives(self, models: List[str]) -> List[str]:
        """
        Expande modelos com alternative_params
        
        Args:
            models: Lista de modelos base
            
        Returns:
            Lista expandida com variações
        """
        from llms import LLM_CONFIGS
        
        expanded = []
        
        for model in models:
            if model not in LLM_CONFIGS:
                logger.warning(f"Modelo {model} não encontrado em configs")
                expanded.append(model)
                continue
            
            config = LLM_CONFIGS[model]
            
            # Adicionar modelo base
            expanded.append(model)
            
            # Adicionar variações
            if "alternative_params" in config:
                for idx, alt_params in enumerate(config["alternative_params"]):
                    # Criar nome da variação
                    alt_name = f"{model}_alt{idx+1}"
                    
                    # Criar config temporária
                    alt_config = {
                        "provider": config["provider"],
                        "model_name": config["model_name"],
                        "description": f"{config['description']} (variação {idx+1})",
                        "default_params": alt_params,
                    }
                    
                    # Adicionar à configuração global temporariamente
                    LLM_CONFIGS[alt_name] = alt_config
                    expanded.append(alt_name)
                    
                    logger.debug(f"Criada variação: {alt_name}")
        
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
        prompt_template: str = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None,
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
        return self.annotation_engine.annotate_single(
            text=text,
            model=model,
            llm=self.llms[model],
            num_repetitions=num_repetitions,
            prompt_template=prompt_template,
            examples=examples,
            use_cache=use_cache
        )
    
    def annotate_dataset(
        self,
        texts: List[str],
        num_repetitions: Optional[int] = None,
        prompt_template: str = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None,
        save_intermediate: bool = True
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
                    prompt_template=prompt_template,
                    examples=examples
                )
                
                # Salvar repetições
                for rep_idx, annotation in enumerate(annotations):
                    text_results[f"{model}_rep{rep_idx+1}"] = annotation
                
                # Consenso interno
                most_common = Counter(annotations).most_common(1)[0]
                text_results[f"{model}_consensus"] = most_common[0]
                text_results[f"{model}_consensus_score"] = most_common[1] / len(annotations)
            
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
        
        # Salvar cache
        self.cache_manager.save()
        
        # DataFrame final
        df = pd.DataFrame(results)
        
        # Salvar resultado completo
        output_file = self.results_dir / "annotations_complete.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.success(f"Anotações completas salvas: {output_file}")
        
        return df
    
    def calculate_consensus(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas de consenso
        
        Args:
            df: DataFrame com anotações
            
        Returns:
            DataFrame com métricas de consenso
        """
        logger.info("Calculando consenso...")
        
        # Coletar colunas de consenso
        consensus_cols = [col for col in df.columns 
                         if '_consensus' in col and '_score' not in col]
        
        # Calcular métricas
        df['all_annotations'] = df[consensus_cols].apply(list, axis=1)
        df['unique_annotations'] = df['all_annotations'].apply(lambda x: len(set(x)))
        df['most_common_annotation'] = df['all_annotations'].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else None
        )
        df['most_common_count'] = df['all_annotations'].apply(
            lambda x: Counter(x).most_common(1)[0][1] if x else 0
        )
        df['consensus_score'] = df['most_common_count'] / len(consensus_cols)
        
        # Classificar nível
        def classify_consensus(score):
            if score >= 0.8:
                return 'high'
            elif score >= 0.6:
                return 'medium'
            else:
                return 'low'
        
        df['consensus_level'] = df['consensus_score'].apply(classify_consensus)
        
        # Casos problemáticos
        def check_problematic(annotations):
            counter = Counter(annotations)
            counts = sorted(counter.values(), reverse=True)
            return len(counts) >= 2 and counts[0] == counts[1]
        
        df['is_problematic'] = df['all_annotations'].apply(check_problematic)
        
        # Estatísticas
        high = (df['consensus_level'] == 'high').sum()
        medium = (df['consensus_level'] == 'medium').sum()
        low = (df['consensus_level'] == 'low').sum()
        problematic = df['is_problematic'].sum()
        
        total = len(df)
        logger.success("Consenso calculado:")
        logger.info(f"  Alto (≥80%): {high} ({high/total:.1%})")
        logger.info(f"  Médio (60-80%): {medium} ({medium/total:.1%})")
        logger.info(f"  Baixo (<60%): {low} ({low/total:.1%})")
        logger.info(f"  Problemáticos: {problematic} ({problematic/total:.1%})")
        
        return df
    
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        return self.cache_manager.stats()
