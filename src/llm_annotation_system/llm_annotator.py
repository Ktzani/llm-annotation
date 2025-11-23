"""
LLM Annotator - Classe principal para gerenciar anotações automáticas
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

# Import LLM providers
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from config import (
    BASE_ANNOTATION_PROMPT,
    FEW_SHOT_PROMPT,
    COT_PROMPT,
    LLM_CONFIGS,
    EXPERIMENT_CONFIG,
)


class LLMAnnotator:
    """
    Classe principal para realizar anotações automáticas usando múltiplas LLMs
    """
    
    def __init__(
        self,
        models: List[str],
        categories: List[str],
        api_keys: Dict[str, str],
        cache_dir: str = "./cache",
        results_dir: str = "./results"
    ):
        """
        Inicializa o anotador com múltiplas LLMs
        
        Args:
            models: Lista de nomes dos modelos a usar
            categories: Lista de categorias para classificação
            api_keys: Dicionário com as chaves de API
            cache_dir: Diretório para cache de respostas
            results_dir: Diretório para salvar resultados
        """
        self.models = models
        self.categories = categories
        self.api_keys = api_keys
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path(results_dir)
        
        # Criar diretórios
        self.cache_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Inicializar clientes LLM
        self.clients = self._initialize_clients()
        
        # Cache de respostas
        self.response_cache = self._load_cache()
        
        print(f"✓ LLMAnnotator inicializado com {len(models)} modelos")
        print(f"✓ Modelos: {', '.join(models)}")
        print(f"✓ Categorias: {', '.join(categories)}")
    
    def _initialize_clients(self) -> Dict[str, Any]:
        """Inicializa os clientes das APIs das LLMs"""
        clients = {}
        
        for model in self.models:
            if model not in LLM_CONFIGS:
                raise ValueError(f"Modelo {model} não configurado em config.py")
            
            provider = LLM_CONFIGS[model]["provider"]
            
            if provider == "openai":
                if OpenAI is None:
                    raise ImportError("openai não instalado. Execute: pip install openai")
                if "openai" not in self.api_keys:
                    raise ValueError("API key da OpenAI não fornecida")
                clients[model] = OpenAI(api_key=self.api_keys["openai"])
                
            elif provider == "anthropic":
                if Anthropic is None:
                    raise ImportError("anthropic não instalado. Execute: pip install anthropic")
                if "anthropic" not in self.api_keys:
                    raise ValueError("API key da Anthropic não fornecida")
                clients[model] = Anthropic(api_key=self.api_keys["anthropic"])
                
            elif provider == "google":
                if genai is None:
                    raise ImportError("google-generativeai não instalado")
                if "google" not in self.api_keys:
                    raise ValueError("API key do Google não fornecida")
                genai.configure(api_key=self.api_keys["google"])
                clients[model] = genai
        
        return clients
    
    def _load_cache(self) -> Dict[str, Any]:
        """Carrega cache de respostas anteriores"""
        cache_file = self.cache_dir / "response_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Salva cache de respostas"""
        cache_file = self.cache_dir / "response_cache.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.response_cache, f, indent=2, ensure_ascii=False)
    
    def _get_cache_key(self, model: str, text: str, params: Dict) -> str:
        """Gera chave única para cache"""
        content = f"{model}|{text}|{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _format_prompt(
        self,
        text: str,
        prompt_template: str = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None
    ) -> str:
        """
        Formata o prompt para a LLM
        
        Args:
            text: Texto a ser classificado
            prompt_template: Template do prompt a usar
            examples: Exemplos para few-shot learning (opcional)
        
        Returns:
            Prompt formatado
        """
        categories_str = "\n".join([f"- {cat}" for cat in self.categories])
        
        if examples and prompt_template == FEW_SHOT_PROMPT:
            examples_str = "\n\n".join([
                f"Text: {ex['text']}\nCategory: {ex['category']}"
                for ex in examples
            ])
            return prompt_template.format(
                examples=examples_str,
                text=text,
                categories=categories_str
            )
        
        return prompt_template.format(
            text=text,
            categories=categories_str
        )
    
    def _call_llm(
        self,
        model: str,
        prompt: str,
        params: Optional[Dict] = None
    ) -> str:
        """
        Chama a LLM e retorna a resposta
        
        Args:
            model: Nome do modelo
            prompt: Prompt formatado
            params: Parâmetros customizados (opcional)
        
        Returns:
            Resposta da LLM
        """
        # Usar parâmetros default se não fornecidos
        if params is None:
            params = LLM_CONFIGS[model]["default_params"].copy()
        
        # Verificar cache
        if EXPERIMENT_CONFIG["use_cache"]:
            cache_key = self._get_cache_key(model, prompt, params)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
        
        provider = LLM_CONFIGS[model]["provider"]
        model_name = LLM_CONFIGS[model]["model_name"]
        client = self.clients[model]
        
        try:
            if provider == "openai":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                result = response.choices[0].message.content.strip()
                
            elif provider == "anthropic":
                response = client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                result = response.content[0].text.strip()
                
            elif provider == "google":
                model_obj = client.GenerativeModel(model_name)
                response = model_obj.generate_content(
                    prompt,
                    generation_config=params
                )
                result = response.text.strip()
            
            # Salvar no cache
            if EXPERIMENT_CONFIG["use_cache"]:
                self.response_cache[cache_key] = result
                if len(self.response_cache) % 10 == 0:  # Salvar a cada 10 respostas
                    self._save_cache()
            
            return result
            
        except Exception as e:
            print(f"Erro ao chamar {model}: {str(e)}")
            return None
    
    def _extract_category(self, response: str) -> str:
        """
        Extrai a categoria da resposta da LLM
        
        Args:
            response: Resposta completa da LLM
        
        Returns:
            Categoria extraída
        """
        if response is None:
            return "ERROR"
        
        response = response.strip()
        
        # Tentar extrair "CLASSIFICATION:" se presente
        if "CLASSIFICATION:" in response:
            response = response.split("CLASSIFICATION:")[-1].strip()
        
        # Remover pontuação e converter para minúsculas para comparação
        response_clean = response.lower().strip('.,!?;:"\'')
        
        # Verificar se a resposta corresponde a alguma categoria
        for category in self.categories:
            if category.lower() == response_clean:
                return category
        
        # Se não encontrou correspondência exata, tentar busca parcial
        for category in self.categories:
            if category.lower() in response_clean or response_clean in category.lower():
                return category
        
        # Retornar a resposta original se não encontrou correspondência
        return response[:50]  # Limitar tamanho
    
    def annotate_single(
        self,
        text: str,
        model: str,
        num_repetitions: int = 1,
        params: Optional[Dict] = None,
        prompt_template: str = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Anota um único texto com uma LLM, com múltiplas repetições
        
        Args:
            text: Texto a ser anotado
            model: Modelo a usar
            num_repetitions: Número de vezes que a LLM deve classificar
            params: Parâmetros customizados
            prompt_template: Template do prompt
            examples: Exemplos para few-shot
        
        Returns:
            Lista de classificações (uma por repetição)
        """
        prompt = self._format_prompt(text, prompt_template, examples)
        classifications = []
        
        for _ in range(num_repetitions):
            response = self._call_llm(model, prompt, params)
            category = self._extract_category(response)
            classifications.append(category)
            
            # Pequeno delay para evitar rate limiting
            time.sleep(0.1)
        
        return classifications
    
    def annotate_dataset(
        self,
        texts: List[str],
        num_repetitions: int = None,
        test_param_variations: bool = False,
        prompt_template: str = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Anota um dataset completo com todas as LLMs configuradas
        
        Args:
            texts: Lista de textos a serem anotados
            num_repetitions: Número de repetições por LLM (usa config se None)
            test_param_variations: Se True, testa variações de parâmetros
            prompt_template: Template do prompt a usar
            examples: Exemplos para few-shot learning
        
        Returns:
            DataFrame com todas as anotações
        """
        if num_repetitions is None:
            num_repetitions = EXPERIMENT_CONFIG["num_repetitions_per_llm"]
        
        results = []
        
        print(f"\n{'='*80}")
        print(f"Iniciando anotação de {len(texts)} textos")
        print(f"Modelos: {len(self.models)} | Repetições por modelo: {num_repetitions}")
        print(f"{'='*80}\n")
        
        for idx, text in enumerate(tqdm(texts, desc="Anotando textos")):
            text_results = {
                'text_id': idx,
                'text': text[:200],  # Primeiros 200 chars para referência
            }
            
            for model in self.models:
                # Anotações com parâmetros default
                annotations = self.annotate_single(
                    text=text,
                    model=model,
                    num_repetitions=num_repetitions,
                    prompt_template=prompt_template,
                    examples=examples
                )
                
                # Salvar cada repetição
                for rep_idx, annotation in enumerate(annotations):
                    text_results[f"{model}_rep{rep_idx+1}"] = annotation
                
                # Salvar consenso interno da LLM
                text_results[f"{model}_consensus"] = Counter(annotations).most_common(1)[0][0]
                text_results[f"{model}_consensus_score"] = Counter(annotations).most_common(1)[0][1] / len(annotations)
                
                # Testar variações de parâmetros se solicitado
                if test_param_variations and "alternative_params" in LLM_CONFIGS[model]:
                    for param_idx, alt_params in enumerate(LLM_CONFIGS[model]["alternative_params"]):
                        alt_annotation = self.annotate_single(
                            text=text,
                            model=model,
                            num_repetitions=1,
                            params=alt_params,
                            prompt_template=prompt_template,
                            examples=examples
                        )[0]
                        text_results[f"{model}_param_var{param_idx+1}"] = alt_annotation
            
            results.append(text_results)
            
            # Salvar resultados intermediários
            if EXPERIMENT_CONFIG["save_intermediate"] and (idx + 1) % 10 == 0:
                pd.DataFrame(results).to_csv(
                    self.results_dir / f"intermediate_results_{idx+1}.csv",
                    index=False,
                    encoding='utf-8'
                )
        
        # Salvar cache final
        self._save_cache()
        
        df = pd.DataFrame(results)
        
        # Salvar resultado final
        output_file = self.results_dir / "annotations_complete.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ Anotações completas salvas em: {output_file}")
        
        return df
    
    def calculate_consensus(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas de consenso para cada instância
        
        Args:
            df: DataFrame com anotações
        
        Returns:
            DataFrame com métricas de consenso adicionadas
        """
        print("\nCalculando métricas de consenso...")
        
        # Coletar todas as anotações por linha
        consensus_cols = [col for col in df.columns if '_consensus' in col and '_score' not in col]
        
        df['all_annotations'] = df[consensus_cols].apply(lambda row: list(row), axis=1)
        df['unique_annotations'] = df['all_annotations'].apply(lambda x: len(set(x)))
        df['most_common_annotation'] = df['all_annotations'].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else None
        )
        df['most_common_count'] = df['all_annotations'].apply(
            lambda x: Counter(x).most_common(1)[0][1] if x else 0
        )
        df['consensus_score'] = df['most_common_count'] / len(consensus_cols)
        
        # Classificar nível de consenso
        def classify_consensus(score):
            if score >= 0.8:
                return 'high'
            elif score >= 0.6:
                return 'medium'
            else:
                return 'low'
        
        df['consensus_level'] = df['consensus_score'].apply(classify_consensus)
        
        # Casos problemáticos (empate 2-2-1, etc)
        def check_problematic(annotations):
            counter = Counter(annotations)
            counts = sorted(counter.values(), reverse=True)
            # Empate entre top 2
            if len(counts) >= 2 and counts[0] == counts[1]:
                return True
            return False
        
        df['is_problematic'] = df['all_annotations'].apply(check_problematic)
        
        print(f"✓ Consenso calculado:")
        print(f"  - Alto consenso (≥80%): {(df['consensus_level'] == 'high').sum()} instâncias")
        print(f"  - Médio consenso (60-80%): {(df['consensus_level'] == 'medium').sum()} instâncias")
        print(f"  - Baixo consenso (<60%): {(df['consensus_level'] == 'low').sum()} instâncias")
        print(f"  - Casos problemáticos: {df['is_problematic'].sum()} instâncias")
        
        return df
