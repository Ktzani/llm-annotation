"""
LLM Provider - Gerencia inicialização e comunicação com LLMs
"""

import os
from typing import Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Providers realmente utilizados
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_huggingface import ChatHuggingFace
    from langchain_huggingface import HuggingFaceEndpoint
except ImportError:
    ChatHuggingFace = None
    HuggingFaceEndpoint = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

# Config imports
import sys

from src.config.llms import LLM_CONFIGS, PROVIDER_CONFIGS

class LLMProvider:
    """
    Gerencia provedores de LLM
    Responsabilidades: inicialização, configuração de API keys, criação de chains
    """

    def __init__(self):
        load_dotenv()               
        self._setup_api_keys()
        logger.debug("LLMProvider inicializado")

    def _setup_api_keys(self):
        """Configura chaves de API a partir do ambiente ou parâmetros"""
    
        expected_keys = {
            'huggingface': 'HUGGINGFACEHUB_API_TOKEN',
            'groq': 'GROQ_API_KEY',
        }

        for provider, env_var in expected_keys.items():
            if os.getenv(env_var):
                logger.debug(f"API key encontrada para provider: {provider}")
                
    def _validate_required_keys(self, provider: str):
        required = {
            "huggingface": "HUGGINGFACEHUB_API_TOKEN",
            "groq": "GROQ_API_KEY",
        }

        env = required.get(provider)
        if env and not os.getenv(env):
            raise RuntimeError(f"Missing API key: {env}")
        
    def initialize_llm(self, model: str) -> Any:
        """
        Inicializa uma LLM
        
        Args:
            model: Nome do modelo
            
        Returns:
            Instância da LLM inicializada
        """
        if model not in LLM_CONFIGS:
            raise ValueError(f"Modelo '{model}' não configurado")

        config = LLM_CONFIGS[model]
        provider = config["provider"]
        model_name = config["model_name"]
        params = config.get("params", {})

        logger.debug(f"Inicializando {model} (provider: {provider})")
        
        # valida API key ANTES de instanciar
        self._validate_required_keys(provider)

        try:
            return self._create_llm_instance(provider, model_name, params)
        except Exception as e:
            logger.error(f"Erro ao inicializar {model}: {e}")
            raise

    def _create_llm_instance(self, provider: str, model_name: str, params: Dict) -> Any:
        """
        Cria instância de LLM baseado no provider
        """

        # ------------------------------------------------------
        # OLLAMA (Modelos Locais)
        # ------------------------------------------------------
        if provider == "ollama":
            if ChatOllama is None:
                raise ImportError("langchain-community não instalado para ChatOllama")
            
            ollama_allowed = {"temperature", "num_predict", "top_p", "stop"}

            ollama_params = self._filter_explicit_params(params, ollama_allowed)

            return ChatOllama(
                model=model_name,
                base_url=PROVIDER_CONFIGS["ollama"]["base_url"],
                **ollama_params
            )

        # ------------------------------------------------------
        # HUGGINGFACE HUB (API inference)
        # ------------------------------------------------------
        elif provider == "huggingface":
            if ChatHuggingFace is None or HuggingFaceEndpoint is None:
                raise ImportError("langchain-huggingface não instalado")
            
            hf_allowed = {
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "do_sample",
                "repetition_penalty",
            }

            hf_params = self._filter_explicit_params(params, hf_allowed)
        
            endpoint = HuggingFaceEndpoint(
                repo_id=model_name,
                task="chat-completion",
                huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
                **hf_params
            )
        
            return ChatHuggingFace(llm=endpoint)

        # ------------------------------------------------------
        # GROQ (ultra rápido)
        # ------------------------------------------------------
        elif provider == "groq":
            if ChatGroq is None:
                raise ImportError("langchain-groq não instalado")
            
            groq_allowed = {"temperature", "max_tokens", "top_p", "stop"}

            groq_params = self._filter_explicit_params(params, groq_allowed)

            return ChatGroq(
                model=model_name,
                **groq_params
            )

        else:
            raise ValueError(f"Provider '{provider}' não suportado")
        
    def _filter_explicit_params(self, params: Dict[str, Any], allowed: set) -> Dict[str, Any]:
        """
        Retorna apenas parâmetros explicitamente definidos
        e suportados pelo provider
        """
        if not params:
            return {}

        return {
            k: v
            for k, v in params.items()
            if k in allowed and v is not None
        }


    def create_chain(self, llm: Any, template: str):
        """
        Cria uma chain LangChain
        
        Args:
            llm: Instância da LLM
            template: Template do prompt
            variables: Variáveis do template
            
        Returns:
            Chain configurada
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain
