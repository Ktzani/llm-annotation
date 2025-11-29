"""
LLM Provider - Gerencia inicialização e comunicação com LLMs
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Providers realmente utilizados
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_community.llms import HuggingFaceHub
except ImportError:
    HuggingFaceHub = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_together import ChatTogether
except ImportError:
    ChatTogether = None

# Config imports
import sys

from src.config.llms import LLM_CONFIGS, PROVIDER_CONFIGS

class LLMProvider:
    """
    Gerencia provedores de LLM
    Responsabilidades: inicialização, configuração de API keys, criação de chains
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Args:
            api_keys: Dicionário com chaves de API (opcional)
        """
        self._setup_api_keys(api_keys)
        logger.debug("LLMProvider inicializado")

    def _setup_api_keys(self, api_keys: Optional[Dict[str, str]]):
        """Configura API keys no ambiente"""
        if not api_keys:
            return

        key_mapping = {
            'huggingface': 'HUGGINGFACEHUB_API_TOKEN',
            'groq': 'GROQ_API_KEY',
            'together': 'TOGETHER_API_KEY',
        }

        for key, env_var in key_mapping.items():
            if key in api_keys:
                os.environ[env_var] = api_keys[key]
                logger.debug(f"API key configurada: {key}")

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
        params = config.get("default_params", {})

        logger.debug(f"Inicializando {model} (provider: {provider})")

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

            return ChatOllama(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                num_predict=params.get("num_predict", 200),
                base_url=PROVIDER_CONFIGS["ollama"]["base_url"],
            )

        # ------------------------------------------------------
        # HUGGINGFACE HUB (API inference)
        # ------------------------------------------------------
        elif provider == "huggingface":
            if HuggingFaceHub is None:
                raise ImportError("langchain-community não instalado para HuggingFaceHub")

            return HuggingFaceHub(
                repo_id=model_name,
                model_kwargs={
                    "temperature": params.get("temperature", 0.0),
                    "max_new_tokens": params.get("max_new_tokens", 200),
                },
            )

        # ------------------------------------------------------
        # GROQ (ultra rápido)
        # ------------------------------------------------------
        elif provider == "groq":
            if ChatGroq is None:
                raise ImportError("langchain-groq não instalado")

            return ChatGroq(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                max_tokens=params.get("max_tokens", 200),
            )

        # ------------------------------------------------------
        # TOGETHER (API barata)
        # ------------------------------------------------------
        elif provider == "together":
            if ChatTogether is None:
                raise ImportError("langchain-together não instalado")

            return ChatTogether(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                max_tokens=params.get("max_tokens", 200),
            )

        else:
            raise ValueError(f"Provider '{provider}' não suportado")

    def create_chain(self, llm: Any, template: str, variables: Dict[str, str]):
        """
        Cria uma chain LangChain
        
        Args:
            llm: Instância da LLM
            template: Template do prompt
            variables: Variáveis do template
            
        Returns:
            Chain configurada
        """
        prompt = ChatPromptTemplate.from_template(template, partial_variables=variables)
        chain = prompt | llm | StrOutputParser()
        return chain
