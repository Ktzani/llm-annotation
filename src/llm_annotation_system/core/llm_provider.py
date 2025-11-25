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

# LLM providers
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_cohere import ChatCohere
except ImportError:
    ChatCohere = None

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
config_path = Path(__file__).parent.parent / 'config'
sys.path.insert(0, str(config_path))

from llms import LLM_CONFIGS, PROVIDER_CONFIGS


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
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'cohere': 'COHERE_API_KEY',
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
            llm = self._create_llm_instance(provider, model_name, params)
            logger.info(f"✓ {model} inicializado")
            return llm
        
        except Exception as e:
            logger.error(f"Erro ao inicializar {model}: {str(e)}")
            raise
    
    def _create_llm_instance(self, provider: str, model_name: str, params: Dict) -> Any:
        """Cria instância de LLM baseado no provider"""
        
        # Proprietários
        if provider == "openai":
            if ChatOpenAI is None:
                raise ImportError("langchain-openai não instalado")
            return ChatOpenAI(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                max_tokens=params.get("max_tokens", 100),
                request_timeout=60,
            )
        
        elif provider == "anthropic":
            if ChatAnthropic is None:
                raise ImportError("langchain-anthropic não instalado")
            return ChatAnthropic(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                max_tokens=params.get("max_tokens", 100),
                timeout=60,
            )
        
        elif provider == "google":
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain-google-genai não instalado")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                max_output_tokens=params.get("max_tokens", 100),
            )
        
        elif provider == "cohere":
            if ChatCohere is None:
                raise ImportError("langchain-cohere não instalado")
            return ChatCohere(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                max_tokens=params.get("max_tokens", 100),
            )
        
        # Open-source
        elif provider == "ollama":
            if ChatOllama is None:
                raise ImportError("langchain-community não instalado ou Ollama não disponível")
            return ChatOllama(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                num_predict=params.get("num_predict", 100),
                base_url=PROVIDER_CONFIGS.get("ollama", {}).get("base_url", "http://localhost:11434"),
            )
        
        elif provider == "huggingface":
            if HuggingFaceHub is None:
                raise ImportError("langchain-community não instalado")
            return HuggingFaceHub(
                repo_id=model_name,
                model_kwargs={
                    "temperature": params.get("temperature", 0.0),
                    "max_new_tokens": params.get("max_new_tokens", 100),
                }
            )
        
        elif provider == "groq":
            if ChatGroq is None:
                raise ImportError("langchain-groq não instalado")
            return ChatGroq(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                max_tokens=params.get("max_tokens", 100),
            )
        
        elif provider == "together":
            if ChatTogether is None:
                raise ImportError("langchain-together não instalado")
            return ChatTogether(
                model=model_name,
                temperature=params.get("temperature", 0.0),
                max_tokens=params.get("max_tokens", 100),
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
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain
