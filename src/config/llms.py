"""
Configurações de LLMs - Apenas Open-Source
"""

# ============================================================================
# MODELOS OPEN-SOURCE
# ============================================================================

LLM_CONFIGS = {
    # ========== OLLAMA (Modelos Locais - RECOMENDADO) ==========
    
    # Meta Llama 3
    "llama3-70b": {
        "provider": "ollama",
        "model_name": "llama3:70b",
        "description": "Meta Llama 3 70B - Melhor modelo open-source",
        "default_params": {
            "temperature": 0.0,
            "num_predict": 100,
        },
        "requirements": "~40GB RAM",
        "download": "ollama pull llama3:70b"
    },
    
    "llama3-8b": {
        "provider": "ollama",
        "model_name": "llama3:8b",
        "description": "Meta Llama 3 8B - Rápido e eficiente",
        "default_params": {
            "temperature": 0.0,
            "num_predict": 100,
        },
        "requirements": "~8GB RAM",
        "download": "ollama pull llama3:8b"
    },
    
    # Mistral
    "mistral-7b": {
        "provider": "ollama",
        "model_name": "mistral:7b",
        "description": "Mistral 7B - Ótimo custo-benefício",
        "default_params": {
            "temperature": 0.0,
            "num_predict": 100,
        },
        "requirements": "~8GB RAM",
        "download": "ollama pull mistral:7b"
    },
    
    "mixtral-8x7b": {
        "provider": "ollama",
        "model_name": "mixtral:8x7b",
        "description": "Mixtral 8x7B MoE - Muito poderoso",
        "default_params": {
            "temperature": 0.0,
            "num_predict": 100,
        },
        "requirements": "~48GB RAM",
        "download": "ollama pull mixtral:8x7b"
    },
    
    # Qwen (Alibaba - Excelente para PT-BR)
    "qwen2-7b": {
        "provider": "ollama",
        "model_name": "qwen2:7b",
        "description": "Qwen 2 7B - Excelente para português",
        "default_params": {
            "temperature": 0.0,
            "num_predict": 100,
        },
        "requirements": "~8GB RAM",
        "download": "ollama pull qwen2:7b"
    },
    
    # Gemma (Google)
    "gemma-7b": {
        "provider": "ollama",
        "model_name": "gemma:7b",
        "description": "Google Gemma 7B - Leve e rápido",
        "default_params": {
            "temperature": 0.0,
            "num_predict": 100,
        },
        "requirements": "~8GB RAM",
        "download": "ollama pull gemma:7b"
    },
    
    # Phi-3 (Microsoft - Super eficiente)
    "phi3-mini": {
        "provider": "ollama",
        "model_name": "phi3:mini",
        "description": "Microsoft Phi-3 Mini - Super eficiente",
        "default_params": {
            "temperature": 0.0,
            "num_predict": 100,
        },
        "requirements": "~4GB RAM",
        "download": "ollama pull phi3:mini"
    },
    
    # ========== GROQ (API - Gratuito e Ultra Rápido) ==========
    
    "llama3-70b-groq": {
        "provider": "groq",
        "model_name": "llama3-70b-8192",
        "description": "Llama 3 70B via Groq (MUITO rápido - 300+ tokens/s)",
        "default_params": {
            "temperature": 0.0,
            "max_tokens": 100,
        },
        "api_key_required": True,
        "speed": "300+ tokens/s",
        "free_tier": True
    },
    
    "mixtral-8x7b-groq": {
        "provider": "groq",
        "model_name": "mixtral-8x7b-32768",
        "description": "Mixtral 8x7B via Groq",
        "default_params": {
            "temperature": 0.0,
            "max_tokens": 100,
        },
        "api_key_required": True,
        "speed": "300+ tokens/s"
    },
    
    # ========== HUGGINGFACE (API - Gratuito com rate limit) ==========
    
    "llama3-70b-hf": {
        "provider": "huggingface",
        "model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
        "description": "Llama 3 70B via HuggingFace API",
        "default_params": {
            "temperature": 0.0,
            "max_new_tokens": 100,
        },
        "api_key_required": True,
        "free_tier": True
    },
    
    "mistral-7b-hf": {
        "provider": "huggingface",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Mistral 7B via HuggingFace API",
        "default_params": {
            "temperature": 0.0,
            "max_new_tokens": 100,
        },
        "api_key_required": True,
        "free_tier": True
    },
    
    # ========== TOGETHER AI (API - Pago mas barato) ==========
    
    "llama3-70b-together": {
        "provider": "together",
        "model_name": "meta-llama/Llama-3-70b-chat-hf",
        "description": "Llama 3 70B via Together AI",
        "default_params": {
            "temperature": 0.0,
            "max_tokens": 100,
        },
        "api_key_required": True,
        "pricing": "$0.88/M tokens"
    },
    
    "mixtral-8x7b-together": {
        "provider": "together",
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "description": "Mixtral 8x7B via Together AI",
        "default_params": {
            "temperature": 0.0,
            "max_tokens": 100,
        },
        "api_key_required": True,
        "pricing": "$0.60/M tokens"
    },
}

# ============================================================================
# MODELOS PADRÃO RECOMENDADOS
# ============================================================================

DEFAULT_MODELS = [
    "llama3-8b",      # Meta Llama 3 8B - Uso geral
    "mistral-7b",     # Mistral 7B - Rápido
    "qwen2-7b",       # Qwen 2 7B - Excelente PT-BR
]

# ============================================================================
# CONFIGURAÇÕES DE PROVIDERS
# ============================================================================

PROVIDER_CONFIGS = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "install": "https://ollama.ai/download",
        "free": True,
        "privacy": "100% local",
    },
    
    "groq": {
        "api_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "speed": "Muito rápido (300+ tokens/s)",
        "free_tier": True,
        "get_key": "https://console.groq.com/keys",
    },
    
    "huggingface": {
        "api_url": "https://api-inference.huggingface.co/models/",
        "api_key_env": "HUGGINGFACE_API_KEY",
        "free_tier": True,
        "get_key": "https://huggingface.co/settings/tokens",
    },
    
    "together": {
        "api_url": "https://api.together.xyz",
        "api_key_env": "TOGETHER_API_KEY",
        "pricing": "$0.20-0.88/M tokens",
        "get_key": "https://api.together.xyz/signup",
    },
}
