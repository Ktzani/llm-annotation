"""
Configurações de LLMs - Apenas Open-Source
"""

# =============================================================================
# CONFIGURAÇÕES DE LLMs - OPEN-SOURCE (Atualizado 2025)
# =============================================================================

LLM_CONFIGS = {
    # -------- Meta Llama 3 --------
    "llama3-70b": {
        "provider": "ollama",
        "model_name": "llama3:70b",
        "description": "Meta Llama 3 70B - Melhor modelo open-source da Meta (2024)",
        "default_params": {"temperature": 0.0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.3, "num_predict": 100},
            {"temperature": 0.5, "num_predict": 100},
        ],
        "requirements": "~40GB RAM",
        "download": "ollama pull llama3:70b"
    },

    "llama3-8b": {
        "provider": "ollama",
        "model_name": "llama3:8b",
        "description": "Meta Llama 3 8B - Rápido e eficiente",
        "default_params": {"temperature": 0.0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.3, "num_predict": 100},
            {"temperature": 0.5, "num_predict": 100},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull llama3:8b"
    },
    "llama3.1-8b": {
        "provider": "ollama",
        "model_name": "llama3.1:8b",
        "description": "Llama 3.1 8B - Melhor modelo 8B da Meta (2025)",
        "default_params": {"temperature": 0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 150},
            {"temperature": 0.7, "num_predict": 150},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull llama3.1:8b"
    },

    "llama3.1-70b": {
        "provider": "ollama",
        "model_name": "llama3.1:70b",
        "description": "Llama 3.1 70B - Muito forte em tasks complexas",
        "default_params": {"temperature": 0.2, "num_predict": 120},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 150},
            {"temperature": 0.7, "num_predict": 200},
        ],
        "requirements": "~40GB RAM",
        "download": "ollama pull llama3.1:70b"
    },

    # -------- Mistral --------
    "mistral-7b": {
        "provider": "ollama",
        "model_name": "mistral:7b",
        "description": "Mistral 7B - Ótimo custo-benefício",
        "default_params": {"temperature": 0.0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.3, "num_predict": 100},
            {"temperature": 0.5, "num_predict": 100},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull mistral:7b"
    },

    "mixtral-8x7b": {
        "provider": "ollama",
        "model_name": "mixtral:8x7b",
        "description": "Mixtral 8x7B MoE - Muito poderoso",
        "default_params": {"temperature": 0.0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.3, "num_predict": 100},
            {"temperature": 0.5, "num_predict": 100},
        ],
        "requirements": "~48GB RAM",
        "download": "ollama pull mixtral:8x7b"
    },
    
    "mistral-nemo-12b": {
        "provider": "ollama",
        "model_name": "mistral-nemo:12b",
        "description": "Mistral Nemo 12B - Muito forte e eficiente",
        "default_params": {"temperature": 0.2, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 120},
            {"temperature": 0.7, "num_predict": 150},
        ],
        "requirements": "~12GB RAM",
        "download": "ollama pull mistral-nemo:12b"
    },

    # -------- Gemma --------
    "gemma-7b": {
        "provider": "ollama",
        "model_name": "gemma:7b",
        "description": "Google Gemma 7B - Leve e rápido",
        "default_params": {"temperature": 0.0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.3, "num_predict": 100},
            {"temperature": 0.5, "num_predict": 100},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull gemma:7b"
    },
    
    "gemma2-9b": {
        "provider": "ollama",
        "model_name": "gemma2:9b",
        "description": "Gemma 2 9B - Forte, leve e rápido",
        "default_params": {"temperature": 0.2, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 120},
            {"temperature": 0.7, "num_predict": 150},
        ],
        "requirements": "~10GB RAM",
        "download": "ollama pull gemma2:9b"
    },

    "gemma2-27b": {
        "provider": "ollama",
        "model_name": "gemma2:27b",
        "description": "Gemma 2 27B - Ótimo custo/benefício",
        "default_params": {"temperature": 0.2, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 120},
            {"temperature": 0.7, "num_predict": 200},
        ],
        "requirements": "~30GB RAM",
        "download": "ollama pull gemma2:27b"
    },
    "gemma3-4b": {
        "provider": "ollama",
        "model_name": "gemma3:4b",
        "description": "Gemma 3 4B - Ótimo custo/benefício",
        "default_params": {"temperature": 0.0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 120},
            {"temperature": 0.7, "num_predict": 200},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull gemma3:4b"
    },

    # -------- Phi-3 Mini --------
    "phi3-mini": {
        "provider": "ollama",
        "model_name": "phi3:mini",
        "description": "Microsoft Phi-3 Mini - Super eficiente",
        "default_params": {"temperature": 0.0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.3, "num_predict": 100},
            {"temperature": 0.5, "num_predict": 100},
        ],
        "requirements": "~4GB RAM",
        "download": "ollama pull phi3:mini"
    },
    "phi3.5-mini": {
        "provider": "ollama",
        "model_name": "phi3.5:mini",
        "description": "Phi-3.5 Mini - Excelente em CPU",
        "default_params": {"temperature": 0.2, "num_predict": 80},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 100},
            {"temperature": 0.7, "num_predict": 120},
        ],
        "requirements": "~4GB RAM",
        "download": "ollama pull phi3.5:mini"
    },

    "phi3.5-medium": {
        "provider": "ollama",
        "model_name": "phi3.5:medium",
        "description": "Phi-3.5 Medium - Melhor modelo leve da Microsoft",
        "default_params": {"temperature": 0.2, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 120},
            {"temperature": 0.7, "num_predict": 150},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull phi3.5:medium"
    },

    # ======================== DEEPSEEK (R1 & V3) =============================

    "deepseek-r1-8b": {
        "provider": "ollama",
        "model_name": "deepseek-r1:8b",
        "description": "DeepSeek R1 8B - Raciocínio muito acima da média",
        "default_params": {"temperature": 0.2, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.5, "num_predict": 100},  # mais criativo
            {"temperature": 0.8, "num_predict": 150},  # brainstorming
        ],
        "requirements": "~10GB RAM",
        "download": "ollama pull deepseek-r1:8b"
    },

    "deepseek-r1-14b": {
        "provider": "ollama",
        "model_name": "deepseek-r1:14b",
        "description": "DeepSeek R1 14B - Melhor custo/benefício para reasoning",
        "default_params": {"temperature": 0.2, "num_predict": 120},
        "alternative_params": [
            {"temperature": 0.5, "num_predict": 150},
            {"temperature": 0.8, "num_predict": 200},
        ],
        "requirements": "~18GB RAM",
        "download": "ollama pull deepseek-r1:14b"
    },

    "deepseek-v3": {
        "provider": "ollama",
        "model_name": "deepseek-v3",
        "description": "DeepSeek V3 - Um dos melhores modelos open-source do mundo",
        "default_params": {"temperature": 0.2, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 150},
            {"temperature": 0.8, "num_predict": 200},
        ],
        "requirements": "~16GB RAM (quantizado)",
        "download": "ollama pull deepseek-v3"
    },

    # ============================= QWEn ================================
    "qwen2-7b": {
        "provider": "ollama",
        "model_name": "qwen2:7b",
        "description": "Qwen 2 7B - Excelente para português",
        "default_params": {"temperature": 0.0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.3, "num_predict": 100},
            {"temperature": 0.5, "num_predict": 100},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull qwen2:7b"
    },

    "qwen2.5-7b": {
        "provider": "ollama",
        "model_name": "qwen2.5:7b",
        "description": "Qwen 2.5 7B - Melhor modelo pequeno para PT-BR (2025)",
        "default_params": {"temperature": 0.2, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 100},
            {"temperature": 0.7, "num_predict": 150},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull qwen2.5:7b"
    },

    "qwen2.5-32b": {
        "provider": "ollama",
        "model_name": "qwen2.5:32b",
        "description": "Qwen 2.5 32B - Um dos melhores OSS do mundo",
        "default_params": {"temperature": 0.2, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 150},
            {"temperature": 0.7, "num_predict": 200},
        ],
        "requirements": "~40GB RAM",
        "download": "ollama pull qwen2.5:32b"
    },
    "qwen3-8b": {
        "provider": "ollama",
        "model_name": "qwen3:8b",
        "description": "Qwen 3 8B - Novo modelo com melhorias significativas",
        "default_params": {"temperature": 0, "num_predict": 100},
        "alternative_params": [
            {"temperature": 0.4, "num_predict": 150},
            {"temperature": 0.7, "num_predict": 150},
        ],
        "requirements": "~8GB RAM",
        "download": "ollama pull qwen3:8b"
    
    }
}


# ============================================================================
# CONFIGURAÇÕES DE PROVIDERS
# ============================================================================

PROVIDER_CONFIGS = {
    "ollama": {
        "provider_name": "Ollama",
        "base_url": "http://localhost:11434",
        "install": "https://ollama.ai/download",
        "free": True,
        "privacy": "100% local",
    },
    
    "groq": {
        "provider_name": "Groq",
        "api_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "speed": "Muito rápido (300+ tokens/s)",
        "free_tier": True,
        "get_key": "https://console.groq.com/keys",
    },
    
    "huggingface": {
        "provider_name": "HuggingFace Inference API",
        "api_url": "https://api-inference.huggingface.co/models/",
        "api_key_env": "HUGGINGFACE_API_KEY",
        "free_tier": True,
        "get_key": "https://huggingface.co/settings/tokens",
    },
    
    "together": {
        "provider_name": "Together AI",
        "api_url": "https://api.together.xyz",
        "api_key_env": "TOGETHER_API_KEY",
        "pricing": "$0.20–0.88 / 1M tokens",
        "get_key": "https://api.together.xyz/signup",
    },
}
