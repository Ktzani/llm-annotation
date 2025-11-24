# =============================================================================
# CONFIGURAÇÕES DOS MODELOS LLM
# =============================================================================

LLM_CONFIGS = {
    "gpt-4-turbo": {
        "provider": "openai",
        "model_name": "gpt-4-turbo-preview",
        "default_params": {
            "temperature": 0.0,  # Determinístico para consistência
            "max_tokens": 50,
            "top_p": 1.0,
        },
        "alternative_params": [
            {"temperature": 0.3, "max_tokens": 50, "top_p": 0.95},
            {"temperature": 0.5, "max_tokens": 50, "top_p": 0.9},
        ]
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "default_params": {
            "temperature": 0.0,
            "max_tokens": 50,
            "top_p": 1.0,
        },
        "alternative_params": [
            {"temperature": 0.3, "max_tokens": 50, "top_p": 0.95},
        ]
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "model_name": "claude-3-opus-20240229",
        "default_params": {
            "temperature": 0.0,
            "max_tokens": 50,
        },
        "alternative_params": [
            {"temperature": 0.3, "max_tokens": 50},
        ]
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "default_params": {
            "temperature": 0.0,
            "max_tokens": 50,
        },
        "alternative_params": [
            {"temperature": 0.3, "max_tokens": 50},
        ]
    },
    "gemini-pro": {
        "provider": "google",
        "model_name": "gemini-pro",
        "default_params": {
            "temperature": 0.0,
            "max_output_tokens": 50,
            "top_p": 1.0,
        },
        "alternative_params": [
            {"temperature": 0.3, "max_output_tokens": 50, "top_p": 0.95},
        ]
    },
}