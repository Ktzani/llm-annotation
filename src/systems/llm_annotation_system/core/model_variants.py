"""
Model Variants - Resolve as variações de parâmetros (alternative_params) dos modelos
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger

from src.config.llms import LLM_CONFIGS

VariantSpec = Union[str, List[str]]
AlternativeParamsSelection = Union[bool, str, List[str], Dict[str, VariantSpec]]


class ModelVariantResolver:
    """
    Resolve as variações de parâmetros dos modelos
    Responsabilidades: validar a seleção, gerar os nomes <modelo>_altN e registrá-los em LLM_CONFIGS
    """

    BASE = "base"
    _ALT_PATTERN = re.compile(r"alt\d+")

    def __init__(
        self,
        selection: AlternativeParamsSelection = False,
        llm_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Args:
            selection: False, "alt2", ["base", "alt1"], {"modelo": "alt2"} ou True (base + todas)
            llm_configs: Registro de modelos (default: LLM_CONFIGS)
        """
        self.selection = selection
        self.llm_configs = LLM_CONFIGS if llm_configs is None else llm_configs
        logger.debug(f"ModelVariantResolver inicializado com seleção: {selection}")

    @classmethod
    def normalize_variants(cls, spec: VariantSpec) -> List[str]:
        """Normaliza a escolha de um modelo: "ALT2" → ["alt2"]"""
        tokens = [spec] if isinstance(spec, str) else list(spec)
        if not tokens:
            raise ValueError("lista de variações vazia (use \"base\", \"alt1\", \"alt2\", ...)")

        normalized = []
        for token in tokens:
            value = str(token).strip().lower()
            if value != cls.BASE and not cls._ALT_PATTERN.fullmatch(value):
                raise ValueError(f"variação '{token}' inválida (use \"base\", \"alt1\", \"alt2\", ...)")
            if value not in normalized:
                normalized.append(value)
        return normalized

    def resolve(self, models: List[str]) -> List[str]:
        """
        Traduz a seleção nos nomes a executar (sem registrar)

        Returns:
            Nomes na ordem de `models` (ex.: ["llama3.1-8b_alt2", "qwen3-8b"])
        """
        return [name for name, _, _ in self._resolve(models)]

    def expand(self, models: List[str]) -> List[str]:
        """Resolve a seleção e registra as variações usadas em LLM_CONFIGS"""
        resolved = self._resolve(models)

        for name, model, variant in resolved:
            if variant is not None:
                self._register_variant(name, model, variant)

        return [name for name, _, _ in resolved]

    def get_alternatives(self, model: str) -> Dict[str, Dict[str, Any]]:
        """Retorna as variações do modelo ({"alt1": {...}}), aceitando também o formato em lista"""
        alternatives = self.llm_configs.get(model, {}).get("alternative_params") or {}

        if isinstance(alternatives, list):
            return {f"alt{i}": params for i, params in enumerate(alternatives, start=1)}

        invalid = [name for name in alternatives if not self._ALT_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(
                f"alternative_params de '{model}' em llms.py tem nomes inválidos {invalid}: "
                f"use \"alt1\", \"alt2\", ... (o nome vira sufixo das colunas: <modelo>_altN)"
            )
        return alternatives

    def _variants_per_model(self, models: List[str]) -> Dict[str, List[str]]:
        """Interpreta a seleção como {modelo: [variações]}"""
        if self.selection is False or self.selection is None:
            return {m: [self.BASE] for m in models}

        if self.selection is True:
            return {m: [self.BASE] + list(self.get_alternatives(m)) for m in models}

        if isinstance(self.selection, dict):
            unknown = [m for m in self.selection if m not in models]
            if unknown:
                raise ValueError(
                    f"use_alternative_params cita modelos que não estão em 'models': {unknown}"
                )
            return {
                m: self.normalize_variants(self.selection[m]) if m in self.selection else [self.BASE]
                for m in models
            }

        variants = self.normalize_variants(self.selection)
        return {m: variants for m in models}

    def _resolve(self, models: List[str]) -> List[Tuple[str, str, Optional[str]]]:
        """Retorna (nome_execução, modelo, variação ou None para base)"""
        resolved = []

        for model, variants in self._variants_per_model(models).items():
            for variant in variants:
                if variant == self.BASE:
                    resolved.append((model, model, None))
                    continue

                available = self.get_alternatives(model)
                if variant not in available:
                    options = ", ".join(available) or "nenhuma"
                    raise ValueError(
                        f"'{model}' não tem a variação {variant} (disponíveis: {options})"
                    )
                resolved.append((f"{model}_{variant}", model, variant))

        return resolved

    def _register_variant(self, name: str, model: str, variant: str) -> None:
        """Registra <modelo>_altN em LLM_CONFIGS com os params da variação"""
        config = self.llm_configs[model]

        self.llm_configs[name] = {
            "provider": config["provider"],
            "model_name": config["model_name"],
            "description": f"{config['description']} (variação {variant})",
            "params": self.get_alternatives(model)[variant],
            # Reaproveita os pesos já carregados no provider "transformers"
            "load_params": config.get("load_params"),
        }
        logger.debug(f"Criada variação: {name}")
