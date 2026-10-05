"""
Anonymizer - Gera o id_anonimo de cada documento
"""
from typing import Dict, Iterable

from loguru import logger

from src.systems.human_validation_system.sampling.seeded_hasher import SeededHasher


class AnonymousIdGenerator:
    """
    Gera ids anônimos estáveis por documento.

    Responsabilidades:
    - Derivar o id apenas de (semente, text_id), sem informação de grupo,
      classe ou rodada
    - Garantir unicidade sobre todo o pool (logo, entre todas as rodadas)
    """

    def __init__(self, hasher: SeededHasher, prefix: str, length: int = 8):
        self.hasher = hasher
        self.prefix = prefix
        self.length = length
        logger.debug(f"AnonymousIdGenerator inicializado (prefixo='{prefix}')")

    def generate(self, text_ids: Iterable[str]) -> Dict[str, str]:
        """Mapa text_id -> id_anonimo; erro em caso de colisão."""
        mapping = {
            str(t): f"{self.prefix}-{self.hasher.key('id_anonimo', t)[: self.length].upper()}"
            for t in text_ids
        }
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Colisão de id_anonimo; aumente `length`")
        return mapping
