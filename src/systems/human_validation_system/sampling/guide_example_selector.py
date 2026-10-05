"""
Guide Example Selector - Reserva os exemplos do guia do avaliador
"""
import pandas as pd
from loguru import logger

from src.systems.human_validation_system.sampling.seeded_hasher import SeededHasher


class GuideExampleSelector:
    """
    Escolhe os exemplos do guia antes de qualquer rodada.

    Responsabilidades:
    - Usar apenas o grupo C (as três LLMs e a referência coincidem)
    - Preferir textos curtos: sorteia entre os do quantil inferior de tamanho da classe
    - Devolver os exemplos para que sejam retirados do pool de amostragem,
      o que os mantém fora desta e de todas as rodadas futuras
    """

    def __init__(
        self,
        hasher: SeededHasher,
        per_class: int,
        length_quantile: float,
        class_column: str = "ground_truth",
    ):
        self.hasher = hasher
        self.per_class = per_class
        self.length_quantile = length_quantile
        self.class_column = class_column
        logger.debug(f"GuideExampleSelector inicializado ({per_class} por classe)")

    def select(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Exemplos por classe (text_id, text, classe), ordenados por classe."""
        group_c = pool[pool["grupo"] == "C"]
        picked = []

        for cls, docs in group_c.groupby(self.class_column):
            length = docs["text"].str.len()
            short = docs[length <= length.quantile(self.length_quantile)]
            ranked = short.assign(_rank=self.hasher.keys("exemplos_guia", short["text_id"]).values)
            chosen = ranked.sort_values("_rank").head(self.per_class)
            if len(chosen) < 2:
                logger.warning(f"Classe {cls}: apenas {len(chosen)} exemplo(s) disponível(is) no grupo C")
            picked.append(chosen)

        examples = pd.concat(picked).drop(columns="_rank")
        logger.info(f"Exemplos do guia reservados: {len(examples)}")
        return examples[["text_id", "text", self.class_column]].reset_index(drop=True)
