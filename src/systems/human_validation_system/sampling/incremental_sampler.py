"""
Incremental Sampler - Sorteia rodadas sucessivas estratificadas por grupo e classe
"""
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd
from loguru import logger

from src.systems.human_validation_system.sampling.proportional_allocator import ProportionalAllocator
from src.systems.human_validation_system.sampling.seeded_hasher import SeededHasher

Allocation = Dict[str, Dict[int, int]]


class IncrementalStratifiedSampler:
    """
    Sorteia rodadas sucessivas sem repetir documentos.

    Desenho: cada estrato (grupo, classe) tem uma fila fixa, a ordem dada por
    SeededHasher. Cada rodada consome os próximos documentos da fila de cada
    estrato, na quantidade decidida pelo ProportionalAllocator. Assim:
    - nenhuma rodada repete documento (a fila só avança)
    - a rodada k é sempre o mesmo trecho da fila (o entregue não muda)
    - toda a sequência é função de (dados, semente, tamanhos das rodadas),
      o que permite reproduzi-la com `replay`

    Responsabilidades:
    - Ordenar cada estrato pela fila da semente
    - Selecionar a próxima rodada dado o que já foi entregue
    - Embaralhar a rodada misturando os grupos (ordem da planilha)
    """

    GROUPS = ("A", "B", "C")

    def __init__(self, hasher: SeededHasher, allocator: ProportionalAllocator, class_column: str = "ground_truth"):
        self.hasher = hasher
        self.allocator = allocator
        self.class_column = class_column
        logger.debug(f"IncrementalStratifiedSampler inicializado (seed={hasher.seed})")

    def _queue(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Pool ordenado pela fila de cada estrato."""
        ranked = pool.assign(_rank=self.hasher.keys("amostragem", pool["text_id"]).values)
        return ranked.sort_values(["grupo", self.class_column, "_rank"]).reset_index(drop=True)

    def next_round(
        self,
        pool: pd.DataFrame,
        delivered_ids: Set[str],
        size: int,
        min_per_class: int = 0,
        groups: Sequence[str] = GROUPS,
    ) -> Tuple[pd.DataFrame, Allocation]:
        """Próxima rodada: `size` documentos por grupo ativo (`groups`), dado o conjunto já entregue."""
        queue = self._queue(pool)
        is_delivered = queue["text_id"].isin(delivered_ids)
        picked, allocation = [], {}

        for group in self.GROUPS:
            in_group = queue["grupo"] == group
            sizes = queue.loc[in_group, self.class_column].value_counts().to_dict()
            remaining = queue[in_group & ~is_delivered]
            available = remaining[self.class_column].value_counts().to_dict()
            delivered = queue.loc[in_group & is_delivered, self.class_column].value_counts().to_dict()

            if group not in groups:
                allocation[group] = {int(c): 0 for c in sorted(sizes)}
                continue

            alloc = self.allocator.allocate(size, sizes, available, delivered, min_per_class=min_per_class)
            allocation[group] = {int(c): int(n) for c, n in sorted(alloc.items())}

            for cls, n in alloc.items():
                if n:
                    picked.append(remaining[remaining[self.class_column] == cls].head(n))

            if sum(alloc.values()) < size:
                logger.warning(f"Grupo {group}: apenas {sum(alloc.values())} de {size} vagas preenchidas (pool esgotado)")

        selected = pd.concat(picked).drop(columns="_rank") if picked else queue.head(0).drop(columns="_rank")
        return selected, allocation

    def shuffle(self, selected: pd.DataFrame, round_number: int) -> pd.DataFrame:
        """Ordem da planilha: embaralha a rodada inteira, misturando os grupos."""
        keys = self.hasher.keys(f"ordem_rodada_{round_number}", selected["text_id"]).values
        return selected.assign(_order=keys).sort_values("_order").drop(columns="_order").reset_index(drop=True)

    def strata_sizes(self, pool: pd.DataFrame) -> Dict[str, Dict[int, int]]:
        """N_h de cada estrato (grupo -> classe -> tamanho), base dos pesos do estimador."""
        counts = pool.groupby(["grupo", self.class_column]).size()
        return {g: {int(c): int(n) for c, n in counts[g].items()} for g in self.GROUPS if g in counts.index}

    def replay(
        self,
        pool: pd.DataFrame,
        sizes: List[int],
        min_per_class: int,
        groups_per_round: List[List[str]],
    ) -> List[List[str]]:
        """Recalcula do zero os text_ids (na ordem da planilha) das rodadas com tamanhos e grupos dados."""
        delivered: Set[str] = set()
        rounds = []
        for round_number, (size, groups) in enumerate(zip(sizes, groups_per_round), start=1):
            selected, _ = self.next_round(pool, delivered, size, min_per_class if round_number == 1 else 0, groups)
            ids = self.shuffle(selected, round_number)["text_id"].tolist()
            delivered.update(ids)
            rounds.append(ids)
        return rounds
