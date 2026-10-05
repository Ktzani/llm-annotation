"""
Proportional Allocator - Distribui as vagas de um grupo entre as classes
"""
import math
from typing import Dict

from loguru import logger


class ProportionalAllocator:
    """
    Distribui as vagas de uma rodada entre as classes de um grupo.

    Responsabilidades:
    - Manter cada classe proporcional ao seu tamanho no grupo em CADA rodada:
      a classe recebe o piso ou o teto da sua cota (desvio < 1 documento)
    - Decidir quem recebe o teto pelo déficit acumulado, o que mantém também o
      total acumulado próximo da proporção
    - Garantir um mínimo de documentos por classe disponível (`min_per_class`,
      usado só na 1ª rodada; nos incrementos uma classe pequena pode não aparecer)
    - Redistribuir a cota de classes esgotadas entre as que ainda têm documentos
    """

    def __init__(self):
        logger.debug("ProportionalAllocator inicializado")

    def allocate(
        self,
        n: int,
        sizes: Dict[int, int],
        available: Dict[int, int],
        delivered: Dict[int, int],
        min_per_class: int = 0,
    ) -> Dict[int, int]:
        """
        Args:
            n: vagas da rodada no grupo
            sizes: tamanho original de cada classe no grupo (define a proporção)
            available: documentos ainda não entregues por classe
            delivered: documentos já entregues por classe nas rodadas anteriores

        Returns:
            Vagas por classe nesta rodada
        """
        alloc = {c: 0 for c in sizes}
        active = [c for c in sorted(sizes) if available.get(c, 0) > 0]
        n = min(n, sum(available.get(c, 0) for c in active))
        if n <= 0:
            return alloc

        total_size = sum(sizes[c] for c in active)
        quota = {c: n * sizes[c] / total_size for c in active}
        for c in active:
            alloc[c] = min(math.floor(quota[c]), available[c])

        floor_of = {c: min(min_per_class, available[c]) for c in active}
        if min_per_class and sum(floor_of.values()) <= n:
            for c in active:
                alloc[c] = max(alloc[c], floor_of[c])
            # O mínimo pode estourar n: devolve vagas de quem está mais acima da cota
            while sum(alloc.values()) > n:
                donor = max((c for c in active if alloc[c] > floor_of[c]), key=lambda c: (alloc[c] - quota[c], -c))
                alloc[donor] -= 1

        # Déficit acumulado após esta rodada: meta proporcional menos o já entregue
        total_after = sum(delivered.get(c, 0) for c in sizes) + n
        target = {c: total_after * sizes[c] / total_size for c in active}

        def priority(c):
            return (target[c] - delivered.get(c, 0) - alloc[c], quota[c] - alloc[c], -c)

        # 1ª passada respeita o teto da cota (ou o mínimo); 2ª só atua se classes esgotarem
        for capped in (True, False):
            while sum(alloc.values()) < n:
                eligible = [
                    c for c in active
                    if alloc[c] < available[c] and (not capped or alloc[c] < max(math.ceil(quota[c]), floor_of[c]))
                ]
                if not eligible:
                    break
                alloc[max(eligible, key=priority)] += 1

        return alloc
