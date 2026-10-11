"""
Display Text Cleaner - Remove a descrição duplicada dos textos do books só na exibição
"""
import re

from loguru import logger


class DisplayTextCleaner:
    """
    Limpa o texto mostrado ao avaliador, sem alterar os dados.

    O `waashk/books` original traz quase todo texto como
    "título . descrição em minúsculas Descrição Original", isto é, a mesma
    descrição duas vezes. Quando a segunda metade é idêntica à primeira
    (ignorando maiúsculas e espaços), mostra só "título . Descrição Original".
    Qualquer outro texto é devolvido sem mudança.
    """

    SEPARATOR = " . "
    MIN_LENGTH = 40

    def __init__(self):
        logger.debug("DisplayTextCleaner inicializado")

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def clean(self, text: str) -> str:
        title, sep, body = text.partition(self.SEPARATOR)
        if not sep or len(body) < 2 * self.MIN_LENGTH:
            return text
        probe = self._norm(body)[: self.MIN_LENGTH]
        lowered = body.lower()
        # Início da 2ª cópia: próxima ocorrência do começo da descrição
        start = lowered.find(probe[:20], len(probe))
        while start != -1:
            first, second = body[:start], body[start:]
            if self._norm(first) == self._norm(second):
                return f"{title}{sep}{second.strip()}"
            start = lowered.find(probe[:20], start + 1)
        return text
