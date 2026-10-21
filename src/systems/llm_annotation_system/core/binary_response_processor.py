"""
Binary Response Processor - Processa as respostas sim/não das LLMs
"""

from typing import Optional
from loguru import logger
import math
import re

from src.systems.llm_annotation_system.core.response_processor import ResponseProcessor


class BinaryResponseProcessor(ResponseProcessor):
    """
    Processa respostas sim/não das LLMs
    Responsabilidades: extrair a resposta (1 = sim, 0 = não, -1 = inválida), sua confiança e o P(sim)
    """

    YES = ("yes", "sim", "true")
    NO = ("no", "não", "nao", "false")

    _THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _ANSWER = re.compile(r"\b(yes|sim|true|no|não|nao|false)\b", re.IGNORECASE)

    @classmethod
    def parse_answer(cls, response: Optional[str]) -> int:
        """Primeira resposta sim/não do texto (fora do <think>): 1, 0 ou -1"""
        if response is None:
            return -1

        text = cls._THINK.sub("", str(response)).strip().lower()
        if text in ("1", "0"):
            return int(text)

        match = cls._ANSWER.search(text)
        if match is None:
            return -1
        return 1 if match.group(1) in cls.YES else 0

    def extract_label_and_confidence(self, response, valid_categories=None):
        content = response.get("content", "")
        label = self.parse_answer(content)
        if label == -1:
            logger.warning(f"Resposta sem sim/não reconhecível: '{str(content)[:50]}'")

        confidence = self._answer_confidence(response.get("logprobs"), label)
        return {"label": label, "confidence": confidence}

    def _answer_confidence(self, logprobs, label: int) -> Optional[float]:
        """Probabilidade do token da resposta, ignorando o raciocínio (mesma lógica do ResponseProcessor)"""
        if not logprobs or label == -1:
            return None

        words = self.YES if label == 1 else self.NO
        inside_think = False
        try:
            for token_info in logprobs:
                token = token_info["token"].strip().lower()

                if token == "<think>":
                    inside_think = True
                    continue
                if token == "</think>":
                    inside_think = False
                    continue

                if not inside_think and token in words:
                    return math.exp(token_info["logprob"])
        except Exception:
            logger.warning("Logprob da resposta não disponível")

        return None

    @staticmethod
    def p_yes(label: int, confidence: Optional[float]) -> Optional[float]:
        """P(sim) da resposta: a confiança do sim ou o complemento da confiança do não"""
        if confidence is None or label == -1:
            return None
        return confidence if label == 1 else 1.0 - confidence
