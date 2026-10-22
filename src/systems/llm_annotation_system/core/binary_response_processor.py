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

    YES = ("sim", "yes", "true", "verdadeiro", "s", "y", "1")
    NO = ("não", "nao", "no", "false", "falso", "n", "0")

    # Raciocínio: tudo até o último fechamento e qualquer bloco aberto sem fechamento (cortado) fica de fora
    _THINK_CLOSE = re.compile(r"</(?:think|thinking|reasoning)>", re.IGNORECASE)
    _THINK_OPEN = re.compile(r"<(?:think|thinking|reasoning)>.*", re.DOTALL | re.IGNORECASE)
    _UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Chave + separador + valor: {"resposta": "SIM"}, Resposta: SIM, **Resposta final:** NAO, a resposta é SIM
    _KEY_VALUE = (
        r"""(?<!\w)[\\"'“”*]*(?:KEYS)(?:[\s_]*final)?[\\"'“”*]*\s*(?::|=|é|is|->|→)"""
        r"""\s*[\\"'“”*]*\s*([^"'“”,;}\]\n\\]*)"""
    )
    _ANSWER_FIELD = re.compile(_KEY_VALUE.replace("KEYS", "resposta|answer|response"), re.IGNORECASE)
    _OTHER_FIELD = re.compile(
        _KEY_VALUE.replace("KEYS", "r[oó]tulo|label|decis[aã]o|veredito|verdict|resultado|result|classifica[cç][aã]o"),
        re.IGNORECASE,
    )
    # Sem campo: só uma resposta curta e isolada no começo ("SIM", "**Não.**", "Sim, ...", "SIM - ...")
    _ALONE = re.compile(r"""^\W*(\w+)[ \t*_`"'“”.!?)\]}]*(?:$|[,;:\n—–-])""")
    _VALUE_TRIM = " \t*_`.!?\"'“”"

    # Palavras inteiras procuradas nos logprobs (letras soltas casariam com pedaços de outros tokens)
    _CONFIDENCE_WORDS = {1: ("yes", "sim", "true"), 0: ("no", "não", "nao", "false")}

    @classmethod
    def parse_answer(cls, response: Optional[str]) -> int:
        """Resposta do campo de resposta (JSON ou "Resposta: X") ou, sem ele, uma resposta curta e isolada: 1, 0 ou -1"""
        if response is None:
            return -1

        text = cls._THINK_CLOSE.split(str(response))[-1]
        text = cls._THINK_OPEN.sub("", text)
        text = cls._UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text).strip()

        label = cls._field_label(text)
        if label is not None:
            return label

        alone = cls._ALONE.match(text)
        return cls._to_label(alone.group(1)) if alone else -1

    @classmethod
    def _field_label(cls, text: str) -> Optional[int]:
        """Rótulo pelos campos de resposta; None quando o texto não tem nenhum"""
        labels = [cls._to_label(value) for value in cls._ANSWER_FIELD.findall(text)]
        if not labels:
            # Outras chaves só valem com valor sim/não ("rotulo": "ABORTO" é só eco do rótulo)
            labels = [label for label in map(cls._to_label, cls._OTHER_FIELD.findall(text)) if label != -1]
        if not labels:
            return None

        # Valores diferentes (ex.: o modelo repetiu o formato "SIM ou NAO") são ambíguos
        return labels[0] if len(set(labels)) == 1 else -1

    @classmethod
    def _to_label(cls, token: str) -> int:
        """sim/yes/true → 1, não/no/false → 0, outra coisa (ex.: "talvez", "N/A") → -1"""
        token = token.strip(cls._VALUE_TRIM).lower()
        if token in cls.YES:
            return 1
        if token in cls.NO:
            return 0
        return -1

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

        words = self._CONFIDENCE_WORDS[label]
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
