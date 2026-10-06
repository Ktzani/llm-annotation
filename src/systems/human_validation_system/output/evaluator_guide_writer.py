"""
Evaluator Guide Writer - Gera o guia do avaliador em Markdown
"""
from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger


class EvaluatorGuideWriter:
    """
    Gera o guia do avaliador de um dataset.

    Responsabilidades:
    - Instrução das duas etapas e do preenchimento da planilha
    - Lista das classes (nome canônico + definição complementar)
    - Exemplos por classe, vindos dos documentos reservados
    """

    FILE_NAME = "guia_avaliador.md"

    def __init__(
        self,
        dataset_name: str,
        label_names: Dict[int, str],
        definitions: Dict[str, str],
        max_example_chars: int,
        insufficient_option: str,
        class_column: str = "ground_truth",
    ):
        self.dataset_name = dataset_name
        self.label_names = label_names
        self.definitions = definitions
        self.max_example_chars = max_example_chars
        self.insufficient_option = insufficient_option
        self.class_column = class_column
        logger.debug(f"EvaluatorGuideWriter inicializado ({dataset_name})")

    def _excerpt(self, text: str) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= self.max_example_chars else text[: self.max_example_chars].rstrip() + "…"

    def render(self, examples: pd.DataFrame) -> str:
        names = [self.label_names[c] for c in sorted(self.label_names)]
        lines = [
            f"# Guia do avaliador — {self.dataset_name}",
            "",
            "## Como avaliar",
            "",
            "Para cada linha da planilha, leia o texto e siga duas etapas:",
            "",
            "1. **Escolha o rótulo mais adequado** para o texto na coluna `rotulo_escolhido`.",
            "2. **Indique se algum outro rótulo também poderia ser razoavelmente justificado**: "
            "marque `sim` ou `não` em `outro_rotulo_possivel`. Se marcar `sim`, escolha esse rótulo "
            "em `qual_outro_rotulo`.",
            "",
            f"Se o texto não traz informação suficiente para decidir, ou nenhuma das classes se aplica, "
            f"escolha `{self.insufficient_option}` em `rotulo_escolhido` (texto truncado, genérico demais ou fora "
            "de todos os rótulos). Em dúvida entre dois rótulos, não use esta opção: escolha o mais adequado e "
            "indique o outro na etapa 2.",
            "",
            "As colunas de rótulo aceitam apenas os valores da lista suspensa, escritos exatamente como abaixo. "
            "Avalie cada texto de forma independente e não consulte os outros avaliadores.",
            "",
            "## Classes",
            "",
            "Use sempre o nome da coluna **Rótulo**, exatamente como escrito. A descrição é apenas "
            "um apoio para entender a classe e não deve ser usada como rótulo.",
            "",
            "| Rótulo (use este nome) | Descrição (apenas apoio) |",
            "|---|---|",
        ]
        for code in sorted(self.label_names):
            definition = self.definitions.get(str(code), "PENDENTE")
            lines.append(f"| `{self.label_names[code]}` | {definition} |")

        lines += ["", "## Exemplos", ""]
        for code in sorted(self.label_names):
            lines += [f"### `{self.label_names[code]}`", ""]
            texts = examples.loc[examples[self.class_column] == code, "text"]
            if texts.empty:
                lines += ["_Sem exemplos disponíveis._", ""]
            for i, text in enumerate(texts, start=1):
                lines += [f"{i}. {self._excerpt(text)}", ""]

        lines += [
            "---",
            "",
            f"Rótulos válidos: {', '.join(f'`{n}`' for n in names)} "
            f"(em `rotulo_escolhido` também `{self.insufficient_option}`)",
            "",
        ]
        return "\n".join(lines)

    def write(self, examples: pd.DataFrame, output_dir: Path, overwrite: bool = False) -> Path:
        path = Path(output_dir) / self.FILE_NAME
        if path.exists() and not overwrite:
            logger.info(f"Guia já existe, mantido: {path}")
            return path
        path.write_text(self.render(examples), encoding="utf-8")
        logger.success(f"Guia salvo: {path}")
        return path
