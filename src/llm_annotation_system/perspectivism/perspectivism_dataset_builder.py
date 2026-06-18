"""
Construção do dataset de PERSPECTIVISMO para fine-tuning.

Perspectivismo: em vez de treinar sobre o voto majoritário agregado
(`resolved_annotation`), treinamos sobre cada anotação de LLM separadamente.
Para um mesmo texto são geradas N linhas (uma por LLM), podendo ter rótulos
diferentes entre si — preservando o desacordo entre anotadores em vez de
colapsá-lo em um único rótulo de consenso.

A entrada é o `annotations.csv` (anotações brutas), que carrega os votos
desagregados de cada LLM nas colunas `<modelo>_consensus` — independente do
consenso agregado. Esta classe explode essas colunas em formato longo e salva o
resultado em uma pasta `perspectivismo/`.
"""

from pathlib import Path
from typing import List, Sequence

import pandas as pd
from loguru import logger

from src.utils.get_text_id_from_text import get_text_id_from_text
from src.utils.data_loader import add_label_description


class PerspectivismDatasetBuilder:
    """
    Constrói o dataset de perspectivismo (formato longo) a partir do CSV anotado
    completo.

    Cada coluna `<modelo>_consensus` representa o rótulo final daquela LLM para o
    texto. O builder gera uma linha por par (texto, LLM), produzindo um dataset
    com as colunas:

        text_id, text, label, llm, label_description

    Parameters
    ----------
    dataset_name:
        Nome do dataset (usado para mapear `label` -> `label_description`).
    label_suffix:
        Sufixo das colunas que carregam o rótulo de cada LLM. Default
        `"_consensus"` (consenso interno de cada LLM entre suas repetições).
    exclude_labels:
        Rótulos a descartar linha a linha (ex.: `-1` para anotações inválidas).
        Diferente do modo agregado, aqui descartamos apenas a linha da LLM
        inválida — as demais perspectivas do mesmo texto são preservadas.
    """

    def __init__(
        self,
        dataset_name: str,
        label_suffix: str = "_consensus",
        exclude_labels: Sequence[int] = (-1,),
    ):
        self.dataset_name = dataset_name
        self.label_suffix = label_suffix
        self.exclude_labels = list(exclude_labels)

    def detect_llm_label_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detecta as colunas de rótulo por LLM (`<modelo>_consensus`).

        Considera apenas colunas que terminam em `label_suffix` e que possuem a
        coluna de score correspondente (`<modelo>_consensus_score`), evitando
        falsos positivos como a coluna agregada global `consensus_score`.
        """
        candidates = []
        for col in df.columns:
            if not col.endswith(self.label_suffix):
                continue
            llm = col[: -len(self.label_suffix)]
            if not llm:
                continue
            # Heurística de segurança: a coluna de score por LLM acompanha o rótulo.
            if f"{col}_score" in df.columns:
                candidates.append(col)

        if not candidates:
            raise ValueError(
                f"Nenhuma coluna de rótulo por LLM (sufixo '{self.label_suffix}') "
                f"encontrada no dataset anotado. Colunas disponíveis: {list(df.columns)}"
            )

        logger.info(
            f"Perspectivismo: {len(candidates)} LLM(s) detectada(s): "
            f"{[c[:-len(self.label_suffix)] for c in candidates]}"
        )
        return candidates

    def build(self, df_full: pd.DataFrame) -> pd.DataFrame:
        """Gera o dataset longo (uma linha por par texto/LLM)."""
        llm_cols = self.detect_llm_label_columns(df_full)

        frames = []
        for col in llm_cols:
            llm = col[: -len(self.label_suffix)]
            sub = df_full[["text", col]].rename(columns={col: "label"}).copy()
            sub["llm"] = llm
            frames.append(sub)

        long_df = pd.concat(frames, ignore_index=True)

        # Normaliza texto e chave canônica (mesma do alinhamento com os splits HF)
        long_df["text"] = long_df["text"].astype(str).str.strip()
        long_df["text_id"] = long_df["text"].apply(get_text_id_from_text)

        # Rótulos: coage para inteiro e descarta NaN / inválidos linha a linha
        long_df["label"] = pd.to_numeric(long_df["label"], errors="coerce")
        before = len(long_df)
        long_df = long_df.dropna(subset=["label"]).reset_index(drop=True)
        long_df["label"] = long_df["label"].astype(int)
        long_df = long_df[~long_df["label"].isin(self.exclude_labels)].reset_index(drop=True)
        removed = before - len(long_df)
        if removed:
            logger.info(f"Perspectivismo: {removed} linha(s) removida(s) (rótulo NaN/inválido).")

        long_df = add_label_description(long_df, dataset_name=self.dataset_name)

        long_df = long_df[["text_id", "text", "label", "llm", "label_description"]]

        logger.info(
            f"Perspectivismo: {long_df['text_id'].nunique()} textos -> {len(long_df)} linhas "
            f"({len(llm_cols)} perspectivas por texto)."
        )
        return long_df

    #: Nome do arquivo salvo dentro da pasta de perspectivismo.
    FILE_NAME = "dataset_perspectivismo.csv"

    @classmethod
    def output_path(cls, output_dir: Path) -> Path:
        """Caminho canônico do dataset de perspectivismo salvo."""
        return Path(output_dir) / cls.FILE_NAME

    def build_and_save(self, df_full: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        """Gera o dataset longo e o persiste em `output_dir/dataset_perspectivismo.csv`."""
        long_df = self.build(df_full)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_path(output_dir)
        long_df.to_csv(output_path, index=False)
        logger.success(f"Dataset de perspectivismo salvo em: {output_path}")

        return long_df

    def load_or_build(
        self,
        df_full: pd.DataFrame,
        output_dir: Path,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Reutiliza o dataset de perspectivismo já salvo, se existir; caso
        contrário, gera e salva a partir de `df_full`.

        Permite gerar o dataset uma única vez (ex.: por `run_perspectivism.py`)
        e reaproveitá-lo no fine-tuning sem refazer a operação. Com `force=True`,
        regenera mesmo que o arquivo já exista.
        """
        output_path = self.output_path(output_dir)

        if output_path.exists() and not force:
            long_df = pd.read_csv(output_path)
            logger.info(
                f"Perspectivismo: dataset existente reutilizado "
                f"({len(long_df)} linhas): {output_path}"
            )
            return long_df

        return self.build_and_save(df_full, output_dir)
