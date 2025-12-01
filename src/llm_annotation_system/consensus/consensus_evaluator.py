"""
Consensus Analyzer - Análise de consenso refatorada
"""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

from src.llm_annotation_system.consensus.consensus_metrics import ConsensusMetrics
from src.llm_annotation_system.consensus.consensus_calculator import ConsensusCalculator

class ConsensusEvaluator:
    """
    Analisa consenso entre anotadores
    
    Coordena:
    - ConsensusMetrics: cálculo de métricas estatísticas
    - ConsensusCalculator: cálculo de consenso e resolução de conflitos
    - Geração de relatórios
    - Identificação de casos problemáticos
    """

    def __init__(
        self,
        categories: List[int],
        calculator: ConsensusCalculator
    ):
        """
        Args:
            categories: Lista de categorias válidas
            calculator: Instância de ConsensusCalculator (injecção de dependência)
        """
        self.categories = categories
        self.calculator = calculator  # componente externo que calcula consenso interno
        self.metrics = ConsensusMetrics(categories)

        logger.debug("ConsensusAnalyzer inicializado")

    # -------------------------------------------------------------------------
    # CONSENSO GLOBAL (usa o ConsensusCalculator)
    # -------------------------------------------------------------------------

    def compute_consensus(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o ConsensusCalculator no dataframe.

        Args:
            df: DataFrame original com colunas '*_consensus'

        Returns:
            DataFrame enriquecido com colunas:
            - all_annotations
            - consensus_score
            - resolved_annotation
            - needs_review
        """
        if self.calculator is None:
            raise RuntimeError(
                "ConsensusAnalyzer precisa receber um ConsensusCalculator para calcular consenso."
            )

        logger.info("Executando cálculo de consenso interno...")
        df = self.calculator.calculate_consensus(df)
        logger.success("Cálculo de consenso finalizado.")
        return df

    # -------------------------------------------------------------------------
    # RELATÓRIO ANALÍTICO COMPLETO
    # -------------------------------------------------------------------------

    def generate_consensus_report(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str],
        output_dir: str = "./results"
    ) -> Dict:
        """
        Gera relatório completo de consenso com métricas estatísticas.

        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas com anotações dos modelos/LLMs
            output_dir: Diretório para salvar relatórios

        Returns:
            Dicionário com DataFrames e valores numéricos
        """
        logger.info("Gerando relatório completo de consenso...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        report = {}

        # ----------------------------------------------------------
        # 1. Concordância par a par
        # ----------------------------------------------------------
        logger.debug("Calculando concordância par a par...")
        agreement_df = self.metrics.calculate_pairwise_agreement(df, annotator_cols)
        report['pairwise_agreement'] = agreement_df
        agreement_df.to_csv(output_path / "pairwise_agreement.csv", index=False)

        # ----------------------------------------------------------
        # 2. Cohen's Kappa para cada par (matriz longa)
        # ----------------------------------------------------------
        logger.debug("Calculando Cohen's Kappa...")
        kappa_results = []

        for i in range(len(annotator_cols)):
            for j in range(i + 1, len(annotator_cols)):
                kappa = self.metrics.calculate_cohen_kappa(
                    df, annotator_cols[i], annotator_cols[j]
                )
                kappa_results.append({
                    'annotator_1': annotator_cols[i],
                    'annotator_2': annotator_cols[j],
                    'cohens_kappa': kappa,
                    'interpretation': self.metrics.interpret_kappa(kappa)
                })

        kappa_df = pd.DataFrame(kappa_results)
        report['cohens_kappa'] = kappa_df
        kappa_df.to_csv(output_path / "cohens_kappa.csv", index=False)

        # ----------------------------------------------------------
        # 3. Fleiss' Kappa global
        # ----------------------------------------------------------
        logger.debug("Calculando Fleiss' Kappa...")
        fleiss = self.metrics.calculate_fleiss_kappa(df, annotator_cols)
        report['fleiss_kappa'] = fleiss
        report['fleiss_interpretation'] = self.metrics.interpret_kappa(fleiss)

        logger.info(f"Fleiss' Kappa: {fleiss:.3f} ({report['fleiss_interpretation']})")

        # ----------------------------------------------------------
        # 4. Casos problemáticos
        # ----------------------------------------------------------
        logger.debug("Identificando casos problemáticos...")
        problematic = self.identify_problematic_cases(df, annotator_cols)
        report['problematic_cases'] = problematic

        if len(problematic) > 0:
            problematic.to_csv(output_path / "problematic_cases.csv", index=False)
            logger.info(f"Casos problemáticos: {len(problematic)}")

        # Final
        logger.success("Relatório de consenso gerado com sucesso.")
        return report

    # -------------------------------------------------------------------------
    # IDENTIFICAÇÃO DE CASOS PROBLEMÁTICOS
    # -------------------------------------------------------------------------

    def identify_problematic_cases(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str],
        threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Identifica casos problemáticos — baixo consenso entre anotadores.

        Args:
            df: DataFrame
            annotator_cols: lista de colunas de anotadores
            threshold: valor mínimo para ser considerado confiável

        Returns:
            DataFrame com:
            - text_id
            - text
            - consensus_score
            - annotations
            - entropy
        """
        problematic = []

        for idx, row in df.iterrows():
            annotations = [row[col] for col in annotator_cols]

            from collections import Counter
            counter = Counter(annotations)

            most_common_count = counter.most_common(1)[0][1]
            consensus_score = most_common_count / len(annotations)

            if consensus_score < threshold:
                problematic.append({
                    'text_id': row.get('text_id', idx),
                    'text': row.get('text', '')[:100],
                    'consensus_score': consensus_score,
                    'annotations': dict(counter),
                    'entropy': self.metrics.calculate_entropy(annotations)
                })

        return pd.DataFrame(problematic)

    # -------------------------------------------------------------------------
    # COMPARAÇÃO ENTRE MODELOS
    # -------------------------------------------------------------------------

    def compare_models(
        self,
        df: pd.DataFrame,
        model_cols: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Compara distribuição de classes entre modelos.

        Args:
            df: DataFrame
            model_cols: dict {nome_modelo: coluna_consenso}

        Returns:
            DataFrame com contagens por categoria
        """
        logger.info("Comparando modelos...")

        comparisons = []

        for model_name, col in model_cols.items():
            from collections import Counter
            distribution = Counter(df[col])

            comparisons.append({
                'model': model_name,
                'total': len(df),
                **{f'count_{cat}': distribution.get(cat, 0) for cat in self.categories}
            })

        comparison_df = pd.DataFrame(comparisons)
        logger.success("Comparação entre modelos concluída.")
        return comparison_df
