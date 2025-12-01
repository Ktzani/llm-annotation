"""
Consensus Analyzer - Análise de consenso refatorada
"""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

from src.llm_annotation_system.consensus.consensus_metrics import ConsensusMetrics
from src.llm_annotation_system.consensus.consensus_calculator import ConsensusCalculator

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
        calculator: ConsensusCalculator,
        output_dir: Optional[str] = './results'
    ):
        """
        Args:
            categories: Lista de categorias válidas
            calculator: Instância de ConsensusCalculator (injecção de dependência)
        """
        self.categories = categories
        self.calculator = calculator  # componente externo que calcula consenso interno
        
        output_path = Path(output_dir)
        output_path = output_path.joinpath("consensus")
        output_path.mkdir(exist_ok=True, parents=True)
        self.output_path = output_path
        
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
    # RELATÓRIO COMPLETO (ORQUESTRADOR)
    # -------------------------------------------------------------------------

    def generate_consensus_report(self, df: pd.DataFrame) -> Dict:
        logger.info("Gerando relatório completo de consenso...")

        annotator_cols = self.calculator._extract_consensus_columns(df)

        report = {
            "pairwise_agreement": self._report_pairwise_agreement(df, annotator_cols),
            "cohens_kappa": self._report_cohens_kappa(df, annotator_cols),
        }

        fleiss, interpretation = self._report_fleiss_kappa(df, annotator_cols)
        report["fleiss_kappa"] = fleiss
        report["fleiss_interpretation"] = interpretation

        problematic = self._report_problematic_cases(df, annotator_cols)
        report["problematic_cases"] = problematic

        logger.success("Relatório de consenso gerado com sucesso.")
        return report

    # -------------------------------------------------------------------------
    # 1. PAIRWISE AGREEMENT
    # -------------------------------------------------------------------------

    def _report_pairwise_agreement(self, df, annotator_cols):
        logger.debug("Calculando concordância par a par...")
        agreement_df = self.metrics.calculate_pairwise_agreement(df, annotator_cols)
        agreement_df.to_csv(self.output_path / "pairwise_agreement.csv", index=False)
        return agreement_df

    # -------------------------------------------------------------------------
    # 2. COHEN'S KAPPA
    # -------------------------------------------------------------------------

    def _report_cohens_kappa(self, df, annotator_cols):
        logger.debug("Calculando Cohen's Kappa...")

        kappa_results = []
        for i in range(len(annotator_cols)):
            for j in range(i + 1, len(annotator_cols)):
                kappa = self.metrics.calculate_cohen_kappa(
                    df, annotator_cols[i], annotator_cols[j]
                )
                kappa_results.append({
                    "annotator_1": annotator_cols[i],
                    "annotator_2": annotator_cols[j],
                    "cohens_kappa": kappa,
                    "interpretation": self.metrics.interpret_kappa(kappa)
                })

        kappa_df = pd.DataFrame(kappa_results)
        kappa_df.to_csv(self.output_path / "cohens_kappa.csv", index=False)
        return kappa_df

    # -------------------------------------------------------------------------
    # 3. FLEISS' KAPPA
    # -------------------------------------------------------------------------

    def _report_fleiss_kappa(self, df, annotator_cols):
        logger.debug("Calculando Fleiss' Kappa...")
        fleiss = self.metrics.calculate_fleiss_kappa(df, annotator_cols)
        interpretation = self.metrics.interpret_kappa(fleiss)

        logger.info(f"Fleiss' Kappa: {fleiss:.3f} ({interpretation})")
        return fleiss, interpretation

    # -------------------------------------------------------------------------
    # 4. CASOS PROBLEMÁTICOS
    # -------------------------------------------------------------------------

    def _report_problematic_cases(self, df, annotator_cols):
        logger.debug("Identificando casos problemáticos...")

        problematic = self.identify_problematic_cases(df, annotator_cols)

        if len(problematic) > 0:
            problematic.to_csv(self.output_path / "problematic_cases.csv", index=False)
            logger.info(f"Casos problemáticos: {len(problematic)}")

        return problematic
    
    def evaluate_ground_truth(
        self,
        df_with_consensus: pd.DataFrame,
        ground_truth_col: str = "ground_truth",
    ):
        """
        Avalia o consenso final contra o ground truth.
        
        Retorna:
            - accuracy (float)
            - classification_report (str)
            - confusion_matrix (np.ndarray)
        """

        # Inserir ground truth
        df_with_consensus = df_with_consensus.copy()

        # ---------------------- ACCURACY ----------------------
        accuracy = accuracy_score(
            df_with_consensus[ground_truth_col],
            df_with_consensus["resolved_annotation"]
        )
        logger.success(f"\n🎯 Accuracy: {accuracy:.2%}")

        # ---------------------- REPORT ----------------------
        logger.info("\nClassification Report:")
        cls_report = classification_report(
            df_with_consensus[ground_truth_col],
            df_with_consensus["resolved_annotation"]
        )
        print(cls_report)

        # ---------------------- CONFUSION MATRIX ----------------------
        cm = confusion_matrix(
            df_with_consensus[ground_truth_col],
            df_with_consensus["resolved_annotation"],
            labels=self.categories
        )

        return accuracy, cls_report, cm

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
