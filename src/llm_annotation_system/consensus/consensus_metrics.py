"""
Consensus Metrics - Cálculo de métricas de concordância
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import cohen_kappa_score
import itertools
from loguru import logger


class ConsensusMetrics:
    """
    Calcula métricas de concordância entre anotadores
    Responsabilidades: métricas estatísticas, comparações par a par
    """
    
    def __init__(self, categories: List[str]):
        """
        Args:
            categories: Lista de categorias possíveis
        """
        self.categories = categories
        logger.debug(f"ConsensusMetrics inicializado com {len(categories)} categorias")
    
    def calculate_pairwise_agreement(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Calcula concordância par a par
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas com anotações
            
        Returns:
            DataFrame com matriz de concordância
        """
        logger.debug("Calculando concordância par a par")
        
        n_annotators = len(annotator_cols)
        agreement_matrix = np.zeros((n_annotators, n_annotators))
        
        # Calcular para cada par
        for i, j in itertools.combinations(range(n_annotators), 2):
            col_i = annotator_cols[i]
            col_j = annotator_cols[j]
            
            # Concordância simples
            agreement = (df[col_i] == df[col_j]).sum() / len(df)
            agreement_matrix[i, j] = agreement
            agreement_matrix[j, i] = agreement
        
        # Diagonal = 1 (concordância consigo mesmo)
        np.fill_diagonal(agreement_matrix, 1.0)
        
        return pd.DataFrame(
            agreement_matrix,
            index=annotator_cols,
            columns=annotator_cols
        )
    
    def calculate_cohen_kappa(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str
    ) -> float:
        """
        Calcula Cohen's Kappa entre dois anotadores
        
        Args:
            df: DataFrame
            col1: Primeira coluna
            col2: Segunda coluna
            
        Returns:
            Cohen's Kappa
        """
        try:
            return cohen_kappa_score(df[col1], df[col2])
        except Exception as e:
            logger.warning(f"Erro ao calcular Cohen's Kappa: {e}")
            return 0.0
    
    def calculate_fleiss_kappa(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str]
    ) -> float:
        """
        Calcula Fleiss' Kappa para múltiplos anotadores
        
        Args:
            df: DataFrame
            annotator_cols: Colunas com anotações
            
        Returns:
            Fleiss' Kappa
        """
        logger.debug("Calculando Fleiss' Kappa")
        
        n = len(df)  # número de itens
        k = len(annotator_cols)  # número de anotadores
        
        # Matriz de frequências
        matrix = np.zeros((n, len(self.categories)))
        
        for idx, row in df.iterrows():
            for col in annotator_cols:
                category = row[col]
                if category in self.categories:
                    cat_idx = self.categories.index(category)
                    matrix[idx, cat_idx] += 1
        
        # Proporção de concordância observada
        P_i = (np.sum(matrix ** 2, axis=1) - k) / (k * (k - 1))
        P_bar = np.mean(P_i)
        
        # Proporção esperada por acaso
        p_j = np.sum(matrix, axis=0) / (n * k)
        P_e = np.sum(p_j ** 2)
        
        # Fleiss' Kappa
        if P_e == 1:
            return 1.0
        
        kappa = (P_bar - P_e) / (1 - P_e)
        return kappa
    
    def calculate_entropy(self, annotations: List[str]) -> float:
        """
        Calcula entropia das anotações
        
        Args:
            annotations: Lista de anotações
            
        Returns:
            Entropia
        """
        counter = Counter(annotations)
        total = len(annotations)
        probs = [count / total for count in counter.values()]
        return entropy(probs, base=2)
    
    def interpret_kappa(self, kappa: float) -> str:
        """
        Interpreta valor de Kappa
        
        Args:
            kappa: Valor de Kappa
            
        Returns:
            Interpretação textual
        """
        if kappa > 0.8:
            return "Excelente"
        elif kappa > 0.6:
            return "Bom"
        elif kappa > 0.4:
            return "Moderado"
        elif kappa > 0.2:
            return "Fraco"
        else:
            return "Muito Fraco"
