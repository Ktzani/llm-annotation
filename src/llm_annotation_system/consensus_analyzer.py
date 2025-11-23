"""
Consensus Analyzer - Módulo para análise de consenso entre LLMs
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import itertools


class ConsensusAnalyzer:
    """
    Classe para analisar consenso e calcular métricas de concordância
    """
    
    def __init__(self, categories: List[str]):
        """
        Inicializa o analisador
        
        Args:
            categories: Lista de categorias possíveis
        """
        self.categories = categories
    
    def calculate_pairwise_agreement(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Calcula concordância par a par entre anotadores
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas que contêm anotações
        
        Returns:
            DataFrame com matriz de concordância
        """
        n_annotators = len(annotator_cols)
        agreement_matrix = np.zeros((n_annotators, n_annotators))
        kappa_matrix = np.zeros((n_annotators, n_annotators))
        
        for i, col1 in enumerate(annotator_cols):
            for j, col2 in enumerate(annotator_cols):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                    kappa_matrix[i, j] = 1.0
                else:
                    # Concordância simples
                    agreements = (df[col1] == df[col2]).sum()
                    agreement_matrix[i, j] = agreements / len(df)
                    
                    # Cohen's Kappa
                    try:
                        kappa = cohen_kappa_score(df[col1], df[col2])
                        kappa_matrix[i, j] = kappa
                    except:
                        kappa_matrix[i, j] = np.nan
        
        agreement_df = pd.DataFrame(
            agreement_matrix,
            index=annotator_cols,
            columns=annotator_cols
        )
        
        kappa_df = pd.DataFrame(
            kappa_matrix,
            index=annotator_cols,
            columns=annotator_cols
        )
        
        return agreement_df, kappa_df
    
    def calculate_fleiss_kappa(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str]
    ) -> float:
        """
        Calcula Fleiss' Kappa para múltiplos anotadores
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas que contêm anotações
        
        Returns:
            Valor de Fleiss' Kappa
        """
        n_items = len(df)
        n_annotators = len(annotator_cols)
        
        # Construir matriz de anotações
        annotation_matrix = np.zeros((n_items, len(self.categories)))
        
        for idx in range(n_items):
            for col in annotator_cols:
                category = df.iloc[idx][col]
                if category in self.categories:
                    cat_idx = self.categories.index(category)
                    annotation_matrix[idx, cat_idx] += 1
        
        # Calcular proporções
        P_i = np.sum(annotation_matrix ** 2, axis=1) - n_annotators
        P_i = P_i / (n_annotators * (n_annotators - 1))
        P_bar = np.mean(P_i)
        
        # Calcular proporção de cada categoria
        P_j = np.sum(annotation_matrix, axis=0) / (n_items * n_annotators)
        P_e_bar = np.sum(P_j ** 2)
        
        # Fleiss' Kappa
        if P_e_bar == 1.0:
            return 1.0
        
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
        return kappa
    
    def calculate_krippendorff_alpha(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str]
    ) -> float:
        """
        Calcula Krippendorff's Alpha
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas que contêm anotações
        
        Returns:
            Valor de Krippendorff's Alpha
        """
        # Implementação simplificada
        # Para implementação completa, usar: import krippendorff
        
        n_items = len(df)
        n_annotators = len(annotator_cols)
        
        # Construir matriz de coincidências
        coincidence_matrix = np.zeros((len(self.categories), len(self.categories)))
        
        for idx in range(n_items):
            annotations = [df.iloc[idx][col] for col in annotator_cols]
            
            for i, cat1 in enumerate(self.categories):
                for j, cat2 in enumerate(self.categories):
                    count1 = annotations.count(cat1)
                    count2 = annotations.count(cat2)
                    
                    if i == j:
                        coincidence_matrix[i, j] += count1 * (count1 - 1)
                    else:
                        coincidence_matrix[i, j] += count1 * count2
        
        # Calcular observado e esperado
        n_c = np.sum(coincidence_matrix)
        n_k = np.sum(coincidence_matrix, axis=1)
        
        D_o = np.sum(coincidence_matrix * (1 - np.eye(len(self.categories))))
        D_e = np.sum(np.outer(n_k, n_k) * (1 - np.eye(len(self.categories))))
        
        if D_e == 0:
            return 1.0
        
        alpha = 1 - (D_o / D_e)
        return alpha
    
    def analyze_disagreement_patterns(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Analisa padrões de discordância entre anotadores
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas que contêm anotações
        
        Returns:
            Dicionário com análises de discordância
        """
        results = {}
        
        # Matriz de confusão agregada entre pares
        all_pairs = []
        for i, col1 in enumerate(annotator_cols):
            for j, col2 in enumerate(annotator_cols):
                if i < j:
                    all_pairs.append((col1, col2))
        
        # Confusão agregada
        confusion_agg = np.zeros((len(self.categories), len(self.categories)))
        
        for col1, col2 in all_pairs:
            for cat1_idx, cat1 in enumerate(self.categories):
                for cat2_idx, cat2 in enumerate(self.categories):
                    count = ((df[col1] == cat1) & (df[col2] == cat2)).sum()
                    confusion_agg[cat1_idx, cat2_idx] += count
        
        confusion_df = pd.DataFrame(
            confusion_agg,
            index=self.categories,
            columns=self.categories
        )
        results['confusion_matrix'] = confusion_df
        
        # Pares de categorias mais confundidas
        confusion_pairs = []
        for i, cat1 in enumerate(self.categories):
            for j, cat2 in enumerate(self.categories):
                if i < j:
                    confusion_pairs.append({
                        'category_1': cat1,
                        'category_2': cat2,
                        'confusion_count': confusion_agg[i, j] + confusion_agg[j, i]
                    })
        
        confusion_pairs_df = pd.DataFrame(confusion_pairs)
        confusion_pairs_df = confusion_pairs_df.sort_values('confusion_count', ascending=False)
        results['most_confused_pairs'] = confusion_pairs_df
        
        return results
    
    def calculate_entropy_per_instance(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str]
    ) -> pd.Series:
        """
        Calcula entropia da distribuição de anotações por instância
        Maior entropia = mais discordância
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas que contêm anotações
        
        Returns:
            Series com entropia por instância
        """
        entropies = []
        
        for idx in range(len(df)):
            annotations = [df.iloc[idx][col] for col in annotator_cols]
            counter = Counter(annotations)
            
            # Calcular distribuição de probabilidade
            probs = np.array([counter[cat] / len(annotations) for cat in counter])
            
            # Calcular entropia
            ent = entropy(probs, base=2)
            entropies.append(ent)
        
        return pd.Series(entropies, name='annotation_entropy')
    
    def identify_difficult_instances(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str],
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Identifica instâncias difíceis (com baixo consenso)
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas que contêm anotações
            threshold: Threshold de consenso (padrão: 0.5)
        
        Returns:
            DataFrame com instâncias difíceis
        """
        # Calcular consenso
        consensus_scores = []
        
        for idx in range(len(df)):
            annotations = [df.iloc[idx][col] for col in annotator_cols]
            counter = Counter(annotations)
            max_count = counter.most_common(1)[0][1]
            consensus = max_count / len(annotations)
            consensus_scores.append(consensus)
        
        df_copy = df.copy()
        df_copy['consensus_score'] = consensus_scores
        
        # Filtrar instâncias difíceis
        difficult = df_copy[df_copy['consensus_score'] < threshold].copy()
        difficult = difficult.sort_values('consensus_score')
        
        return difficult
    
    def calculate_distance_metrics(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str]
    ) -> Dict[str, float]:
        """
        Calcula múltiplas métricas de distância/concordância
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas que contêm anotações
        
        Returns:
            Dicionário com métricas
        """
        metrics = {}
        
        # Hamming distance média
        hamming_distances = []
        for i, col1 in enumerate(annotator_cols):
            for j, col2 in enumerate(annotator_cols):
                if i < j:
                    hamming = (df[col1] != df[col2]).sum() / len(df)
                    hamming_distances.append(hamming)
        
        metrics['mean_hamming_distance'] = np.mean(hamming_distances)
        metrics['std_hamming_distance'] = np.std(hamming_distances)
        
        # Jaccard similarity média
        jaccard_similarities = []
        for i, col1 in enumerate(annotator_cols):
            for j, col2 in enumerate(annotator_cols):
                if i < j:
                    intersection = (df[col1] == df[col2]).sum()
                    union = len(df)
                    jaccard = intersection / union if union > 0 else 0
                    jaccard_similarities.append(jaccard)
        
        metrics['mean_jaccard_similarity'] = np.mean(jaccard_similarities)
        metrics['std_jaccard_similarity'] = np.std(jaccard_similarities)
        
        # Cohen's Kappa médio
        kappa_scores = []
        for i, col1 in enumerate(annotator_cols):
            for j, col2 in enumerate(annotator_cols):
                if i < j:
                    try:
                        kappa = cohen_kappa_score(df[col1], df[col2])
                        kappa_scores.append(kappa)
                    except:
                        pass
        
        if kappa_scores:
            metrics['mean_cohen_kappa'] = np.mean(kappa_scores)
            metrics['std_cohen_kappa'] = np.std(kappa_scores)
        
        # Fleiss' Kappa
        try:
            metrics['fleiss_kappa'] = self.calculate_fleiss_kappa(df, annotator_cols)
        except:
            metrics['fleiss_kappa'] = None
        
        # Krippendorff's Alpha
        try:
            metrics['krippendorff_alpha'] = self.calculate_krippendorff_alpha(df, annotator_cols)
        except:
            metrics['krippendorff_alpha'] = None
        
        return metrics
    
    def generate_consensus_report(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str],
        output_dir: str = "./results"
    ) -> Dict[str, any]:
        """
        Gera relatório completo de consenso
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas que contêm anotações
            output_dir: Diretório para salvar resultados
        
        Returns:
            Dicionário com todas as métricas e análises
        """
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("GERANDO RELATÓRIO DE CONSENSO")
        print("="*80)
        
        report = {}
        
        # 1. Métricas de distância
        print("\n1. Calculando métricas de distância...")
        report['distance_metrics'] = self.calculate_distance_metrics(df, annotator_cols)
        
        # 2. Concordância par a par
        print("2. Calculando concordância par a par...")
        agreement_df, kappa_df = self.calculate_pairwise_agreement(df, annotator_cols)
        report['pairwise_agreement'] = agreement_df
        report['pairwise_kappa'] = kappa_df
        
        # Salvar
        agreement_df.to_csv(output_dir / "pairwise_agreement.csv")
        kappa_df.to_csv(output_dir / "pairwise_kappa.csv")
        
        # 3. Padrões de discordância
        print("3. Analisando padrões de discordância...")
        disagreement = self.analyze_disagreement_patterns(df, annotator_cols)
        report['disagreement_patterns'] = disagreement
        
        disagreement['confusion_matrix'].to_csv(output_dir / "confusion_matrix.csv")
        disagreement['most_confused_pairs'].to_csv(
            output_dir / "most_confused_pairs.csv",
            index=False
        )
        
        # 4. Entropia por instância
        print("4. Calculando entropia por instância...")
        entropy_series = self.calculate_entropy_per_instance(df, annotator_cols)
        report['instance_entropy'] = entropy_series
        
        # 5. Instâncias difíceis
        print("5. Identificando instâncias difíceis...")
        difficult = self.identify_difficult_instances(df, annotator_cols, threshold=0.6)
        report['difficult_instances'] = difficult
        
        difficult.to_csv(output_dir / "difficult_instances.csv", index=False)
        
        # Imprimir sumário
        print("\n" + "="*80)
        print("SUMÁRIO DO RELATÓRIO")
        print("="*80)
        print(f"\nMétricas de Distância:")
        for metric, value in report['distance_metrics'].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nConcordância Média: {agreement_df.values[np.triu_indices_from(agreement_df.values, k=1)].mean():.4f}")
        print(f"Cohen's Kappa Médio: {kappa_df.values[np.triu_indices_from(kappa_df.values, k=1)].mean():.4f}")
        
        print(f"\nInstâncias Difíceis (consenso < 60%): {len(difficult)}")
        print(f"Entropia Média: {entropy_series.mean():.4f}")
        
        print("\n" + "="*80)
        
        return report
