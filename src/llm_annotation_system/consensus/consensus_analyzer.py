"""
Consensus Analyzer - Análise de consenso refatorada
"""

import pandas as pd
from typing import List, Dict
from pathlib import Path
from loguru import logger

from consensus_metrics import ConsensusMetrics


class ConsensusAnalyzer:
    """
    Analisa consenso entre anotadores
    
    Coordena:
    - ConsensusMetrics: cálculo de métricas
    - Geração de relatórios
    - Identificação de casos problemáticos
    """
    
    def __init__(self, categories: List[str]):
        """
        Args:
            categories: Lista de categorias
        """
        self.categories = categories
        self.metrics = ConsensusMetrics(categories)
        logger.debug("ConsensusAnalyzer inicializado")
    
    def generate_consensus_report(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str],
        output_dir: str = "./results"
    ) -> Dict:
        """
        Gera relatório completo de consenso
        
        Args:
            df: DataFrame com anotações
            annotator_cols: Colunas com anotações
            output_dir: Diretório para salvar
            
        Returns:
            Dicionário com métricas
        """
        logger.info("Gerando relatório de consenso...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        report = {}
        
        # 1. Concordância par a par
        logger.debug("Calculando concordância par a par")
        agreement_df = self.metrics.calculate_pairwise_agreement(df, annotator_cols)
        report['pairwise_agreement'] = agreement_df
        
        # Salvar matriz
        agreement_df.to_csv(output_path / "pairwise_agreement.csv")
        
        # 2. Cohen's Kappa para cada par
        logger.debug("Calculando Cohen's Kappa")
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
        
        # Salvar
        kappa_df.to_csv(output_path / "cohens_kappa.csv", index=False)
        
        # 3. Fleiss' Kappa
        logger.debug("Calculando Fleiss' Kappa")
        fleiss = self.metrics.calculate_fleiss_kappa(df, annotator_cols)
        report['fleiss_kappa'] = fleiss
        report['fleiss_interpretation'] = self.metrics.interpret_kappa(fleiss)
        
        logger.info(f"Fleiss' Kappa: {fleiss:.3f} ({report['fleiss_interpretation']})")
        
        # 4. Casos problemáticos
        logger.debug("Identificando casos problemáticos")
        problematic = self.identify_problematic_cases(df, annotator_cols)
        report['problematic_cases'] = problematic
        
        # Salvar
        if len(problematic) > 0:
            problematic.to_csv(output_path / "problematic_cases.csv", index=False)
            logger.info(f"Casos problemáticos: {len(problematic)}")
        
        logger.success("Relatório gerado")
        
        return report
    
    def identify_problematic_cases(
        self,
        df: pd.DataFrame,
        annotator_cols: List[str],
        threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Identifica casos problemáticos (baixo consenso)
        
        Args:
            df: DataFrame
            annotator_cols: Colunas com anotações
            threshold: Threshold de consenso
            
        Returns:
            DataFrame com casos problemáticos
        """
        problematic = []
        
        for idx, row in df.iterrows():
            # Coletar anotações
            annotations = [row[col] for col in annotator_cols]
            
            # Calcular consenso
            from collections import Counter
            counter = Counter(annotations)
            most_common_count = counter.most_common(1)[0][1]
            consensus_score = most_common_count / len(annotations)
            
            # É problemático?
            if consensus_score < threshold:
                problematic.append({
                    'text_id': row.get('text_id', idx),
                    'text': row.get('text', '')[:100],
                    'consensus_score': consensus_score,
                    'annotations': dict(counter),
                    'entropy': self.metrics.calculate_entropy(annotations)
                })
        
        return pd.DataFrame(problematic)
    
    def compare_models(
        self,
        df: pd.DataFrame,
        model_cols: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Compara performance entre modelos
        
        Args:
            df: DataFrame
            model_cols: Dicionário {nome_modelo: coluna_consenso}
            
        Returns:
            DataFrame com comparação
        """
        logger.info("Comparando modelos...")
        
        comparisons = []
        
        for model_name, col in model_cols.items():
            from collections import Counter
            
            # Distribuição de categorias
            distribution = Counter(df[col])
            
            comparisons.append({
                'model': model_name,
                'total': len(df),
                **{f'count_{cat}': distribution.get(cat, 0) for cat in self.categories}
            })
        
        comparison_df = pd.DataFrame(comparisons)
        
        logger.success("Comparação concluída")
        
        return comparison_df
