"""
Script para Análise de Consenso entre LLMs - Anotação sem LLM Hacking
Converte o notebook analise_consenso_llms.ipynb em script executável
"""

import sys
import json
from pathlib import Path
from loguru import logger
import pandas as pd

from src.utils.data_loader import load_hf_dataset
from src.llm_annotation_system.annotation.llm_annotator import LLMAnnotator
from src.llm_annotation_system.consensus.consensus_calculator import ConsensusCalculator
from src.llm_annotation_system.consensus.consensus_evaluator import ConsensusEvaluator
from src.llm_annotation_system.consensus.consensus_visualizer import ConsensusVisualizer
from src.experiments.base_experiment import EXPERIMENT_CONFIG


def setup_logger():
    """Configura o logger"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.success("✓ Setup completo")


def load_dataset(dataset_name: str):
    """Carrega o dataset especificado"""
    logger.info(f"Carregando dataset: {dataset_name}")
    texts, categories, ground_truth = load_hf_dataset(dataset_name)
    
    logger.info(f"Textos: {len(texts)}")
    logger.info(f"Categorias: {categories}")
    logger.info(f"Ground truth: {'Sim' if ground_truth else 'Não'}")
    
    # Amostra
    logger.info("Amostra dos textos:")
    for i, text in enumerate(texts[:3]):
        logger.info(f"{i+1}. {text[:100]}...")
        if ground_truth:
            logger.info(f"   Label: {ground_truth[i]}")
    
    return texts, categories, ground_truth


def initialize_annotator(dataset_name: str, categories: list):
    """Inicializa o LLMAnnotator"""
    logger.info("Inicializando annotator...")
    
    DEFAULT_MODELS = EXPERIMENT_CONFIG["default_models"]
    PROMPT_TEMPLATE = EXPERIMENT_CONFIG["prompt_template"]
    USE_ALTERNATIVE_PARAMS = EXPERIMENT_CONFIG["use_alternative_params"]
    
    annotator = LLMAnnotator(
        dataset_name=dataset_name,
        categories=categories,
        models=DEFAULT_MODELS,
        prompt_template=PROMPT_TEMPLATE,
        use_langchain_cache=True,
        use_alternative_params=USE_ALTERNATIVE_PARAMS
    )
    
    logger.success(f"✓ Annotator inicializado com {len(annotator.models)} modelos")
    return annotator


def run_annotation(annotator: LLMAnnotator, texts: list, num_repetitions: int = 1, 
                   use_cache: bool = True):
    
    logger.info("Iniciando anotação")
    logger.info(f"Textos: {len(texts)} | Modelos: {len(annotator.models)} | Repetições: {num_repetitions}")
    logger.info(f"Total de anotações: {len(texts) * len(annotator.models) * num_repetitions}")

    df_annotations = annotator.annotate_dataset(
        texts=texts,
        num_repetitions=num_repetitions,
        use_cache=use_cache
    )
    logger.success("✓ Anotações completas")
    
    return df_annotations


def calculate_model_metrics(annotator: LLMAnnotator, df_annotations: pd.DataFrame, ground_truth: list):
    """Calcula métricas por modelo"""
    if not ground_truth:
        logger.warning("Ground truth não disponível - pulando métricas por modelo")
        return None
    
    df_annotations["ground_truth"] = ground_truth
    df_metrics = annotator.evaluate_model_metrics(
        df_annotations, 
        ground_truth_col="ground_truth", 
        output_csv=True
    )
    return df_metrics


def compute_consensus(df_annotations: pd.DataFrame, categories: list, output_dir: Path):
    """Calcula o consenso entre modelos"""
    logger.info("Executando cálculo de consenso interno...")
    
    calculator = ConsensusCalculator(
        consensus_threshold=EXPERIMENT_CONFIG['consensus'].get('threshold', 0.8),
        default_strategy=EXPERIMENT_CONFIG['consensus'].get('strategy', "majority_vote")
    )
    
    analyzer = ConsensusEvaluator(
        categories=categories, 
        calculator=calculator, 
        output_dir=output_dir
    )
    
    df_with_consensus = analyzer.compute_consensus(df_annotations)
    
    # Estatísticas
    logger.info("\n📊 Estatísticas de Consenso:")
    logger.info(f"  Média: {df_with_consensus['consensus_score'].mean():.2%}")
    logger.info(f"  Mediana: {df_with_consensus['consensus_score'].median():.2%}")
    logger.info(f"  Desvio padrão: {df_with_consensus['consensus_score'].std():.2%}")
    
    # Distribuição por nível
    levels = df_with_consensus['consensus_level'].value_counts()
    logger.info("Distribuição por nível:")
    for level, count in levels.items():
        logger.info(f"  {level}: {count} ({count/len(df_with_consensus):.1%})")
    
    return df_with_consensus, analyzer, levels


def generate_report(analyzer: ConsensusEvaluator, df_with_consensus: pd.DataFrame):
    """Gera relatório completo de consenso"""
    logger.info("Gerando relatório completo de consenso...")
    report = analyzer.generate_consensus_report(df=df_with_consensus)
    logger.success("✓ Relatório gerado")
    return report


def create_visualizations(visualizer: ConsensusVisualizer, df_with_consensus: pd.DataFrame, 
                         levels: pd.Series, report: dict):
    """Cria visualizações"""
    logger.info("\n📊 Gerando visualizações...")
    
    # Score e níveis
    visualizer.plot_score_and_levels(
        df_with_consensus=df_with_consensus,
        levels=levels
    )
    
    # Heatmap de concordância
    logger.info("\n📊 Gerando heatmap de concordância...")
    visualizer.plot_agreement_heatmap(
        agreement_df=report['pairwise_agreement'],
        title='Matriz de Concordância entre Modelos',
    )
    
    # Cohen's Kappa
    logger.info("\n📊 Gerando heatmap de Cohen's Kappa...")
    visualizer.plot_kappa_heatmap(
        kappa_df=report['cohens_kappa']
    )


def evaluate_ground_truth(analyzer: ConsensusEvaluator, visualizer: ConsensusVisualizer,
                         df_with_consensus: pd.DataFrame, categories: list, ground_truth: list):
    """Avalia contra ground truth se disponível"""
    if not ground_truth:
        logger.info("⚠️ Ground truth não disponível – pulando validação")
        return None, None, None
    
    logger.info("Validando com ground truth...")
    accuracy, cls_report, cm = analyzer.evaluate_ground_truth(
        df_with_consensus=df_with_consensus
    )
    
    visualizer.plot_confusion_matrix(
        cm=cm,
        categories=categories
    )
    
    return accuracy, cls_report, cm


def show_problematic_cases(report: dict):
    """Exibe casos problemáticos"""
    problematic = report.get('problematic_cases')
    if problematic is not None and len(problematic) > 0:
        logger.warning(f"\n⚠️  {len(problematic)} casos problemáticos identificados")
        print("\nCasos problemáticos (primeiros 10):")
        print(problematic.head(10).to_string())
    else:
        logger.success("\n✓ Nenhum caso problemático identificado")


def export_results(df_with_consensus: pd.DataFrame, annotator: LLMAnnotator, 
                  dataset_name: str, texts: list, categories: list, 
                  ground_truth: list, num_repetitions: int, report: dict,
                  cls_report: dict = None):
    """Exporta todos os resultados"""
    logger.info("\nExportando resultados...")
    
    results_dir = annotator.results_dir / "summary"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # CSVs
    df_with_consensus.to_csv(results_dir / 'dataset_anotado_completo.csv', index=False)
    logger.info(f"✓ Salvos: {len(df_with_consensus)} registros")
    
    # Alta confiança
    high_conf = df_with_consensus[df_with_consensus['consensus_score'] >= 0.8]
    high_conf.to_csv(results_dir / 'alta_confianca.csv', index=False)
    logger.info(f"✓ Alta confiança: {len(high_conf)} registros")
    
    # Necessita revisão
    low_conf = df_with_consensus[df_with_consensus['consensus_score'] < 0.8]
    low_conf.to_csv(results_dir / 'necessita_revisao.csv', index=False)
    logger.info(f"✓ Necessita revisão: {len(low_conf)} registros")
    
    # Sumário JSON
    summary = {
        'dataset': {
            'name': dataset_name,
            'total_texts': len(texts),
            'categories': categories,
            'has_ground_truth': ground_truth is not None
        },
        'config': {
            'models': EXPERIMENT_CONFIG["default_models"],
            'total_models': len(annotator.models),
            'use_alternative_params': annotator.use_alternative_params,
            'num_repetitions': num_repetitions,
            'total_annotations': len(texts) * len(annotator.models) * num_repetitions
        },
        'results': {
            'consensus_mean': float(df_with_consensus['consensus_score'].mean()),
            'consensus_median': float(df_with_consensus['consensus_score'].median()),
            'high_consensus': int((df_with_consensus['consensus_level'] == 'high').sum()),
            'medium_consensus': int((df_with_consensus['consensus_level'] == 'medium').sum()),
            'low_consensus': int((df_with_consensus['consensus_level'] == 'low').sum()),
        },
        'metrics': {
            'fleiss_kappa': float(report['fleiss_kappa']),
            'fleiss_interpretation': report['fleiss_interpretation']
        }
    }
    
    if ground_truth and cls_report:
        summary['validation'] = {
            'classification_report': cls_report
        }
    
    with open(results_dir / 'sumario_experimento.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.success("✓ Resultados exportados com sucesso!")
    return results_dir


def print_final_summary(dataset_name: str, texts: list, categories: list, 
                       annotator: LLMAnnotator, num_repetitions: int,
                       df_with_consensus: pd.DataFrame, report: dict, 
                       results_dir: Path, accuracy: float = None, ground_truth: list = None):
    """Imprime resumo final"""
    logger.info("\n" + "="*80)
    logger.success("RESUMO DO EXPERIMENTO")
    logger.info("="*80)
    
    logger.info(f"\n📊 Dataset: {dataset_name}")
    logger.info(f"  Textos: {len(texts)}")
    logger.info(f"  Categorias: {len(categories)}")
    
    logger.info(f"\n🤖 Configuração:")
    logger.info(f"  Modelos base: {len(EXPERIMENT_CONFIG["default_models"])}")
    logger.info(f"  Total modelos: {len(annotator.models)}")
    logger.info(f"  Alternative params: {annotator.use_alternative_params}")
    logger.info(f"  Repetições: {num_repetitions}")
    
    logger.info(f"\n📈 Consenso:")
    logger.info(f"  Média: {df_with_consensus['consensus_score'].mean():.2%}")
    logger.info(f"  Fleiss' Kappa: {report['fleiss_kappa']:.3f} ({report['fleiss_interpretation']})")
    
    if ground_truth and accuracy is not None:
        logger.info(f"\n🎯 Validação:")
        logger.info(f"  Accuracy: {accuracy:.2%}")
    
    logger.info(f"\n📁 Arquivos gerados em: {results_dir}/")
    
    cache_stats = annotator.get_cache_stats()
    logger.info(f"\n💾 Cache: {cache_stats['total_entries']} entradas")
    
    logger.success("\n✅ Análise completa!")


def main():
    """Função principal"""
    # Configuração
    setup_logger()
    
    # Parâmetros
    dataset_name = "sst1"  # Ajuste conforme necessário
    num_repetitions = EXPERIMENT_CONFIG["num_repetitions_per_llm"]
    
    # 1. Carregar dataset
    texts, categories, ground_truth = load_dataset(dataset_name)
    
    # 2. Inicializar annotator
    annotator = initialize_annotator(dataset_name, categories)
    
    # 3. Executar anotação
    df_annotations = run_annotation(
        annotator, texts, num_repetitions
    )
    
    # 4. Métricas por modelo
    df_metrics = calculate_model_metrics(annotator, df_annotations, ground_truth)
    
    # 5. Calcular consenso
    df_with_consensus, analyzer, levels = compute_consensus(
        df_annotations, categories, annotator.results_dir
    )
    
    # 6. Gerar relatório
    report = generate_report(analyzer, df_with_consensus)
    
    # 7. Visualizações
    visualizer = ConsensusVisualizer(output_dir=annotator.results_dir)
    create_visualizations(visualizer, df_with_consensus, levels, report)
    
    # 8. Casos problemáticos
    show_problematic_cases(report)
    
    # 9. Validação com ground truth
    accuracy, cls_report, cm = evaluate_ground_truth(
        analyzer, visualizer, df_with_consensus, categories, ground_truth
    )
    
    # 10. Exportar resultados
    results_dir = export_results(
        df_with_consensus, annotator, dataset_name, texts, categories,
        ground_truth, num_repetitions, report, cls_report
    )
    
    # 11. Resumo final
    print_final_summary(
        dataset_name, texts, categories, annotator, num_repetitions,
        df_with_consensus, report, results_dir, accuracy, ground_truth
    )


if __name__ == "__main__":
    main()