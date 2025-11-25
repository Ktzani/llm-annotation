"""
Main HuggingFace - Sistema de Anotação com Datasets HuggingFace (waashk)

Modelos: Apenas Open-Source (Llama 3, Mistral, Qwen 2)
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Configurar logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / 'llm_annotation_system'))
sys.path.insert(0, str(Path(__file__).parent / 'config'))
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from llm_annotator_refactored import LLMAnnotator
from consensus_analyzer_refactored import ConsensusAnalyzer
from data_loader import load_hf_dataset, list_available_datasets, discover_dataset_structure


def executar_anotacao(
    dataset_name: str,
    use_alternative_params: bool = False,
    num_repetitions: int = 3
):
    """
    Executa anotação completa de um dataset
    
    Args:
        dataset_name: Nome do dataset
        use_alternative_params: Se True, usa variações de temperatura
        num_repetitions: Número de repetições por modelo
    """
    
    logger.info(f"Dataset: {dataset_name}")
    
    # Modelos open-source
    models = [
        "llama3-8b",      # Meta Llama 3 8B
        "mistral-7b",     # Mistral 7B
        "qwen2-7b",       # Qwen 2 7B (excelente PT-BR)
    ]
    
    # Carregar dataset
    logger.info("Carregando dataset do HuggingFace...")
    texts, categories, ground_truth = load_hf_dataset(dataset_name)
    
    logger.info(f"Textos: {len(texts)}")
    logger.info(f"Categorias: {categories}")
    logger.info(f"Ground truth: {'Sim' if ground_truth else 'Não'}")
    
    # Inicializar anotador
    logger.info("Inicializando anotador...")
    annotator = LLMAnnotator(
        models=models,
        categories=categories,
        api_keys=None,  # Open-source não precisa
        use_langchain_cache=True,
        use_alternative_params=use_alternative_params
    )
    
    if use_alternative_params:
        logger.warning(f"Alternative params ativado: {len(annotator.models)} variações")
    
    # Anotar
    logger.info("Iniciando anotação...")
    logger.info(f"  Modelos: {len(annotator.models)}")
    logger.info(f"  Repetições: {num_repetitions}")
    logger.info(f"  Total anotações: {len(texts) * len(annotator.models) * num_repetitions}")
    
    df = annotator.annotate_dataset(texts, num_repetitions=num_repetitions)
    
    # Adicionar ground truth
    if ground_truth:
        df['ground_truth'] = ground_truth
    
    # Calcular consenso
    logger.info("Calculando consenso...")
    df = annotator.calculate_consensus(df)
    
    # Análise
    logger.info("Gerando relatório de consenso...")
    analyzer = ConsensusAnalyzer(categories)
    consensus_cols = [col for col in df.columns if '_consensus' in col and '_score' not in col]
    
    report = analyzer.generate_consensus_report(
        df=df,
        annotator_cols=consensus_cols,
        output_dir="./results"
    )
    
    # Resultados
    logger.success("\n" + "="*80)
    logger.success("RESULTADOS")
    logger.success("="*80 + "\n")
    
    logger.info("📊 Consenso:")
    logger.info(f"  Média: {df['consensus_score'].mean():.2%}")
    logger.info(f"  Alto (≥80%): {(df['consensus_score'] >= 0.8).sum()}/{len(df)}")
    logger.info(f"  Médio (60-80%): {((df['consensus_score'] >= 0.6) & (df['consensus_score'] < 0.8)).sum()}/{len(df)}")
    logger.info(f"  Baixo (<60%): {(df['consensus_score'] < 0.6).sum()}/{len(df)}")
    logger.info(f"  Problemáticos: {df['is_problematic'].sum()}/{len(df)}")
    
    logger.info(f"\n📈 Fleiss' Kappa: {report['fleiss_kappa']:.3f} ({report['fleiss_interpretation']})")
    
    # Validação com ground truth
    if ground_truth:
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(df['ground_truth'], df['most_common_annotation'])
        
        logger.success(f"\n🎯 Validação vs Ground Truth:")
        logger.success(f"  Accuracy: {accuracy:.2%}")
        
        logger.info("\nClassification Report:")
        print(classification_report(df['ground_truth'], df['most_common_annotation']))
        
        # Accuracy por nível de consenso
        logger.info("\nAccuracy por nível de consenso:")
        for level in ['high', 'medium', 'low']:
            df_level = df[df['consensus_level'] == level]
            if len(df_level) > 0:
                acc = accuracy_score(df_level['ground_truth'], df_level['most_common_annotation'])
                logger.info(f"  {level}: {acc:.2%} ({len(df_level)} casos)")
    
    # Salvar
    logger.info("\n💾 Salvando resultados:")
    
    # Dataset completo
    output_file = Path("./results") / f"{dataset_name}_anotado.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"  ✓ {output_file}")
    
    # Alta confiança
    high_conf = df[df['consensus_score'] >= 0.8]
    if len(high_conf) > 0:
        high_file = Path("./results") / f"{dataset_name}_alta_confianca.csv"
        high_conf.to_csv(high_file, index=False)
        logger.info(f"  ✓ {high_file} ({len(high_conf)} registros)")
    
    # Baixa confiança
    low_conf = df[df['consensus_score'] < 0.6]
    if len(low_conf) > 0:
        low_file = Path("./results") / f"{dataset_name}_revisar.csv"
        low_conf.to_csv(low_file, index=False)
        logger.info(f"  ✓ {low_file} ({len(low_conf)} registros)")
    
    # Cache stats
    cache_stats = annotator.get_cache_stats()
    logger.info(f"\n💾 Cache: {cache_stats['total_entries']} entradas")
    
    logger.success("\n✅ Processamento completo!")


def modo_listar():
    """Lista datasets disponíveis"""
    
    logger.info("Datasets disponíveis em src/config/datasets.py:")
    
    for dataset in list_available_datasets():
        logger.info(f"  - {dataset}")
    
    logger.info("\nPara adicionar novos datasets:")
    logger.info("  1. Edite src/config/datasets.py")
    logger.info("  2. Adicione configuração em HUGGINGFACE_DATASETS")
    logger.info("  3. Execute: python src/main.py --mode huggingface --dataset SEU_DATASET")


def modo_descobrir(dataset_path: str):
    """
    Descobre estrutura de um dataset
    
    Args:
        dataset_path: Path do dataset (ex: waashk/agnews)
    """
    
    logger.info(f"Descobrindo estrutura: {dataset_path}")
    
    try:
        info = discover_dataset_structure(dataset_path)
        
        logger.success("\n✓ Estrutura descoberta:")
        logger.info(f"  Colunas: {info['columns']}")
        logger.info(f"  Registros: {info['num_rows']}")
        
        logger.info("\nAmostra:")
        print(info['sample'])
        
        logger.info("\n💡 Adicione esta configuração em src/config/datasets.py:")
        logger.info(f'''
HUGGINGFACE_DATASETS = {{
    "{dataset_path.split('/')[-1]}": {{
        "path": "{dataset_path}",
        "text_column": "AJUSTE_AQUI",  # Ex: "text"
        "label_column": "AJUSTE_AQUI",  # Ex: "label" ou None
        "categories": None,  # Auto-extrai se tiver labels
        "description": "Descrição do dataset"
    }},
}}
''')
    
    except Exception as e:
        logger.error(f"Erro ao descobrir estrutura: {e}")


def modo_multiplos(datasets: list, use_alternative_params: bool = False):
    """
    Processa múltiplos datasets
    
    Args:
        datasets: Lista de nomes de datasets
        use_alternative_params: Se True, usa variações
    """
    
    logger.info(f"Processando {len(datasets)} datasets")
    
    for i, dataset_name in enumerate(datasets, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Dataset {i}/{len(datasets)}: {dataset_name}")
        logger.info(f"{'='*80}\n")
        
        try:
            executar_anotacao(
                dataset_name=dataset_name,
                use_alternative_params=use_alternative_params
            )
            logger.success(f"✓ {dataset_name} processado\n")
        
        except Exception as e:
            logger.error(f"✗ Erro em {dataset_name}: {e}\n")
            continue


def main():
    """Main com argumentos"""
    
    parser = argparse.ArgumentParser(
        description='Sistema de Anotação LLM - HuggingFace Datasets'
    )
    
    parser.add_argument(
        '--modo',
        choices=['anotar', 'listar', 'descobrir', 'multiplos'],
        default='anotar',
        help='Modo de execução'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='agnews',
        help='Nome do dataset (modo anotar/descobrir)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Lista de datasets (modo multiplos)'
    )
    
    parser.add_argument(
        '--alternative-params',
        action='store_true',
        help='Usar alternative params (temp=0, 0.3, 0.5)'
    )
    
    parser.add_argument(
        '--repetitions',
        type=int,
        default=3,
        help='Número de repetições por modelo'
    )
    
    args = parser.parse_args()
    
    # Executar modo
    if args.modo == 'anotar':
        executar_anotacao(
            dataset_name=args.dataset,
            use_alternative_params=args.alternative_params,
            num_repetitions=args.repetitions
        )
    
    elif args.modo == 'listar':
        modo_listar()
    
    elif args.modo == 'descobrir':
        if not args.dataset:
            logger.error("Especifique --dataset")
            return
        modo_descobrir(args.dataset)
    
    elif args.modo == 'multiplos':
        if not args.datasets:
            logger.error("Especifique --datasets")
            return
        modo_multiplos(
            datasets=args.datasets,
            use_alternative_params=args.alternative_params
        )


if __name__ == "__main__":
    main()
