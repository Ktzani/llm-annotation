"""
Main - Sistema de Anotação LLM com Modelos Open-Source
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

from src.llm_annotation_system.llm_annotator import LLMAnnotator
from src.llm_annotation_system.consensus_analyzer import ConsensusAnalyzer
from src.utils.data_loader import load_hf_dataset, list_available_datasets


def exemplo_teste():
    """Exemplo rápido com dados de teste"""
    
    logger.info("Modo: Exemplo com dados de teste")
    
    # Modelos open-source
    models = [
        "llama3-8b",      # Meta Llama 3 8B
        "mistral-7b",     # Mistral 7B
        "qwen2-7b",       # Qwen 2 7B (excelente PT-BR)
    ]
    
    categories = ["Positivo", "Negativo", "Neutro"]
    
    # Dados de teste
    texts = [
        "Este produto é excelente! Recomendo muito.",
        "Péssima qualidade, não vale o preço.",
        "O produto é ok, nada de especial.",
        "Adorei! Melhor compra que já fiz.",
        "Não funcionou como esperado.",
    ]
    
    # Anotar
    logger.info(f"Modelos: {models}")
    logger.info(f"Textos: {len(texts)}")
    
    annotator = LLMAnnotator(
        models=models,
        categories=categories,
        api_keys=None,  # Open-source não precisa
        use_langchain_cache=True
    )
    
    df = annotator.annotate_dataset(texts, num_repetitions=3)
    df = annotator.calculate_consensus(df)
    
    # Análise
    analyzer = ConsensusAnalyzer(categories)
    consensus_cols = [col for col in df.columns if '_consensus' in col and '_score' not in col]
    report = analyzer.generate_consensus_report(df, consensus_cols, "./results")
    
    # Resultados
    logger.success("\nResultados:")
    print(df[['text', 'most_common_annotation', 'consensus_score']].to_string())
    logger.info(f"\nFleiss' Kappa: {report['fleiss_kappa']:.3f} ({report['fleiss_interpretation']})")


def usar_huggingface(dataset_name: str):
    """Usa dataset do HuggingFace"""
    
    logger.info(f"Modo: Dataset HuggingFace - {dataset_name}")
    
    # Modelos open-source
    models = [
        "llama3-8b",
        "mistral-7b",
        "qwen2-7b",
    ]
    
    # Carregar dataset
    texts, categories, ground_truth = load_hf_dataset(dataset_name)
    
    logger.info(f"Modelos: {models}")
    logger.info(f"Textos: {len(texts)}")
    logger.info(f"Categorias: {categories}")
    logger.info(f"Ground truth: {'Sim' if ground_truth else 'Não'}")
    
    # Anotar
    annotator = LLMAnnotator(
        models=models,
        categories=categories,
        api_keys=None,
        use_langchain_cache=True
    )
    
    df = annotator.annotate_dataset(texts, num_repetitions=3)
    
    # Adicionar ground truth
    if ground_truth:
        df['ground_truth'] = ground_truth
    
    df = annotator.calculate_consensus(df)
    
    # Análise
    analyzer = ConsensusAnalyzer(categories)
    consensus_cols = [col for col in df.columns if '_consensus' in col and '_score' not in col]
    report = analyzer.generate_consensus_report(df, consensus_cols, "./results")
    
    # Resultados
    logger.success("\nResultados:")
    print(df[['text', 'most_common_annotation', 'consensus_score', 'consensus_level']].head(10).to_string())
    
    logger.info(f"\nFleiss' Kappa: {report['fleiss_kappa']:.3f} ({report['fleiss_interpretation']})")
    
    # Validação
    if ground_truth:
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(df['ground_truth'], df['most_common_annotation'])
        logger.success(f"\nAccuracy vs Ground Truth: {accuracy:.2%}")
        
        logger.info("\nClassification Report:")
        print(classification_report(df['ground_truth'], df['most_common_annotation']))


def main():
    """Main com argumentos"""
    
    parser = argparse.ArgumentParser(description='Sistema de Anotação LLM - Open Source')
    parser.add_argument('--mode', choices=['teste', 'huggingface', 'list'], default='teste',
                        help='Modo de execução')
    parser.add_argument('--dataset', type=str, default='agnews',
                        help='Nome do dataset (para modo huggingface)')
    
    args = parser.parse_args()
    
    if args.mode == 'list':
        logger.info("Datasets disponíveis:")
        for dataset in list_available_datasets():
            logger.info(f"  - {dataset}")
        return
    
    if args.mode == 'teste':
        exemplo_teste()
    elif args.mode == 'huggingface':
        usar_huggingface(args.dataset)


if __name__ == "__main__":
    main()
