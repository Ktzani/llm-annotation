"""
Exemplo de uso do sistema com datasets do HuggingFace (waashk)

Este script demonstra como carregar seus datasets do HuggingFace
e executar o sistema de anotação automática.

Execute: python src/main_huggingface.py
"""

import sys
import os
from pathlib import Path

# Adicionar diretórios ao path
sys.path.insert(0, str(Path(__file__).parent / "llm_annotation_system"))
sys.path.insert(0, str(Path(__file__).parent / "config"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from llm_annotator import LLMAnnotator
from consensus_analyzer import ConsensusAnalyzer
from visualizer import ConsensusVisualizer
from dataset_config import (
    load_hf_dataset,
    load_hf_dataset_as_dataframe,
    list_available_datasets,
    discover_dataset_structure,
    save_annotated_dataset
)
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()


def main_exemplo_basico():
    """
    Exemplo 1: Carregar dataset do HuggingFace e anotar
    """
    print("\n" + "="*80)
    print(" " * 20 + "ANOTAÇÃO COM DATASETS HUGGINGFACE")
    print("="*80)
    
    # 1. LISTAR DATASETS DISPONÍVEIS
    print("\n1. Datasets configurados:")
    for ds_name in list_available_datasets():
        print(f"   • {ds_name}")
    
    # 2. CARREGAR DATASET
    print("\n2. Carregando dataset...")
    
    # OPÇÃO A: Usar dataset pré-configurado
    dataset_name = "exemplo_com_labels"  # AJUSTE para seu dataset
    
    # OPÇÃO B: Ou descobrir estrutura primeiro (descomente)
    # discover_dataset_structure("waashk/seu-dataset")
    # return
    
    texts, categories, ground_truth = load_hf_dataset(dataset_name)
    
    # 3. CONFIGURAR API KEYS
    print("\n3. Configurando modelos...")
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", "sua-api-key"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", "sua-api-key"),
        "google": os.getenv("GOOGLE_API_KEY", "sua-api-key"),
    }
    
    models = [
        "gpt-4-turbo",
        "claude-3-opus",
        "gemini-pro",
    ]
    
    # 4. INICIALIZAR ANOTADOR
    print("\n4. Inicializando anotador...")
    annotator = LLMAnnotator(
        models=models,
        categories=categories,
        api_keys=api_keys,
        cache_dir="./cache",
        results_dir="./results"
    )
    
    # 5. ANOTAR DATASET
    print("\n5. Iniciando anotação...")
    print(f"   → {len(texts)} textos")
    print(f"   → {len(models)} modelos")
    print(f"   → 3 repetições por modelo")
    
    df_annotations = annotator.annotate_dataset(
        texts=texts,
        num_repetitions=3,
        test_param_variations=False,
    )
    
    # 6. CALCULAR CONSENSO
    print("\n6. Calculando consenso...")
    df_with_consensus = annotator.calculate_consensus(df_annotations)
    
    # 7. ADICIONAR GROUND TRUTH (se disponível)
    if ground_truth:
        print("\n7. Adicionando ground truth para validação...")
        df_with_consensus['ground_truth'] = ground_truth
        
        # Calcular accuracy
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(
            df_with_consensus['ground_truth'],
            df_with_consensus['most_common_annotation']
        )
        
        print(f"\n📊 VALIDAÇÃO COM GROUND TRUTH:")
        print(f"   Accuracy do consenso: {accuracy:.2%}")
        
        print(f"\n   Relatório detalhado:")
        print(classification_report(
            df_with_consensus['ground_truth'],
            df_with_consensus['most_common_annotation'],
            target_names=categories
        ))
    
    # 8. ANÁLISE DETALHADA
    print("\n8. Gerando análise de consenso...")
    analyzer = ConsensusAnalyzer(categories=categories)
    
    consensus_cols = [col for col in df_with_consensus.columns 
                      if '_consensus' in col and '_score' not in col]
    
    report = analyzer.generate_consensus_report(
        df=df_with_consensus,
        annotator_cols=consensus_cols,
        output_dir="./results"
    )
    
    # 9. VISUALIZAÇÕES
    print("\n9. Gerando visualizações...")
    visualizer = ConsensusVisualizer(output_dir="./results/figures")
    
    visualizer.plot_agreement_heatmap(report['pairwise_agreement'])
    visualizer.plot_consensus_distribution(df_with_consensus)
    visualizer.plot_model_comparison(df_with_consensus, models=models)
    visualizer.create_interactive_dashboard(df_with_consensus, report)
    
    # 10. SALVAR RESULTADOS
    print("\n10. Salvando resultados...")
    save_annotated_dataset(
        df_with_consensus,
        output_path="./results/dataset_anotado_final.csv"
    )
    
    # 11. SUMÁRIO FINAL
    print("\n" + "="*80)
    print(" " * 30 + "SUMÁRIO FINAL")
    print("="*80)
    
    print(f"\n📊 Estatísticas:")
    print(f"   • Total de textos: {len(df_with_consensus)}")
    print(f"   • Modelos usados: {len(models)}")
    print(f"   • Categorias: {len(categories)}")
    
    print(f"\n🎯 Consenso:")
    print(f"   • Consenso médio: {df_with_consensus['consensus_score'].mean():.2%}")
    print(f"   • Alto consenso (≥80%): {(df_with_consensus['consensus_score'] >= 0.8).sum()}")
    print(f"   • Casos problemáticos: {df_with_consensus['is_problematic'].sum()}")
    
    if 'mean_cohen_kappa' in report['distance_metrics']:
        print(f"\n📈 Métricas:")
        print(f"   • Cohen's Kappa: {report['distance_metrics']['mean_cohen_kappa']:.4f}")
        print(f"   • Fleiss' Kappa: {report['distance_metrics']['fleiss_kappa']:.4f}")
    
    if ground_truth:
        print(f"\n✅ Validação:")
        print(f"   • Accuracy vs Ground Truth: {accuracy:.2%}")
    
    print(f"\n📁 Arquivos salvos:")
    print(f"   • Dataset anotado: ./results/dataset_anotado_final.csv")
    print(f"   • Relatórios: ./results/")
    print(f"   • Visualizações: ./results/figures/")
    print(f"   • Dashboard: ./results/figures/interactive_dashboard.html")
    
    print("\n" + "="*80)
    print("✅ Anotação completa!")
    print("="*80 + "\n")


def main_carregar_customizado():
    """
    Exemplo 2: Carregar dataset customizado sem pré-configurar
    """
    from dataset_config import load_custom_dataset
    
    print("\n" + "="*80)
    print(" " * 20 + "CARREGAR DATASET CUSTOMIZADO")
    print("="*80 + "\n")
    
    # Carregar diretamente
    texts, categories, labels = load_custom_dataset(
        hf_path="waashk/seu-dataset",  # AJUSTE
        text_column="text",             # AJUSTE
        label_column="label",           # AJUSTE ou None
        categories=None,                # ou defina manualmente
        combine_splits=["train", "test"],  # Combinar splits
        sample_size=100                 # Amostra pequena primeiro
    )
    
    print(f"✅ Dataset carregado:")
    print(f"   • {len(texts)} textos")
    print(f"   • Categorias: {categories}")
    
    # Continuar com anotação...
    # (mesmo código do exemplo anterior)


def main_descobrir_estrutura():
    """
    Exemplo 3: Descobrir estrutura de um dataset
    """
    print("\n" + "="*80)
    print(" " * 20 + "DESCOBRIR ESTRUTURA DO DATASET")
    print("="*80 + "\n")
    
    # Substitua pelo seu dataset
    hf_path = "waashk/seu-dataset"  # AJUSTE
    
    discover_dataset_structure(hf_path, num_examples=3)
    
    print("\n💡 Dica: Use a sugestão de configuração acima")
    print("   e adicione em src/config/dataset_config.py")


def main_multiplos_datasets():
    """
    Exemplo 4: Processar múltiplos datasets
    """
    print("\n" + "="*80)
    print(" " * 20 + "PROCESSAR MÚLTIPLOS DATASETS")
    print("="*80 + "\n")
    
    # Lista de datasets para processar
    dataset_names = [
        "exemplo_com_labels",
        "exemplo_sem_labels",
        # Adicione mais...
    ]
    
    for dataset_name in dataset_names:
        try:
            print(f"\n{'='*80}")
            print(f"Processando: {dataset_name}")
            print(f"{'='*80}\n")
            
            texts, categories, labels = load_hf_dataset(dataset_name)
            
            # Processar cada dataset...
            # (adicione código de anotação aqui)
            
            print(f"✅ {dataset_name} processado com sucesso!\n")
            
        except Exception as e:
            print(f"❌ Erro em {dataset_name}: {str(e)}\n")
            continue


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sistema de Anotação com Datasets HuggingFace"
    )
    parser.add_argument(
        "--modo",
        type=str,
        choices=["basico", "customizado", "descobrir", "multiplos"],
        default="basico",
        help="Modo de execução"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Nome do dataset (para modo descobrir)"
    )
    
    args = parser.parse_args()
    
    if args.modo == "basico":
        main_exemplo_basico()
    
    elif args.modo == "customizado":
        main_carregar_customizado()
    
    elif args.modo == "descobrir":
        if args.dataset:
            from dataset_config import discover_dataset_structure
            discover_dataset_structure(args.dataset)
        else:
            print("❌ Erro: especifique --dataset para descobrir")
            print("Exemplo: python src/main_huggingface.py --modo descobrir --dataset waashk/seu-dataset")
    
    elif args.modo == "multiplos":
        main_multiplos_datasets()
