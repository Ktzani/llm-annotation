"""
Exemplo de uso simplificado do sistema de anotação com LLMs
Execute este script para um teste rápido
"""
from llm_annotator import LLMAnnotator
from consensus_analyzer import ConsensusAnalyzer
from visualizer import ConsensusVisualizer

def main():
    """Exemplo de uso completo do sistema"""
    
    print("\n" + "="*80)
    print(" " * 20 + "SISTEMA DE ANOTAÇÃO AUTOMÁTICA COM LLMS")
    print("="*80)
    
    # 1. Configuração
    print("\n1. Configurando sistema...")
    
    # API Keys (SUBSTITUA PELOS SEUS)
    api_keys = {
        "openai": "sua-api-key-aqui",
        "anthropic": "sua-api-key-aqui",
        "google": "sua-api-key-aqui",
    }
    
    # Modelos a usar
    models = [
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "gemini-pro",
    ]
    
    # Categorias
    categories = ["Positivo", "Negativo", "Neutro"]
    
    # Textos de exemplo
    texts = [
        "Este produto é excelente! Recomendo muito.",
        "Péssima qualidade, não funciona como esperado.",
        "O produto é ok, nada de especial.",
        "Maravilhoso! Superou minhas expectativas.",
        "Horrível, totalmente decepcionado.",
        "Funciona bem, mas o preço poderia ser melhor.",
        "Adorei! Voltaria a comprar com certeza.",
        "Não vale o dinheiro investido.",
        "É razoável para o preço.",
        "Esperava mais, mas não é ruim.",
    ]
    
    # 2. Inicializar anotador
    print("2. Inicializando anotador...")
    annotator = LLMAnnotator(
        models=models,
        categories=categories,
        api_keys=api_keys,
        cache_dir="./cache",
        results_dir="./results"
    )
    
    # 3. Anotar dataset
    print("\n3. Anotando dataset...")
    df_annotations = annotator.annotate_dataset(
        texts=texts,
        num_repetitions=3,  # Cada LLM anota 3 vezes
        test_param_variations=False,  # Mudar para True para testar variações
    )
    
    # 4. Calcular consenso
    print("\n4. Calculando consenso...")
    df_with_consensus = annotator.calculate_consensus(df_annotations)
    
    # 5. Análise detalhada
    print("\n5. Gerando análise detalhada...")
    analyzer = ConsensusAnalyzer(categories=categories)
    
    consensus_cols = [col for col in df_with_consensus.columns 
                      if '_consensus' in col and '_score' not in col]
    
    report = analyzer.generate_consensus_report(
        df=df_with_consensus,
        annotator_cols=consensus_cols,
        output_dir="./results"
    )
    
    # 6. Visualizações
    print("\n6. Gerando visualizações...")
    visualizer = ConsensusVisualizer(output_dir="./results/figures")
    
    visualizer.plot_agreement_heatmap(
        report['pairwise_agreement'],
        title="Concordância entre Modelos LLM"
    )
    
    visualizer.plot_consensus_distribution(df_with_consensus)
    
    if 'disagreement_patterns' in report:
        visualizer.plot_confusion_matrix(
            report['disagreement_patterns']['confusion_matrix']
        )
    
    visualizer.plot_model_comparison(
        df_with_consensus,
        models=models
    )
    
    visualizer.create_interactive_dashboard(
        df_with_consensus,
        report
    )
    
    # 7. Sumário final
    print("\n" + "="*80)
    print(" " * 30 + "SUMÁRIO")
    print("="*80)
    
    print(f"\nTotal de textos anotados: {len(df_with_consensus)}")
    print(f"Modelos utilizados: {len(models)}")
    print(f"Repetições por modelo: 3")
    
    print(f"\nConsenso médio: {df_with_consensus['consensus_score'].mean():.2%}")
    print(f"Alto consenso (≥80%): {(df_with_consensus['consensus_score'] >= 0.8).sum()}")
    print(f"Casos problemáticos: {df_with_consensus['is_problematic'].sum()}")
    
    if 'mean_cohen_kappa' in report['distance_metrics']:
        print(f"\nCohen's Kappa médio: {report['distance_metrics']['mean_cohen_kappa']:.4f}")
    
    print("\n✓ Análise completa!")
    print("📁 Resultados salvos em: ./results/")
    print("📊 Visualizações em: ./results/figures/")
    print("🌐 Dashboard interativo: ./results/figures/interactive_dashboard.html")
    
    print("\n" + "="*80)
    
    # Salvar dataset final
    df_with_consensus.to_csv(
        "./results/annotated_dataset_final.csv",
        index=False,
        encoding='utf-8'
    )
    print("\n✓ Dataset anotado salvo: ./results/annotated_dataset_final.csv")


if __name__ == "__main__":
    main()
