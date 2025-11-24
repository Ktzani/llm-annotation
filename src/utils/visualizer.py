"""
Visualization Module - Visualizações para análise de consenso
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


class ConsensusVisualizer:
    """
    Classe para criar visualizações dos resultados de consenso
    """
    
    def __init__(self, output_dir: str = "./results/figures"):
        """
        Inicializa o visualizador
        
        Args:
            output_dir: Diretório para salvar figuras
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_agreement_heatmap(
        self,
        agreement_df: pd.DataFrame,
        title: str = "Matriz de Concordância entre Anotadores",
        save_name: str = "agreement_heatmap.png"
    ):
        """
        Plota heatmap de concordância entre anotadores
        
        Args:
            agreement_df: DataFrame com matriz de concordância
            title: Título do gráfico
            save_name: Nome do arquivo para salvar
        """
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            agreement_df,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Taxa de Concordância'}
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Anotador', fontsize=12)
        plt.ylabel('Anotador', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Heatmap de concordância salvo: {save_name}")
    
    def plot_consensus_distribution(
        self,
        df: pd.DataFrame,
        save_name: str = "consensus_distribution.png"
    ):
        """
        Plota distribuição de níveis de consenso
        
        Args:
            df: DataFrame com coluna 'consensus_score'
            save_name: Nome do arquivo para salvar
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histograma
        axes[0, 0].hist(df['consensus_score'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Score de Consenso')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].set_title('Distribuição de Scores de Consenso')
        axes[0, 0].axvline(df['consensus_score'].mean(), color='red', 
                           linestyle='--', label=f'Média: {df["consensus_score"].mean():.3f}')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(df['consensus_score'])
        axes[0, 1].set_ylabel('Score de Consenso')
        axes[0, 1].set_title('Box Plot - Scores de Consenso')
        axes[0, 1].set_xticklabels(['Todos'])
        
        # Gráfico de pizza por nível
        if 'consensus_level' in df.columns:
            level_counts = df['consensus_level'].value_counts()
            axes[1, 0].pie(level_counts, labels=level_counts.index, autopct='%1.1f%%',
                           colors=['#2ecc71', '#f39c12', '#e74c3c'])
            axes[1, 0].set_title('Distribuição por Nível de Consenso')
        
        # Estatísticas
        stats_text = f"""
        Estatísticas:
        
        Média: {df['consensus_score'].mean():.3f}
        Mediana: {df['consensus_score'].median():.3f}
        Desvio Padrão: {df['consensus_score'].std():.3f}
        Mínimo: {df['consensus_score'].min():.3f}
        Máximo: {df['consensus_score'].max():.3f}
        
        Consenso Alto (≥80%): {(df['consensus_score'] >= 0.8).sum()}
        Consenso Médio (60-80%): {((df['consensus_score'] >= 0.6) & (df['consensus_score'] < 0.8)).sum()}
        Consenso Baixo (<60%): {(df['consensus_score'] < 0.6).sum()}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                        verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        
        plt.suptitle('Análise de Distribuição de Consenso', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Distribuição de consenso salva: {save_name}")
    
    def plot_confusion_matrix(
        self,
        confusion_df: pd.DataFrame,
        save_name: str = "confusion_matrix.png"
    ):
        """
        Plota matriz de confusão agregada
        
        Args:
            confusion_df: DataFrame com matriz de confusão
            save_name: Nome do arquivo para salvar
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            confusion_df,
            annot=True,
            fmt='.0f',
            cmap='Blues',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Contagem'}
        )
        
        plt.title('Matriz de Confusão Agregada entre Anotadores', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Categoria Predita', fontsize=12)
        plt.ylabel('Categoria Real', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Matriz de confusão salva: {save_name}")
    
    def plot_model_comparison(
        self,
        df: pd.DataFrame,
        models: List[str],
        save_name: str = "model_comparison.png"
    ):
        """
        Compara performance e consistência dos modelos
        
        Args:
            df: DataFrame com anotações
            models: Lista de modelos
            save_name: Nome do arquivo para salvar
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Consenso interno de cada modelo
        consensus_cols = [f"{model}_consensus_score" for model in models if f"{model}_consensus_score" in df.columns]
        
        if consensus_cols:
            consensus_data = df[consensus_cols]
            consensus_data.columns = [col.replace('_consensus_score', '') for col in consensus_cols]
            
            # Box plot de consenso interno
            consensus_data.boxplot(ax=axes[0, 0])
            axes[0, 0].set_ylabel('Score de Consenso Interno')
            axes[0, 0].set_title('Consenso Interno por Modelo')
            axes[0, 0].set_xticklabels(consensus_data.columns, rotation=45, ha='right')
            
            # Média de consenso interno
            means = consensus_data.mean().sort_values(ascending=False)
            axes[0, 1].barh(range(len(means)), means.values)
            axes[0, 1].set_yticks(range(len(means)))
            axes[0, 1].set_yticklabels(means.index)
            axes[0, 1].set_xlabel('Consenso Interno Médio')
            axes[0, 1].set_title('Ranking de Consenso Interno')
            axes[0, 1].set_xlim(0, 1)
            
            for i, v in enumerate(means.values):
                axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Distribuição de categorias por modelo
        main_consensus_cols = [f"{model}_consensus" for model in models if f"{model}_consensus" in df.columns]
        
        if main_consensus_cols:
            # Contar categorias
            category_counts = {}
            for col in main_consensus_cols:
                model_name = col.replace('_consensus', '')
                counts = df[col].value_counts()
                category_counts[model_name] = counts
            
            # Criar DataFrame para plot
            cat_df = pd.DataFrame(category_counts).fillna(0)
            
            # Stacked bar chart
            cat_df.T.plot(kind='bar', stacked=True, ax=axes[1, 0])
            axes[1, 0].set_ylabel('Contagem')
            axes[1, 0].set_title('Distribuição de Categorias por Modelo')
            axes[1, 0].set_xlabel('Modelo')
            axes[1, 0].legend(title='Categoria', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Diversidade de anotações
        if 'unique_annotations' in df.columns:
            diversity_by_model = []
            for model in models:
                rep_cols = [col for col in df.columns if col.startswith(f"{model}_rep")]
                if rep_cols:
                    # Calcular diversidade: quantas categorias diferentes o modelo usou
                    unique_per_instance = df[rep_cols].nunique(axis=1)
                    diversity_by_model.append({
                        'model': model,
                        'mean_diversity': unique_per_instance.mean(),
                        'std_diversity': unique_per_instance.std()
                    })
            
            if diversity_by_model:
                div_df = pd.DataFrame(diversity_by_model).sort_values('mean_diversity')
                
                axes[1, 1].barh(range(len(div_df)), div_df['mean_diversity'].values)
                axes[1, 1].set_yticks(range(len(div_df)))
                axes[1, 1].set_yticklabels(div_df['model'].values)
                axes[1, 1].set_xlabel('Diversidade Média de Respostas')
                axes[1, 1].set_title('Diversidade Interna por Modelo\n(menor = mais consistente)')
                
                for i, (mean, std) in enumerate(zip(div_df['mean_diversity'].values, div_df['std_diversity'].values)):
                    axes[1, 1].text(mean + 0.02, i, f'{mean:.2f}±{std:.2f}', va='center')
        
        plt.suptitle('Comparação entre Modelos', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparação de modelos salva: {save_name}")
    
    def plot_parameter_impact(
        self,
        df: pd.DataFrame,
        model: str,
        save_name: str = None
    ):
        """
        Visualiza impacto de variações de parâmetros
        
        Args:
            df: DataFrame com anotações
            model: Nome do modelo
            save_name: Nome do arquivo para salvar
        """
        if save_name is None:
            save_name = f"{model}_parameter_impact.png"
        
        # Encontrar colunas de variações de parâmetros
        param_cols = [col for col in df.columns if col.startswith(f"{model}_param_var")]
        consensus_col = f"{model}_consensus"
        
        if not param_cols or consensus_col not in df.columns:
            print(f"⚠ Dados de variação de parâmetros não encontrados para {model}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Concordância com baseline (consensus)
        agreements = []
        for col in param_cols:
            agreement = (df[col] == df[consensus_col]).sum() / len(df)
            agreements.append({
                'variation': col.replace(f"{model}_param_var", "Var "),
                'agreement': agreement
            })
        
        agree_df = pd.DataFrame(agreements)
        
        # Bar chart de concordância
        axes[0, 0].bar(range(len(agree_df)), agree_df['agreement'].values)
        axes[0, 0].set_xticks(range(len(agree_df)))
        axes[0, 0].set_xticklabels(agree_df['variation'].values)
        axes[0, 0].set_ylabel('Taxa de Concordância com Baseline')
        axes[0, 0].set_title(f'Impacto de Variações de Parâmetros - {model}')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axhline(y=0.8, color='r', linestyle='--', label='Threshold 80%')
        axes[0, 0].legend()
        
        for i, v in enumerate(agree_df['agreement'].values):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Distribuição de categorias por variação
        all_cols = [consensus_col] + param_cols
        category_dist = {}
        
        for col in all_cols:
            col_name = col.replace(f"{model}_", "").replace("_", " ").title()
            category_dist[col_name] = df[col].value_counts()
        
        cat_dist_df = pd.DataFrame(category_dist).fillna(0)
        
        cat_dist_df.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_ylabel('Contagem')
        axes[0, 1].set_title('Distribuição de Categorias por Variação')
        axes[0, 1].legend(title='Configuração')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Matriz de concordância entre variações
        n_vars = len(all_cols)
        agreement_matrix = np.zeros((n_vars, n_vars))
        
        for i, col1 in enumerate(all_cols):
            for j, col2 in enumerate(all_cols):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreement_matrix[i, j] = (df[col1] == df[col2]).sum() / len(df)
        
        col_labels = [col.replace(f"{model}_", "").replace("_", "\n") for col in all_cols]
        
        sns.heatmap(
            agreement_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            square=True,
            xticklabels=col_labels,
            yticklabels=col_labels,
            ax=axes[1, 0],
            cbar_kws={'label': 'Concordância'}
        )
        axes[1, 0].set_title('Matriz de Concordância entre Variações')
        
        # Estatísticas
        stats_text = f"""
        Estatísticas de Variação de Parâmetros:
        
        Modelo: {model}
        Configurações testadas: {len(all_cols)}
        
        Concordância média com baseline:
        {agree_df['agreement'].mean():.3f} ± {agree_df['agreement'].std():.3f}
        
        Maior concordância: {agree_df['agreement'].max():.3f}
        Menor concordância: {agree_df['agreement'].min():.3f}
        
        Variação mais estável: {agree_df.iloc[agree_df['agreement'].argmax()]['variation']}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10,
                        verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Impacto de parâmetros salvo: {save_name}")
    
    def create_interactive_dashboard(
        self,
        df: pd.DataFrame,
        report: Dict,
        save_name: str = "interactive_dashboard.html"
    ):
        """
        Cria dashboard interativo com Plotly
        
        Args:
            df: DataFrame com anotações
            report: Dicionário com relatório de consenso
            save_name: Nome do arquivo HTML para salvar
        """
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribuição de Consenso',
                'Concordância entre Modelos',
                'Entropia por Instância',
                'Categorias Mais Confundidas'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'heatmap'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ]
        )
        
        # 1. Histograma de consenso
        fig.add_trace(
            go.Histogram(x=df['consensus_score'], name='Consenso', nbinsx=20),
            row=1, col=1
        )
        
        # 2. Heatmap de concordância
        if 'pairwise_agreement' in report:
            agreement_df = report['pairwise_agreement']
            fig.add_trace(
                go.Heatmap(
                    z=agreement_df.values,
                    x=agreement_df.columns,
                    y=agreement_df.index,
                    colorscale='RdYlGn',
                    name='Concordância'
                ),
                row=1, col=2
            )
        
        # 3. Scatter de entropia
        if 'instance_entropy' in report:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(df))),
                    y=report['instance_entropy'],
                    mode='markers',
                    name='Entropia',
                    marker=dict(
                        color=df['consensus_score'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Consenso')
                    )
                ),
                row=2, col=1
            )
        
        # 4. Pares mais confundidos
        if 'disagreement_patterns' in report and 'most_confused_pairs' in report['disagreement_patterns']:
            confused = report['disagreement_patterns']['most_confused_pairs'].head(10)
            confused['pair'] = confused['category_1'] + ' ↔ ' + confused['category_2']
            
            fig.add_trace(
                go.Bar(
                    x=confused['confusion_count'],
                    y=confused['pair'],
                    orientation='h',
                    name='Confusões'
                ),
                row=2, col=2
            )
        
        # Atualizar layout
        fig.update_layout(
            title_text="Dashboard de Análise de Consenso",
            showlegend=False,
            height=800
        )
        
        # Salvar
        fig.write_html(self.output_dir / save_name)
        
        print(f"✓ Dashboard interativo salvo: {save_name}")
