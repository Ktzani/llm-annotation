"""
Visualization Module - Visualizações para análise de consenso (versão 100% Plotly)
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


class ConsensusVisualizer:
    """
    Classe para criar visualizações dos resultados de consenso (AGORA 100% PLOTLY)
    """
    
    def __init__(self, output_dir: str = "./results/graphics"):
        self.output_dir = Path(output_dir)
        self.output_dir = self.output_dir.joinpath("graphics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_score_and_levels(
        self,
        df_with_consensus,
        levels,
        save_name: str = "score_and_levels.html"
    ): 
        all_levels = ["high", "medium", "low"]

        levels_full = levels.reindex(all_levels, fill_value=0)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Distribuição de Scores de Consenso",
                "Casos por Nível de Consenso"
            )
        )
        fig.add_trace(
            go.Histogram(
                x=df_with_consensus['consensus_score'],
                nbinsx=20,
                marker=dict(line=dict(width=1, color="black")),
                name="Consensus Score"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=levels_full.index,
                y=levels_full.values,
                marker=dict(color=["#2ECC71", "#F5B041", "#E74C3C"]),
                width=0.5,
            ),
            row=1, col=2
        )

        fig.update_layout(
            width=1100,
            height=450,
            title="Análise Visual do Consenso",
            showlegend=False,
            bargap=0.30
        )

        fig.update_xaxes(title_text="Consensus Score", row=1, col=1)
        fig.update_yaxes(title_text="Frequência", row=1, col=1)

        fig.update_xaxes(title_text="Nível de Consenso", row=1, col=2)
        fig.update_yaxes(title_text="Contagem", row=1, col=2)
        
        fig.show()
        
        if save_name:
            fig.write_html(str(self.output_dir / save_name))
            print(f"✓ Gráfico salvo: {save_name}")


    def plot_agreement_heatmap(
        self,
        agreement_df: pd.DataFrame,
        title: str = "Matriz de Concordância entre Anotadores",
        save_name: str = "agreement_heatmap.html"
    ):
        cleaned_df = agreement_df.copy()
        cleaned_df.index = cleaned_df.index.str.replace("_consensus", "")
        cleaned_df.columns = cleaned_df.columns.str.replace("_consensus", "")
        
        fig = go.Figure(
            data=go.Heatmap(
                z=agreement_df.values,
                x=agreement_df.columns,
                y=agreement_df.index,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Taxa de Concordância"),
                text=np.round(agreement_df.values, 3),
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            title=title,
            width=900,
            height=800
        )
        
        fig.show()

        if save_name:
            fig.write_html(str(self.output_dir / save_name))
            print(f"✓ Heatmap salvo: {save_name}")
            
    def plot_kappa_heatmap(
        self,
        kappa_df: pd.DataFrame,
        title: str = "Cohen's Kappa - Matriz de Concordância",
        save_name: str = "kappa_heatmap.html"
    ):

        df = kappa_df.copy()
        df["annotator_1"] = df["annotator_1"].str.replace("_consensus", "")
        df["annotator_2"] = df["annotator_2"].str.replace("_consensus", "")

        annotators = sorted(
            list(set(df["annotator_1"]).union(set(df["annotator_2"])))
        )

        matrix = pd.DataFrame(
            np.eye(len(annotators)),
            index=annotators,
            columns=annotators
        )

        for _, row in df.iterrows():
            a1, a2, kappa = row["annotator_1"], row["annotator_2"], row["cohens_kappa"]
            matrix.loc[a1, a2] = kappa
            matrix.loc[a2, a1] = kappa

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale="greens",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Cohen's Kappa"),
                text=np.round(matrix.values, 2), 
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            title=title,
            width=900,
            height=800
        )

        fig.show()

        if save_name:
            fig.write_html(str(self.output_dir / save_name))
            print(f"✓ Heatmap salvo: {save_name}")

    def plot_confusion_matrix(
        self, 
        cm,
        categories,
        save_name="confusion_matrix.html"
    ):
        cm_plot = np.flipud(cm)
        categories_rev = categories[::-1]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm_plot,
                x=categories,
                y=categories_rev,
                colorscale="Reds",
                text=cm_plot,
                texttemplate="%{text}",
                colorbar=dict(title="Contagens"),
            )
        )

        fig.update_layout(
            title="Matriz de Confusão (Real × Predito)",
            width=900,
            height=800,
            xaxis_title="Predito",
            yaxis_title="Real (Ground Truth)"
        )
        fig.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=categories,
            ticktext=categories
        )

        fig.update_yaxes(
            type="category",
            tickmode="array",
            tickvals=categories_rev,
            ticktext=categories_rev
        )

        fig.show()

        if save_name:
            output_path = self.output_dir / save_name
            fig.write_html(str(output_path))
            print(f"✓ Matriz de confusão salva em: {output_path}")


    def plot_parameter_impact(
        self,
        df: pd.DataFrame,
        model: str,
        save_name: str = None
    ):
        if save_name is None:
            save_name = f"{model}_parameter_impact.html"

        param_cols = [c for c in df.columns if c.startswith(f"{model}_param_var")]
        consensus_col = f"{model}_consensus"

        if not param_cols or consensus_col not in df:
            print(f"⚠ Variações não encontradas para {model}")
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Concordância com Baseline",
                "Distribuição de Categorias",
                "Matriz de Concordância",
                "Estatísticas"
            ]
        )

        agreements = [
            (col, (df[col] == df[consensus_col]).mean())
            for col in param_cols
        ]

        names = [a[0].replace(f"{model}_param_var", "Var ") for a in agreements]
        values = [a[1] for a in agreements]

        fig.add_trace(
            go.Bar(x=names, y=values),
            row=1, col=1
        )

        # --- Distribuição ---
        cols = [consensus_col] + param_cols
        dist = pd.DataFrame({
            c: df[c].value_counts() for c in cols
        }).fillna(0)

        for cfg in dist.columns:
            fig.add_trace(
                go.Bar(
                    x=dist.index,
                    y=dist[cfg],
                    name=cfg
                ),
                row=1, col=2
            )
            
        n = len(cols)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i, j] = (df[cols[i]] == df[cols[j]]).mean()

        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=cols,
                y=cols,
                colorscale="RdYlGn",
                zmin=0,
                zmax=1
            ),
            row=2, col=1
        )

        stats = f"""
Configurações: {len(cols)}

Concordância média: {np.mean(values):.3f}
Maior: {np.max(values):.3f}
Menor: {np.min(values):.3f}
"""
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                text=[stats],
                mode="text",
                textfont=dict(family="monospace")
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=1100,
            title=f"Impacto de Parâmetros - {model}"
        )
        
        fig.show()

        fig.write_html(str(self.output_dir / save_name))
        print(f"✓ Impacto salvo: {save_name}")

    # # -------------------------------------------------------------------------
    # # DASHBOARD INTERATIVO (já era Plotly)
    # # -------------------------------------------------------------------------
    # def create_interactive_dashboard(
    #     self,
    #     df: pd.DataFrame,
    #     report: Dict,
    #     save_name: str = "interactive_dashboard.html"
    # ):
    #     fig = make_subplots(
    #         rows=2, cols=2,
    #         subplot_titles=(
    #             'Distribuição de Consenso',
    #             'Concordância entre Modelos',
    #             'Entropia por Instância',
    #             'Categorias Mais Confundidas'
    #         ),
    #         specs=[
    #             [{'type': 'histogram'}, {'type': 'heatmap'}],
    #             [{'type': 'scatter'}, {'type': 'bar'}]
    #         ]
    #     )

    #     fig.add_trace(
    #         go.Histogram(x=df['consensus_score']),
    #         row=1, col=1
    #     )

    #     if 'pairwise_agreement' in report:
    #         a = report['pairwise_agreement']
    #         fig.add_trace(
    #             go.Heatmap(z=a.values, x=a.columns, y=a.index),
    #             row=1, col=2
    #         )

    #     if 'instance_entropy' in report:
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=list(range(len(df))),
    #                 y=report['instance_entropy'],
    #                 mode='markers',
    #                 marker=dict(
    #                     color=df['consensus_score'],
    #                     colorscale="Viridis",
    #                     showscale=True
    #                 )
    #             ),
    #             row=2, col=1
    #         )

    #     if (
    #         'disagreement_patterns' in report
    #         and 'most_confused_pairs' in report['disagreement_patterns']
    #     ):
    #         confused = report['disagreement_patterns']['most_confused_pairs'].head(10)
    #         fig.add_trace(
    #             go.Bar(
    #                 x=confused['confusion_count'],
    #                 y=confused['category_1'] + " ↔ " + confused['category_2'],
    #                 orientation='h'
    #             ),
    #             row=2, col=2
    #         )

    #     fig.update_layout(
    #         title="Dashboard de Análise de Consenso",
    #         height=900
    #     )

    #     fig.write_html(self.output_dir / save_name)
    #     print(f"✓ Dashboard salvo: {save_name}")
    
