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