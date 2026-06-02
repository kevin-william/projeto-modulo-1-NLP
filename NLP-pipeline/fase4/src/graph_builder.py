"""
Construção, análise e visualização do grafo de conhecimento.
Consolida: grafo_conhecimento.py + visualizacao_grafo.py (4 gráficos essenciais)
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ── Constantes inline ─────────────────────────────────────────────────────
TOP_N_GRAPH   = 300
TOP_N_LABELS  = 25
TOP_N_RANKING = 10
DPI           = 150
FIGSIZE_GRAPH = (16, 12)
FIGSIZE_BARS  = (12, 8)

_CORES: Dict[str, str] = {
    "ORG":          "#4C72B0",
    "ORGANIZATION": "#4C72B0",
    "PERSON":       "#55A868",
    "PER":          "#55A868",
    "GPE":          "#C44E52",
    "LOC":          "#DD8452",
    "MISC":         "#8172B2",
    "GPE_LOC":      "#C44E52",
    "DATE":         "#937860",
    "NORP":         "#DA8BC3",
}
_COR_DEFAULT = "#AAAAAA"

logger = logging.getLogger(__name__)


def _cor(tipo: str) -> str:
    return _CORES.get(tipo, _COR_DEFAULT)


class KnowledgeGraphBuilder:
    """Grafo NetworkX com centralidade e 4 visualizações essenciais."""

    def __init__(self, directed: bool = False):
        self.graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
        self._centrality: Dict[str, Dict] = {}

    # ── CONSTRUÇÃO ────────────────────────────────────────────────────────
    def add_relations(self, relations) -> None:
        """Adiciona relações como arestas. Aceita lista de tuplas ou DataFrame."""
        if isinstance(relations, pd.DataFrame):
            if relations.empty:
                return
            src_col = "sujeito" if "sujeito" in relations.columns else "source"
            tgt_col = "objeto"  if "objeto"  in relations.columns else "target"
            rel_col = "verbo"   if "verbo"   in relations.columns else "relation"
            for _, row in relations.iterrows():
                s = str(row[src_col])
                t = str(row[tgt_col])
                verb = str(row[rel_col]) if rel_col in relations.columns else ""
                if self.graph.has_edge(s, t):
                    self.graph[s][t]["weight"] = (
                        self.graph[s][t].get("weight", 1) + 1
                    )
                else:
                    self.graph.add_edge(s, t, weight=1, relation=verb)
        else:
            for item in relations:
                s, v, o = str(item[0]), str(item[1]), str(item[2])
                if self.graph.has_edge(s, o):
                    self.graph[s][o]["weight"] = (
                        self.graph[s][o].get("weight", 1) + 1
                    )
                else:
                    self.graph.add_edge(s, o, weight=1, relation=v)

    def add_entities(
        self,
        entities,
        entity_type: Optional[str] = None,
    ) -> None:
        """Adiciona entidades como nós com atributo 'tipo'."""
        if isinstance(entities, pd.DataFrame):
            col_text = (
                "canonical_form" if "canonical_form" in entities.columns
                else ("text" if "text" in entities.columns else "texto")
            )
            col_type = (
                "label" if "label" in entities.columns else "tipo"
            )
            freq = Counter(entities[col_text].dropna().tolist())
            for ent, count in freq.items():
                tipo = entity_type
                if tipo is None and col_type in entities.columns:
                    vals = entities.loc[entities[col_text] == ent, col_type].values
                    tipo = str(vals[0]) if len(vals) > 0 else "MISC"
                if not self.graph.has_node(ent):
                    self.graph.add_node(ent, tipo=tipo or "MISC", frequencia=count)
                else:
                    self.graph.nodes[ent]["frequencia"] = count
                    if tipo:
                        self.graph.nodes[ent]["tipo"] = tipo
        else:
            for ent in entities:
                if not self.graph.has_node(str(ent)):
                    self.graph.add_node(
                        str(ent), tipo=entity_type or "MISC", frequencia=1
                    )

    def build_from_dataframes(
        self,
        edges_df: pd.DataFrame,
        entities_df: pd.DataFrame,
    ) -> None:
        """Constrói grafo a partir de DataFrames."""
        self.add_relations(edges_df)
        self.add_entities(entities_df)

    # ── ANÁLISE ───────────────────────────────────────────────────────────
    def calculate_centrality(self) -> Dict[str, Dict]:
        """4 métricas: degree, betweenness, closeness, eigenvector."""
        if self.graph.number_of_nodes() == 0:
            return {}

        # Trabalha com grafo não-dirigido do maior componente conexo
        g: nx.Graph = (
            self.graph
            if isinstance(self.graph, nx.Graph)
            else self.graph.to_undirected()
        )
        if not nx.is_connected(g):
            biggest = max(nx.connected_components(g), key=len)
            g = g.subgraph(biggest).copy()

        result: Dict[str, Dict] = {}
        result["degree"] = dict(nx.degree_centrality(g))

        try:
            result["betweenness"] = nx.betweenness_centrality(g, normalized=True)
        except Exception:
            result["betweenness"] = {}

        try:
            result["closeness"] = dict(nx.closeness_centrality(g))
        except Exception:
            result["closeness"] = {}

        try:
            result["eigenvector"] = nx.eigenvector_centrality(g, max_iter=500)
        except Exception:
            result["eigenvector"] = {}

        self._centrality = result
        return result

    def get_top_nodes(
        self, metric: str = "betweenness", top_n: int = TOP_N_RANKING
    ) -> List[Tuple]:
        """Top N nós por métrica de centralidade."""
        if not self._centrality:
            self.calculate_centrality()
        vals = self._centrality.get(metric, {})
        return sorted(vals.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_graph_stats(self) -> Dict:
        """nodes, edges, density, avg_degree, is_connected."""
        g = self.graph
        n = g.number_of_nodes()
        e = g.number_of_edges()
        ug = g if isinstance(g, nx.Graph) else g.to_undirected()
        return {
            "nodes":        n,
            "edges":        e,
            "density":      round(nx.density(g), 6) if n > 1 else 0.0,
            "avg_degree":   round(
                sum(d for _, d in g.degree()) / max(n, 1), 2
            ),
            "is_connected": nx.is_connected(ug) if n > 0 else False,
        }

    # ── VISUALIZAÇÕES (4 GRÁFICOS) ────────────────────────────────────────
    def plot_graph(
        self, output_path: str, top_n: int = TOP_N_GRAPH
    ) -> None:
        """Gráfico 1: Grafo de conhecimento (top N nós, layout spring)."""
        g = self.graph
        if g.number_of_nodes() == 0:
            return
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Sub-grafo com top_n nós por grau
        top_nodes = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:top_n]
        top_set = {n for n, _ in top_nodes}
        sub = g.subgraph(top_set).copy()

        n_sub = sub.number_of_nodes()
        k_val = 1.2 / max(n_sub ** 0.5, 1)
        try:
            pos = nx.spring_layout(sub, k=k_val, iterations=80, seed=42)
        except Exception:
            pos = nx.random_layout(sub, seed=42)

        graus = dict(sub.degree())
        max_grau = max(graus.values(), default=0) or 1
        # escala raiz quadrada — hubs em destaque sem esmagar os menores
        sizes = [max(80, 2000 * (graus.get(n, 1) / max_grau) ** 0.5) for n in sub.nodes()]
        cores = [_cor(sub.nodes[n].get("tipo", "MISC")) for n in sub.nodes()]

        if self._centrality:
            bet = self._centrality.get(
                "betweenness", self._centrality.get("degree", {})
            )
            top_label_list = sorted(bet, key=bet.get, reverse=True)[:TOP_N_LABELS]
        else:
            top_label_list = list(top_set)[:TOP_N_LABELS]
        top_labels = set(top_label_list)
        top10 = set(top_label_list[:10])

        labels = {n: (n if n in top_labels else "") for n in sub.nodes()}
        pesos  = [sub[u][v].get("weight", 1) for u, v in sub.edges()]
        max_p  = max(pesos) if pesos else 1
        widths = [0.4 + 2.0 * p / max_p for p in pesos]
        alphas = [0.1 + 0.5 * p / max_p for p in pesos]

        fig, ax = plt.subplots(figsize=(20, 16))
        nx.draw_networkx_nodes(
            sub, pos, node_color=cores, node_size=sizes, ax=ax, alpha=0.85
        )
        for (u, v), larg, alpha in zip(list(sub.edges()), widths, alphas):
            nx.draw_networkx_edges(
                sub, pos, edgelist=[(u, v)], width=larg,
                alpha=alpha, ax=ax, edge_color="#888888"
            )
        labels_normais = {n: lbl for n, lbl in labels.items() if n in top_labels and n not in top10}
        nx.draw_networkx_labels(
            sub, pos, labels=labels_normais, font_size=8, ax=ax,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, edgecolor="none"),
        )
        labels_destaque = {n: lbl for n, lbl in labels.items() if n in top10}
        nx.draw_networkx_labels(
            sub, pos, labels=labels_destaque, font_size=9, font_weight="semibold", ax=ax,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.80, edgecolor="#CCCCCC"),
        )

        tipos = set(sub.nodes[n].get("tipo", "MISC") for n in sub.nodes())
        patches = [mpatches.Patch(color=_cor(t), label=t) for t in sorted(tipos)]
        ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.85)
        ax.set_title(
            f"Grafo de Conhecimento (Top {top_n} nós)\n"
            f"{sub.number_of_nodes()} nós | {sub.number_of_edges()} arestas",
            fontsize=13,
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Grafo salvo: %s", output_path)

    def plot_centrality_distribution(self, output_path: str) -> None:
        """Gráfico 2: Distribuição das 4 métricas de centralidade (histograma 2×2)."""
        if not self._centrality:
            self.calculate_centrality()
        metricas = {k: list(v.values()) for k, v in self._centrality.items() if v}
        if not metricas:
            return
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        n_plots = len(metricas)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (metrica, vals) in zip(axes.flatten(), metricas.items()):
            ax.hist(vals, bins=50, color="#4C72B0", edgecolor="none", log=True)
            ax.set_title(f"{metrica} centrality")
            ax.set_xlabel("Centralidade")
            ax.set_ylabel("Nº nós (escala log)")
            ax.axvline(
                x=float(np.mean(vals)), color="#C44E52", linewidth=1.2,
                linestyle="--", label=f"média={np.mean(vals):.4f}"
            )
            ax.legend(fontsize=7)
        for ax in axes.flatten()[n_plots:]:
            ax.set_visible(False)
        plt.suptitle(
            "Distribuição de Centralidade — 4 métricas\n"
            "(escala log: maioria dos nós tem centralidade próxima de 0; "
            "poucos hubs concentram alta centralidade)",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Distribuição centralidade salva: %s", output_path)

    def plot_top_entities_by_type(
        self, entities_df: pd.DataFrame, output_path: str
    ) -> None:
        """Gráfico 3: Top entidades por tipo (barras horizontais)."""
        if entities_df.empty:
            return
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        col_text = (
            "canonical_form" if "canonical_form" in entities_df.columns
            else ("text" if "text" in entities_df.columns else "texto")
        )
        col_type = (
            "label" if "label" in entities_df.columns else "tipo"
        )
        tipos_alvo = ["ORG", "ORGANIZATION", "PERSON", "PER", "GPE", "LOC"]
        df = entities_df[entities_df[col_type].isin(tipos_alvo)]
        tipos = [t for t in tipos_alvo if t in df[col_type].unique()][:4]
        if not tipos:
            return

        fig, axes = plt.subplots(1, len(tipos), figsize=(5 * len(tipos), 6))
        if len(tipos) == 1:
            axes = [axes]
        for ax, tipo in zip(axes, tipos):
            sub = df[df[col_type] == tipo][col_text].value_counts().head(10)
            if sub.empty:
                ax.set_visible(False)
                continue
            ax.barh(sub.index, sub.values, color=_cor(tipo))
            ax.set_title(tipo)
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=7)
        plt.suptitle("Top entidades por tipo", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Top entidades por tipo salvo: %s", output_path)

    def plot_relations_heatmap(
        self, output_path: str, top_n: int = 15
    ) -> None:
        """Gráfico 4: Heatmap de co-ocorrência (matriz de adjacência)."""
        if self.graph.number_of_edges() == 0:
            return
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn nao instalado — heatmap nao gerado")
            return

        top_nodes = sorted(
            set(n for edge in self.graph.edges() for n in edge),
            key=lambda n: self.graph.degree(n),
            reverse=True,
        )[:top_n]
        if len(top_nodes) < 2:
            return

        matriz = pd.DataFrame(0.0, index=top_nodes, columns=top_nodes)
        for u, v, data in self.graph.edges(data=True):
            w = float(data.get("weight", 1))
            if u in top_nodes and v in top_nodes:
                matriz.loc[u, v] += w
                matriz.loc[v, u] += w

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            matriz,
            ax=ax,
            cmap="YlOrRd",
            linewidths=0.5,
            xticklabels=True,
            yticklabels=True,
        )
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title("Heatmap de co-ocorrência entre entidades")
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap salvo: %s", output_path)

    def generate_all_plots(
        self, entities_df: pd.DataFrame, output_dir: str
    ) -> None:
        """Gera os 4 gráficos de uma vez."""
        os.makedirs(output_dir, exist_ok=True)
        if not self._centrality:
            self.calculate_centrality()
        self.plot_graph(
            os.path.join(output_dir, "01_knowledge_graph.png")
        )
        self.plot_centrality_distribution(
            os.path.join(output_dir, "02_centrality_distribution.png")
        )
        self.plot_top_entities_by_type(
            entities_df,
            os.path.join(output_dir, "03_top_entities_by_type.png"),
        )
        self.plot_relations_heatmap(
            os.path.join(output_dir, "04_relations_heatmap.png")
        )
        logger.info("4 gráficos gerados em: %s", output_dir)

    # ── EXPORTAÇÃO ────────────────────────────────────────────────────────
    def export_csv(self, nodes_path: str, edges_path: str) -> None:
        """Exporta nós e arestas para CSV."""
        for path in (nodes_path, edges_path):
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)

        nodes_rows = [
            {"node": n, **attrs}
            for n, attrs in self.graph.nodes(data=True)
        ]
        pd.DataFrame(nodes_rows).to_csv(nodes_path, index=False, encoding="utf-8")

        edges_rows = [
            {"source": u, "target": v, **data}
            for u, v, data in self.graph.edges(data=True)
        ]
        pd.DataFrame(edges_rows).to_csv(edges_path, index=False, encoding="utf-8")
        logger.info("Nós: %s | Arestas: %s", nodes_path, edges_path)

    def export_json(self, output_path: str) -> None:
        """Exporta grafo em JSON (NetworkX node-link format)."""
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
        logger.info("Grafo JSON salvo: %s", output_path)
