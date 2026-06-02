"""
Visualizacao do grafo de conhecimento com matplotlib e PyVis.

Gera graficos estaticos, interativos e o infografico de resumo para
audiencia nao tecnica.
"""

from __future__ import annotations

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

from fase4_config import (
    DPI_OUTPUT,
    FIGSIZE_BARRAS,
    FIGSIZE_GRAFO,
    NODE_SIZE_FACTOR,
)
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)

# Paleta por tipo de entidade
_CORES_TIPO = {
    "ORG": "#4C72B0",
    "ORGANIZATION": "#4C72B0",
    "PERSON": "#55A868",
    "PER": "#55A868",
    "GPE": "#C44E52",
    "LOC": "#DD8452",
    "MISC": "#8172B2",
    "GPE_LOC": "#C44E52",
    "DATE": "#937860",
    "NORP": "#DA8BC3",
    "documento": "#CCCCCC",
}
_COR_DEFAULT = "#AAAAAA"


def _cor_para_tipo(tipo: str) -> str:
    return _CORES_TIPO.get(tipo, _COR_DEFAULT)


_TOP_N_GRAPH  = 300   # nós exibidos no grafo matplotlib
_TOP_N_LABELS = 25    # nós com label visível (por betweenness)


def _obter_layout(grafo: nx.Graph) -> Dict:
    n = grafo.number_of_nodes()
    k = 1.2 / max(n ** 0.5, 1)
    try:
        return nx.spring_layout(grafo, k=k, iterations=80, seed=42)
    except Exception:
        return nx.random_layout(grafo, seed=42)


def plotar_grafo_matplotlib(
    grafo: nx.Graph,
    centralidade: Dict[str, pd.DataFrame],
    caminho_saida: str,
    titulo: str = "Grafo de Conhecimento",
    figsize: Tuple[int, int] = (20, 16),
    dpi: int = 200,
) -> None:
    if grafo.number_of_nodes() == 0:
        logger.warning("Grafo vazio — grafo matplotlib nao gerado")
        return
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    # ── Subgrafo: top 300 nós por grau ──────────────────────────────────
    graus_total = dict(grafo.degree())
    top_nos = sorted(graus_total, key=graus_total.get, reverse=True)[:_TOP_N_GRAPH]
    sub = grafo.subgraph(top_nos).copy()

    pos = _obter_layout(sub)

    # Grau para tamanho (escala raiz quadrada → diferencia hubs sem esmagar)
    graus = dict(sub.degree())
    max_grau = max(graus.values(), default=0) or 1
    tamanhos = [
        max(80, 2000 * (graus.get(n, 1) / max_grau) ** 0.5)
        for n in sub.nodes()
    ]

    cores = [
        _cor_para_tipo(sub.nodes[n].get("tipo", "MISC")) for n in sub.nodes()
    ]

    # Labels: top 25 por betweenness; top 10 em negrito
    df_central = centralidade.get("betweenness", centralidade.get("degree", pd.DataFrame()))
    top_labels: list = []
    if not df_central.empty and "entidade" in df_central.columns:
        top_labels = df_central.head(_TOP_N_LABELS)["entidade"].tolist()
    top_labels_set = set(top_labels)
    top10_set = set(top_labels[:10])

    labels = {n: (n if n in top_labels_set else "") for n in sub.nodes()}

    # Pesos das arestas (alpha e largura proporcionais ao peso)
    pesos = [
        sub[u][v].get("frequencia", sub[u][v].get("weight", 1))
        for u, v in sub.edges()
    ]
    max_peso = max(pesos) if pesos else 1
    larguras = [0.4 + 2.0 * p / max_peso for p in pesos]
    alphas   = [0.1 + 0.5 * p / max_peso for p in pesos]

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(sub, pos, node_color=cores, node_size=tamanhos, ax=ax, alpha=0.85)

    # Arestas com alpha individual
    edge_list = list(sub.edges())
    for (u, v), larg, alpha in zip(edge_list, larguras, alphas):
        nx.draw_networkx_edges(
            sub, pos, edgelist=[(u, v)], width=larg,
            alpha=alpha, ax=ax, edge_color="#888888"
        )

    # Labels normais (top 11-25)
    labels_normais = {n: lbl for n, lbl in labels.items() if n in top_labels_set and n not in top10_set}
    nx.draw_networkx_labels(
        sub, pos, labels=labels_normais, font_size=8, ax=ax,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, edgecolor="none"),
    )

    # Labels em destaque para top-10
    labels_destaque = {n: lbl for n, lbl in labels.items() if n in top10_set}
    nx.draw_networkx_labels(
        sub, pos, labels=labels_destaque, font_size=9, font_weight="semibold", ax=ax,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.80, edgecolor="#CCCCCC"),
    )

    # Legenda
    tipos_presentes = set(sub.nodes[n].get("tipo", "MISC") for n in sub.nodes())
    patches = [
        mpatches.Patch(color=_cor_para_tipo(t), label=t)
        for t in sorted(tipos_presentes)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.85)

    ax.set_title(
        f"{titulo} (Top {_TOP_N_GRAPH} nós)\n"
        f"{sub.number_of_nodes()} nós | {sub.number_of_edges()} arestas",
        fontsize=13,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafo matplotlib salvo: %s", caminho_saida)


def plotar_grafo_pyvis(
    grafo: nx.Graph,
    centralidade: Dict[str, pd.DataFrame],
    caminho_saida: str,
) -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("PyVis nao instalado — grafo interativo nao gerado")
        return

    if grafo.number_of_nodes() == 0:
        return
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    net = Network(height="750px", width="100%", notebook=False, directed=False)
    net.toggle_physics(True)

    df_central = centralidade.get("betweenness", pd.DataFrame())
    central_map: Dict[str, float] = {}
    if not df_central.empty and "entidade" in df_central.columns:
        for _, row in df_central.iterrows():
            central_map[row["entidade"]] = row["centralidade"]

    graus = dict(grafo.degree())
    max_grau = max(graus.values(), default=0) or 1

    for node, attrs in grafo.nodes(data=True):
        tipo = attrs.get("tipo", "MISC")
        freq = attrs.get("frequencia", 0)
        central = central_map.get(node, 0.0)
        grau = graus.get(node, 1)
        size = max(10, 40 * grau / max_grau)
        cor = _cor_para_tipo(tipo)
        title = f"{node}\nTipo: {tipo}\nFreq: {freq}\nCentralidade: {central:.4f}"
        net.add_node(str(node), label=str(node), color=cor, size=size, title=title)

    for u, v, data in grafo.edges(data=True):
        peso = data.get("frequencia", data.get("weight", 1))
        net.add_edge(str(u), str(v), value=peso)

    net.write_html(caminho_saida)
    logger.info("Grafo interativo PyVis salvo: %s", caminho_saida)


def plotar_distribuicao_centralidade(
    centralidade: pd.DataFrame,
    metrica: str,
    caminho_saida: str,
    top_n: int = 10,
) -> None:
    if centralidade.empty:
        return
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df = centralidade.head(top_n).copy()
    cores = df["tipo"].apply(_cor_para_tipo).tolist() if "tipo" in df.columns else None

    fig, ax = plt.subplots(figsize=FIGSIZE_BARRAS)
    bars = ax.barh(df["entidade"], df["centralidade"], color=cores or "#4C72B0")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_title(f"Top {top_n} — {metrica} centrality", fontsize=12)
    ax.set_xlabel("Centralidade")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Distribuicao centralidade salva: %s", caminho_saida)


def plotar_comparacao_centralidades(
    centralidade: Dict[str, pd.DataFrame], caminho_saida: str
) -> None:
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df_bet = centralidade.get("betweenness", pd.DataFrame())
    df_deg = centralidade.get("degree", pd.DataFrame())
    if df_bet.empty or df_deg.empty:
        return

    df_merged = df_bet[["entidade", "centralidade"]].rename(
        columns={"centralidade": "betweenness"}
    ).merge(
        df_deg[["entidade", "centralidade"]].rename(columns={"centralidade": "degree"}),
        on="entidade",
    )
    df_eig = centralidade.get("eigenvector", pd.DataFrame())
    if not df_eig.empty:
        df_merged = df_merged.merge(
            df_eig[["entidade", "centralidade"]].rename(columns={"centralidade": "eigenvector"}),
            on="entidade",
        )
    else:
        df_merged["eigenvector"] = 0.0

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df_merged["betweenness"],
        df_merged["degree"],
        c=df_merged["eigenvector"],
        cmap="viridis",
        s=60,
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label="eigenvector")
    for _, row in df_merged.head(5).iterrows():
        ax.annotate(row["entidade"], (row["betweenness"], row["degree"]), fontsize=7)
    ax.set_xlabel("Betweenness Centrality")
    ax.set_ylabel("Degree Centrality")
    ax.set_title("Comparacao de Centralidades")
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparacao centralidades salva: %s", caminho_saida)


def plotar_top_entidades_por_tipo(
    entidades_df: pd.DataFrame, caminho_saida: str
) -> None:
    if entidades_df.empty or "tipo" not in entidades_df.columns:
        return
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    tipos_principais = ["ORG", "PERSON", "PER", "GPE", "LOC", "ORGANIZATION"]
    df_filtrado = entidades_df[entidades_df["tipo"].isin(tipos_principais)]
    if df_filtrado.empty:
        return
    tipos = df_filtrado["tipo"].unique()[:4]
    fig, axes = plt.subplots(1, len(tipos), figsize=(5 * len(tipos), 6))
    if len(tipos) == 1:
        axes = [axes]
    for ax, tipo in zip(axes, tipos):
        sub = (
            df_filtrado[df_filtrado["tipo"] == tipo]["texto"]
            .value_counts()
            .head(10)
        )
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.barh(sub.index, sub.values, color=_cor_para_tipo(tipo))
        ax.set_title(tipo)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle("Top entidades por tipo", fontsize=12)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Top entidades por tipo salvo: %s", caminho_saida)


def plotar_frequencia_entidades(
    entidades_df: pd.DataFrame, caminho_saida: str, top_n: int = 20
) -> None:
    if entidades_df.empty or "texto" not in entidades_df.columns:
        return
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    freq = entidades_df["texto"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=FIGSIZE_BARRAS)
    ax.barh(freq.index, freq.values, color="#4C72B0")
    ax.set_title(f"Top {top_n} entidades por frequencia")
    ax.set_xlabel("Frequencia")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Frequencia entidades salva: %s", caminho_saida)


def plotar_heatmap_relacoes(
    arestas_df: pd.DataFrame, caminho_saida: str
) -> None:
    if arestas_df.empty:
        return
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn nao instalado — heatmap nao gerado")
        return

    top_nos = set(arestas_df["source"].value_counts().head(15).index) | set(
        arestas_df["target"].value_counts().head(15).index
    )
    top_nos = sorted(top_nos)
    if len(top_nos) < 2:
        return
    matriz = pd.DataFrame(0, index=top_nos, columns=top_nos)
    for _, row in arestas_df.iterrows():
        s, t = row.get("source"), row.get("target")
        freq = row.get("frequencia", 1)
        if s in top_nos and t in top_nos:
            matriz.loc[s, t] += freq
            matriz.loc[t, s] += freq

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matriz, ax=ax, cmap="YlOrRd", linewidths=0.5,
        xticklabels=True, yticklabels=True,
    )
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Heatmap de co-ocorrencia entre entidades")
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap relacoes salvo: %s", caminho_saida)


def plotar_comunidades_grafo(
    grafo: nx.Graph,
    comunidades: Dict[int, List[str]],
    caminho_saida: str,
    top_n: int = 40,
) -> None:
    if grafo.number_of_nodes() == 0:
        return
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    # Filtra para os top-N nós com maior grau
    top_nos = [
        n for n, _ in sorted(grafo.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    ]
    sub = grafo.subgraph(top_nos).copy()
    if sub.number_of_nodes() == 0:
        logger.warning("Subgrafo vazio apos filtro top_n=%d", top_n)
        return

    # Reconstrói mapeamento nó → comunidade apenas para os nós selecionados
    node_comunidade: Dict[str, int] = {}
    for com_id, membros in comunidades.items():
        for m in membros:
            if m in top_nos:
                node_comunidade[m] = com_id
    comunidades_visiveis = len(set(node_comunidade.values()))

    pos = _obter_layout(sub)
    cmap = matplotlib.colormaps["tab20"].resampled(max(1, comunidades_visiveis))
    cores = [cmap(node_comunidade.get(n, 0) % 20) for n in sub.nodes()]

    graus = dict(sub.degree())
    max_grau = max(graus.values(), default=1) or 1
    tamanhos = [max(200, 1500 * graus[n] / max_grau) for n in sub.nodes()]

    fig, ax = plt.subplots(figsize=FIGSIZE_GRAFO)
    nx.draw_networkx_nodes(sub, pos, node_color=cores, node_size=tamanhos,
                           ax=ax, alpha=0.85)
    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.35, edge_color="#888888")
    nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax)
    ax.set_title(
        f"Comunidades detectadas — top {top_n} entidades\n"
        f"({comunidades_visiveis} grupos visíveis de {len(comunidades)} total)",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comunidades grafo salvo: %s", caminho_saida)


def gerar_infografico_resumo(
    grafo: nx.Graph,
    centralidade: Dict[str, pd.DataFrame],
    entidades_df: pd.DataFrame,
    resposta_analitica: str,
    caminho_saida: str,
) -> None:
    """
    Figura composta de 4 quadrantes para audiencia nao tecnica.
    """
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Grafo de Conhecimento — Resumo Executivo", fontsize=16, fontweight="bold")

    # Q1: grafo simplificado (top-10 nos)
    ax1 = fig.add_subplot(2, 2, 1)
    df_central = centralidade.get("betweenness", centralidade.get("degree", pd.DataFrame()))
    if not df_central.empty and "entidade" in df_central.columns:
        top_10 = set(df_central.head(10)["entidade"].tolist())
        sub_grafo = grafo.subgraph([n for n in grafo.nodes() if n in top_10]).copy()
    else:
        nos = list(grafo.nodes())[:10]
        sub_grafo = grafo.subgraph(nos).copy()

    if sub_grafo.number_of_nodes() > 0:
        pos_sub = _obter_layout(sub_grafo)
        cores_sub = [
            _cor_para_tipo(sub_grafo.nodes[n].get("tipo", "MISC"))
            for n in sub_grafo.nodes()
        ]
        nx.draw_networkx(
            sub_grafo, pos_sub, ax=ax1, node_color=cores_sub,
            with_labels=True, node_size=400, font_size=7
        )
    ax1.set_title("Top 10 entidades mais centrais", fontsize=10)
    ax1.axis("off")

    # Q2: ranking de centralidade
    ax2 = fig.add_subplot(2, 2, 2)
    if not df_central.empty:
        top_rank = df_central.head(10)
        cores_rank = top_rank["tipo"].apply(_cor_para_tipo).tolist() if "tipo" in top_rank.columns else "#4C72B0"
        ax2.barh(top_rank["entidade"], top_rank["centralidade"], color=cores_rank)
        ax2.set_title("Ranking de centralidade (betweenness)", fontsize=10)
        ax2.invert_yaxis()
        ax2.tick_params(labelsize=8)
    else:
        ax2.text(0.5, 0.5, "Sem dados de centralidade", ha="center", va="center")
        ax2.set_title("Centralidade", fontsize=10)
    ax2.axis("on")

    # Q3: texto resumo
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis("off")
    resumo = "\n".join(resposta_analitica.split("\n")[:20])
    resumo_wrapped = "\n".join(
        "\n".join(textwrap.wrap(linha, 60)) for linha in resumo.split("\n")
    )
    ax3.text(0.02, 0.98, resumo_wrapped, transform=ax3.transAxes,
             fontsize=8, va="top", ha="left", family="monospace")
    ax3.set_title("Principais achados", fontsize=10)

    # Q4: metricas chave
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    n_nos = grafo.number_of_nodes()
    n_arestas = grafo.number_of_edges()
    densidade = nx.density(grafo) if n_nos > 1 else 0.0
    n_comps = nx.number_connected_components(grafo)
    metricas_texto = (
        f"Metricas do Grafo\n\n"
        f"Nos (entidades): {n_nos}\n"
        f"Arestas (relacoes): {n_arestas}\n"
        f"Densidade: {densidade:.4f}\n"
        f"Componentes conexas: {n_comps}\n"
    )
    if not entidades_df.empty and "tipo" in entidades_df.columns:
        contagem_tipos = entidades_df["tipo"].value_counts().head(4)
        metricas_texto += "\nTipos de entidade:\n"
        for tipo, cnt in contagem_tipos.items():
            metricas_texto += f"  {tipo}: {cnt}\n"
    ax4.text(0.1, 0.9, metricas_texto, transform=ax4.transAxes,
             fontsize=10, va="top", family="monospace")
    ax4.set_title("Metricas-chave", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(caminho_saida, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Infografico resumo salvo: %s", caminho_saida)
