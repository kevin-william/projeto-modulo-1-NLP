import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def plotar_top_palavras_por_topico(topicos_com_pesos, nome_modelo, diretorio_saida):
    """Plota barras horizontais proporcionais ao peso de cada palavra por topico.

    Parametros
    ----------
    topicos_com_pesos : List[List[Tuple[str, float]]]
        Cada elemento eh uma lista de (palavra, peso) para um topico.
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    n_topicos = len(topicos_com_pesos)
    n_cols = 2
    n_rows = (n_topicos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, topico in enumerate(topicos_com_pesos):
        ax = axes[idx]
        palavras = [p for p, _ in topico]
        pesos = [w for _, w in topico]

        max_peso = max(pesos) if pesos and max(pesos) > 0 else 1.0
        larguras = [p / max_peso for p in pesos]

        # Ordem crescente para que a palavra mais relevante fique no topo
        palavras_rev = palavras[::-1]
        larguras_rev = larguras[::-1]
        posicoes = list(range(len(palavras_rev)))

        ax.barh(posicoes, larguras_rev, color="steelblue", edgecolor="black", alpha=0.75)
        ax.set_yticks(posicoes)
        ax.set_yticklabels(palavras_rev, fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"], fontsize=7)
        ax.set_xlabel("Peso relativo", fontsize=8)
        ax.set_title(f"Topico {idx + 1}", fontsize=10)

    for idx in range(n_topicos, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Top Palavras por Topico — {nome_modelo.upper()}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    caminho = os.path.join(diretorio_saida, f"{nome_modelo}_topicos.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("Grafico salvo: %s", caminho)


def plotar_distribuicao_topicos_documentos(distribuicao, titulos, nome_modelo, diretorio_saida):
    os.makedirs(diretorio_saida, exist_ok=True)

    n_docs = min(len(titulos), 30)
    distribuicao_display = distribuicao[:n_docs] if hasattr(distribuicao, "__len__") else np.array(distribuicao)[:n_docs]

    fig, ax = plt.subplots(figsize=(12, max(6, n_docs * 0.3)))
    im = ax.imshow(distribuicao_display, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Topico")
    ax.set_ylabel("Documento")
    ax.set_title(f"Distribuicao Topico-Documento (primeiros {n_docs} docs) — {nome_modelo.upper()}")
    ax.set_xticks(range(distribuicao_display.shape[1]))
    ax.set_xticklabels([f"T{i+1}" for i in range(distribuicao_display.shape[1])], fontsize=8)
    ax.set_yticks(range(n_docs))
    ax.set_yticklabels([t[:30] for t in titulos[:n_docs]], fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)

    caminho = os.path.join(diretorio_saida, f"{nome_modelo}_heatmap.png")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("Grafico salvo: %s", caminho)


def plotar_comparacao_metricas(dataframe_comparacao, diretorio_saida):
    os.makedirs(diretorio_saida, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    modelos = dataframe_comparacao["modelo"].tolist()
    coerencia = dataframe_comparacao["coerencia_media"].tolist()
    cores = ["#2196F3", "#4CAF50", "#FF9800"]

    barras = ax.bar(modelos, coerencia, color=cores[:len(modelos)], edgecolor="black", alpha=0.8)
    ax.set_ylabel("Coerencia Media")
    ax.set_title("Comparacao de Coerencia entre Modelos")

    for bar, valor in zip(barras, coerencia):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{valor:.4f}", ha="center", va="bottom", fontsize=10)

    caminho = os.path.join(diretorio_saida, "comparacao_coerencia.png")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("Grafico salvo: %s", caminho)


def plotar_wordcloud_topicos(topicos_com_pesos, nome_modelo, diretorio_saida):
    """Gera uma grade de WordClouds, uma por tópico, com palavras ponderadas pelos pesos.

    Parametros
    ----------
    topicos_com_pesos : List[List[Tuple[str, float]]]
        Cada elemento eh uma lista de (palavra, peso) para um topico.
    nome_modelo : str
        Identificador do modelo (usado no nome do arquivo de saida).
    diretorio_saida : str
        Diretório onde o PNG sera salvo.
    """
    from wordcloud import WordCloud

    os.makedirs(diretorio_saida, exist_ok=True)

    n_topicos = len(topicos_com_pesos)
    n_cols = min(3, n_topicos)
    n_rows = math.ceil(n_topicos / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, topico in enumerate(topicos_com_pesos):
        ax = axes[idx]
        freq = {palavra: max(peso, 1e-6) for palavra, peso in topico}
        wc = WordCloud(
            background_color="white",
            width=800,
            height=600,
            collocations=False,
            prefer_horizontal=1.0,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topico {idx + 1}", fontsize=11)

    for idx in range(n_topicos, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"WordCloud por Topico — {nome_modelo.upper()}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    caminho = os.path.join(diretorio_saida, f"{nome_modelo}_wordcloud_topicos.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("Grafico salvo: %s", caminho)


def gerar_pylda_vis(lda_model, corpus, dictionary, diretorio_saida):
    """Gera visualização interativa pyLDAvis para um modelo LDA Gensim.

    Parametros
    ----------
    lda_model : gensim.models.LdaModel
    corpus : list — BoW corpus no formato Gensim (lista de lista de tuplas).
    dictionary : gensim.corpora.Dictionary
    diretorio_saida : str
    """
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis

    os.makedirs(diretorio_saida, exist_ok=True)

    vis = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    caminho_html = os.path.join(diretorio_saida, "lda_pyldavis.html")
    pyLDAvis.save_html(vis, caminho_html)
    logger.info("pyLDAvis salvo: %s", caminho_html)
