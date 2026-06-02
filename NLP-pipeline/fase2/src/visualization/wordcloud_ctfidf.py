import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def gerar_grid_wordclouds_ctfidf(ctfidf_por_categoria, caminho_saida):
    """Gera um grid de WordClouds — uma por categoria — baseado nos scores c-TF-IDF.

    Args:
        ctfidf_por_categoria: dict[str, pd.Series] — scores por categoria (saida de CalculadoraCTfIdf).
        caminho_saida: str — caminho do arquivo de imagem a salvar.
    """
    n = len(ctfidf_por_categoria)
    if n == 0:
        logger.warning("Nenhuma categoria disponivel para gerar WordCloud c-TF-IDF.")
        return

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    else:
        axes = [list(row) for row in axes]

    axes_flat = [ax for row in axes for ax in row]

    for i, (categoria, scores) in enumerate(ctfidf_por_categoria.items()):
        ax = axes_flat[i]
        freq_dict = {k: float(v) for k, v in scores.items() if v > 0}
        if freq_dict:
            wc = WordCloud(
                background_color="white",
                max_words=100,
                max_font_size=30,
                min_font_size=10,
                prefer_horizontal=1,
                width=600,
                height=400,
            ).generate_from_frequencies(freq_dict)
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(categoria, fontsize=14)
        ax.axis("off")

    for j in range(len(ctfidf_por_categoria), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grid de WordClouds c-TF-IDF salvo em: %s", caminho_saida)
