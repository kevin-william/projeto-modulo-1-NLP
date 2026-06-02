import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def executar_eda(documentos, titulos, diretorio_saida):
    logger.info("Iniciando EDA textual sobre %d documentos", len(documentos))

    os.makedirs(diretorio_saida, exist_ok=True)

    comprimentos = [len(doc.split()) for doc in documentos]

    todas_palavras = []
    for doc in documentos:
        todas_palavras.extend(doc.split())
    frequencias = Counter(todas_palavras)
    tamanho_vocabulario = len(frequencias)

    comprimento_medio = float(np.mean(comprimentos))
    comprimento_mediano = float(np.median(comprimentos))
    comprimento_std = float(np.std(comprimentos))
    comprimento_min = int(np.min(comprimentos))
    comprimento_max = int(np.max(comprimentos))

    top_20 = frequencias.most_common(20)

    metricas = {
        "num_documentos": len(documentos),
        "comprimento_medio": comprimento_medio,
        "comprimento_mediano": comprimento_mediano,
        "comprimento_std": comprimento_std,
        "comprimento_min": comprimento_min,
        "comprimento_max": comprimento_max,
        "tamanho_vocabulario": tamanho_vocabulario,
        "total_tokens": len(todas_palavras),
        "top_20_termos": top_20,
    }

    logger.info("Comprimento dos documentos: media=%.1f mediana=%.1f std=%.1f min=%d max=%d",
                comprimento_medio, comprimento_mediano, comprimento_std, comprimento_min, comprimento_max)
    logger.info("Tamanho do vocabulario: %d", tamanho_vocabulario)
    logger.info("Total de tokens no corpus: %d", len(todas_palavras))

    # Top-20 termos
    if top_20:
        palavras, contagens = zip(*top_20)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(palavras)[::-1], list(contagens)[::-1], edgecolor="black", alpha=0.7)
        ax.set_xlabel("Frequencia")
        ax.set_title("Top-20 Termos Mais Frequentes")
        fig.tight_layout()
        fig.savefig(os.path.join(diretorio_saida, "eda_top_termos.png"), dpi=150)
        plt.close(fig)
        logger.info("Grafico salvo: eda_top_termos.png")

    # Curva de Zipf
    frequencias_ordenadas = sorted(frequencias.values(), reverse=True)
    ranks = list(range(1, len(frequencias_ordenadas) + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ranks, frequencias_ordenadas, marker=".", linestyle="none", alpha=0.5)
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Frequencia (log)")
    ax.set_title("Curva de Zipf do Corpus")
    fig.tight_layout()
    fig.savefig(os.path.join(diretorio_saida, "eda_zipf.png"), dpi=150)
    plt.close(fig)
    logger.info("Grafico salvo: eda_zipf.png")

    logger.info("EDA concluida com sucesso")
    return metricas
