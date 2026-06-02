import json
import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fase1_config import CAMINHO_ANALISE_VOCABULARIO, DIRETORIO_SAIDA
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def analisar_vocabulario(tokens_brutos, tokens_filtrados, caminho_saida=None):
    if caminho_saida is None:
        caminho_saida = CAMINHO_ANALISE_VOCABULARIO

    palavras_brutas = [t["lema"].lower() for t in tokens_brutos if t["lema"].strip()]
    palavras_filtradas = [t["lema"].lower() for t in tokens_filtrados if t["lema"].strip()]

    vocabulario_bruto = set(palavras_brutas)
    vocabulario_filtrado = set(palavras_filtradas)

    contador_bruto = Counter(palavras_brutas)
    contador_filtrado = Counter(palavras_filtradas)

    percentual_reducao = round(
        (1 - len(vocabulario_filtrado) / len(vocabulario_bruto)) * 100, 2
    ) if vocabulario_bruto else 0

    analise = {
        "quantidade_vocabulario_bruto": len(vocabulario_bruto),
        "quantidade_vocabulario_filtrado": len(vocabulario_filtrado),
        "percentual_reducao_vocabulario": percentual_reducao,
        "top_20_bruto": contador_bruto.most_common(20),
        "top_20_filtrado": contador_filtrado.most_common(20),
        "amostra_palavras_removidas": list(vocabulario_bruto - vocabulario_filtrado)[:50],
    }

    logger.info("Analise de vocabulario: raw=%d, filtered=%d, reducao=%.2f%%",
                analise["quantidade_vocabulario_bruto"],
                analise["quantidade_vocabulario_filtrado"],
                analise["percentual_reducao_vocabulario"])

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        json.dump(analise, arquivo, ensure_ascii=False, indent=2)
    logger.info("Analise salva em: %s", caminho_saida)

    return analise


def plotar_distribuicao_pos(dataframe, caminho_saida=None):
    if caminho_saida is None:
        caminho_saida = os.path.join(DIRETORIO_SAIDA, "pos_distribution.png")

    contagem_pos = dataframe["pos"].value_counts()

    plt.figure(figsize=(12, 6))
    contagem_pos.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Distribuicao de POS Tags", fontsize=14)
    plt.xlabel("POS Tag", fontsize=12)
    plt.ylabel("Frequencia", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafico de distribuicao POS salvo em: %s", caminho_saida)

    return contagem_pos.to_dict()


def plotar_comparacao_frequencia(contador_bruto, contador_filtrado, top_n=20, caminho_saida=None):
    if caminho_saida is None:
        caminho_saida = os.path.join(DIRETORIO_SAIDA, "freq_comparison.png")

    top_bruto = contador_bruto.most_common(top_n)
    top_filtrado = contador_filtrado.most_common(top_n)

    palavras_brutas, contagens_brutas = zip(*top_bruto) if top_bruto else ([], [])
    palavras_filtradas, contagens_filtradas = zip(*top_filtrado) if top_filtrado else ([], [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.barh(list(palavras_brutas), list(contagens_brutas), color="steelblue")
    ax1.set_title(f"Top {top_n} Palavras (Antes)", fontsize=12)
    ax1.invert_yaxis()

    ax2.barh(list(palavras_filtradas), list(contagens_filtradas), color="darkorange")
    ax2.set_title(f"Top {top_n} Palavras (Depois)", fontsize=12)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grafico comparativo de frequencia salvo em: %s", caminho_saida)
