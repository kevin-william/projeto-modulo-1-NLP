import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def calcular_coerencia_gensim(topicos, tokens, metodo="c_v"):
    """Calcula coerência via Gensim CoherenceModel para qualquer modelo (LSA, NMF, etc.).

    Args:
        topicos: List[List[str]] — lista de listas de palavras por tópico.
        tokens: List[List[str]] — tokens por documento (não strings brutas).
        metodo: métrica de coerência ('c_v', 'u_mass', 'c_npmi', etc.).

    Returns:
        float — coerência média do modelo.
    """
    if not topicos or not tokens:
        return 0.0

    dictionary = Dictionary(tokens)
    cm = CoherenceModel(
        topics=topicos,
        texts=tokens,
        dictionary=dictionary,
        coherence=metodo,
    )
    coerencia = float(cm.get_coherence())
    logger.info("Coerencia (%s): %.4f", metodo, coerencia)
    return coerencia


def comparar_modelos(resultados):
    linhas = []
    for nome_modelo, dados in resultados.items():
        linhas.append({
            "modelo": nome_modelo,
            "num_topicos": dados.get("num_topicos", 0),
            "coerencia_media": dados.get("coerencia", 0.0),
            "perplexidade": dados.get("perplexidade", None),
        })

    dataframe = pd.DataFrame(linhas)
    logger.info("Tabela comparativa gerada com %d modelos", len(dataframe))
    return dataframe
