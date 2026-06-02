import re
from fase1_config import CAMINHO_ENTRADA, MARCADOR_INICIO_ARTIGO, MARCADOR_FIM_ARTIGO, MINIMO_PALAVRAS_ARTIGO
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def carregar_artigos(caminho_arquivo=None):
    if caminho_arquivo is None:
        caminho_arquivo = CAMINHO_ENTRADA

    logger.info("Carregando artigos de: %s", caminho_arquivo)

    with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
        texto = arquivo.read()

    artigos = []
    padrao = re.compile(
        re.escape(MARCADOR_INICIO_ARTIGO)
        + r"\s*"
        + r"Title:\s*(.+?)$\s*"
        + r"URL:\s*(.+?)$\s*"
        + re.escape("=========================")
        + r"\s*"
        + r"(.*?)"
        + r"\s*"
        + re.escape(MARCADOR_FIM_ARTIGO),
        re.MULTILINE | re.DOTALL,
    )

    for correspondencia in padrao.finditer(texto):
        titulo = correspondencia.group(1).strip()
        url = correspondencia.group(2).strip()
        conteudo = correspondencia.group(3).strip()
        artigos.append({"titulo": titulo, "url": url, "conteudo": conteudo})
        logger.info("Artigo carregado: '%s' (%d chars)", titulo, len(conteudo))

    logger.info("Total de artigos carregados: %d", len(artigos))
    return artigos


def obter_estatisticas_corpus(artigos):
    total_artigos = len(artigos)
    tamanhos = [len(artigo["conteudo"]) for artigo in artigos]
    total_caracteres = sum(tamanhos)
    media_caracteres = total_caracteres / total_artigos if total_artigos > 0 else 0
    minimo_caracteres = min(tamanhos) if tamanhos else 0
    maximo_caracteres = max(tamanhos) if tamanhos else 0

    estatisticas = {
        "total_artigos": total_artigos,
        "total_caracteres": total_caracteres,
        "media_caracteres_por_artigo": round(media_caracteres, 2),
        "minimo_caracteres": minimo_caracteres,
        "maximo_caracteres": maximo_caracteres,
    }

    logger.info("Estatisticas do corpus: %s", estatisticas)
    return estatisticas


def filtrar_artigos_por_tamanho(artigos, minimo_palavras=None):
    """
    Remove artigos com menos de N palavras do corpus.

    Args:
        artigos: Lista de dicionários de artigos (titulo, url, conteudo).
        minimo_palavras: Número mínimo de palavras. Usa MINIMO_PALAVRAS_ARTIGO se None.

    Returns:
        Tupla (artigos_validos, artigos_removidos).
    """
    if minimo_palavras is None:
        minimo_palavras = MINIMO_PALAVRAS_ARTIGO

    artigos_validos = []
    artigos_removidos = []

    for artigo in artigos:
        quantidade_palavras = len(artigo["conteudo"].split())
        if quantidade_palavras >= minimo_palavras:
            artigos_validos.append(artigo)
        else:
            artigos_removidos.append(artigo)
            logger.info(
                "Artigo removido por ser muito curto: '%s' (%d palavras, minimo=%d)",
                artigo["titulo"], quantidade_palavras, minimo_palavras,
            )

    logger.info(
        "Filtro minimo de palavras (%d): %d artigos mantidos, %d artigos removidos",
        minimo_palavras, len(artigos_validos), len(artigos_removidos),
    )
    return artigos_validos, artigos_removidos
