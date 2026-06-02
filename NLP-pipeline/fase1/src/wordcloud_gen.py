import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import obter_stopwords
from fase1_config import (
    CAMINHO_NUVEM_PALAVRAS,
    LARGURA_NUVEM_PALAVRAS,
    ALTURA_NUVEM_PALAVRAS,
    MAXIMO_PALAVRAS_NUVEM,
    PALETA_CORES_NUVEM,
    COR_FUNDO_NUVEM,
    TAMANHO_MINIMO_FONTE_NUVEM,
    TAMANHO_MAXIMO_FONTE_NUVEM,
    SEMENTE_NUVEM_PALAVRAS,
)
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def _validar_parametros_wordcloud(
    largura,
    altura,
    maximo_palavras,
    paleta_cores,
    cor_fundo,
    tamanho_minimo_fonte,
    tamanho_maximo_fonte,
    random_state,
):
    valores_inteiros_positivos = {
        "largura": largura,
        "altura": altura,
        "maximo_palavras": maximo_palavras,
        "tamanho_minimo_fonte": tamanho_minimo_fonte,
        "tamanho_maximo_fonte": tamanho_maximo_fonte,
    }

    for nome, valor in valores_inteiros_positivos.items():
        if not isinstance(valor, int) or valor <= 0:
            raise ValueError(f"Parametro '{nome}' deve ser inteiro positivo")

    if tamanho_maximo_fonte < tamanho_minimo_fonte:
        raise ValueError("Parametro 'tamanho_maximo_fonte' deve ser >= 'tamanho_minimo_fonte'")

    valores_textuais = {
        "paleta_cores": paleta_cores,
        "cor_fundo": cor_fundo,
    }
    for nome, valor in valores_textuais.items():
        if not isinstance(valor, str) or not valor.strip():
            raise ValueError(f"Parametro '{nome}' deve ser string nao vazia")

    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("Parametro 'random_state' deve ser inteiro ou None")


def gerar_nuvem_palavras(
    tokens,
    largura=None,
    altura=None,
    maximo_palavras=None,
    paleta_cores=None,
    cor_fundo=None,
    tamanho_minimo_fonte=None,
    tamanho_maximo_fonte=None,
    random_state=None,
    incluir_stopwords=False,
    caminho_saida=None,
):
    if caminho_saida is None:
        caminho_saida = CAMINHO_NUVEM_PALAVRAS

    if largura is None:
        largura = LARGURA_NUVEM_PALAVRAS
    if altura is None:
        altura = ALTURA_NUVEM_PALAVRAS
    if maximo_palavras is None:
        maximo_palavras = MAXIMO_PALAVRAS_NUVEM
    if paleta_cores is None:
        paleta_cores = PALETA_CORES_NUVEM
    if cor_fundo is None:
        cor_fundo = COR_FUNDO_NUVEM
    if tamanho_minimo_fonte is None:
        tamanho_minimo_fonte = TAMANHO_MINIMO_FONTE_NUVEM
    if tamanho_maximo_fonte is None:
        tamanho_maximo_fonte = TAMANHO_MAXIMO_FONTE_NUVEM
    if random_state is None:
        random_state = SEMENTE_NUVEM_PALAVRAS

    _validar_parametros_wordcloud(
        largura,
        altura,
        maximo_palavras,
        paleta_cores,
        cor_fundo,
        tamanho_minimo_fonte,
        tamanho_maximo_fonte,
        random_state,
    )

    def _obter_texto_token(token):
        return str(token.get("processado", token.get("lema", token.get("texto", token.get("text", "")))))

    if incluir_stopwords:
        texto_palavras = " ".join(
            _obter_texto_token(token) for token in tokens if _obter_texto_token(token).strip()
        )
    else:
        stopwords_atuais = obter_stopwords()
        texto_palavras = " ".join(
            texto_token for token in tokens
            if (texto_token := _obter_texto_token(token)).lower() not in stopwords_atuais and texto_token.strip()
        )

    if not texto_palavras.strip():
        logger.warning("Nenhum token para gerar wordcloud")
        return

    nuvem = WordCloud(
        prefer_horizontal=1,
        width=largura,
        height=altura,
        max_words=maximo_palavras,
        colormap=paleta_cores,
        background_color=cor_fundo,
        min_font_size=tamanho_minimo_fonte,
        max_font_size=tamanho_maximo_fonte,
        random_state=random_state,
    ).generate(texto_palavras)

    plt.figure(figsize=(16, 8))
    plt.imshow(nuvem, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("WordCloud salva em: %s", caminho_saida)


def gerar_nuvem_palavras_por_pos(tokens, filtro_pos, caminho_saida=None):
    if caminho_saida is None:
        caminho_saida = CAMINHO_NUVEM_PALAVRAS

    tokens_filtrados = [token for token in tokens if token.get("pos") == filtro_pos]
    logger.info("Gerando wordcloud para POS=%s (%d tokens)", filtro_pos, len(tokens_filtrados))
    gerar_nuvem_palavras(tokens_filtrados, caminho_saida=caminho_saida)
