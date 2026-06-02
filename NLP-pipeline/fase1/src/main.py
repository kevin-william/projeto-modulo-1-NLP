import sys
import os
import pandas as pd

DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIRETORIO_SCRIPT)
sys.path.insert(0, os.path.join(DIRETORIO_SCRIPT, "..", "..", "shared"))

from collections import Counter
from fase1_config import (
    CAMINHO_PARQUET_SAIDA,
    CAMINHO_NUVEM_PALAVRAS,
    CAMINHO_ANALISE_VOCABULARIO,
    CAMINHO_TABELA_COMPARACAO,
    DIRETORIO_INPUT,
    DIRETORIO_SAIDA,
    METODOS_PROCESSAMENTO_TOKENS,
    POS_TAGS_PERMITIDOS,
)
from shared.utils import ensure_dir
from logger import inicializar_sistema_log
from corpus_loader import carregar_artigos, obter_estatisticas_corpus, filtrar_artigos_por_tamanho
from preprocessing import obter_stopwords, inicializar_nltk, gerar_tabela_comparacao_stemming_lematizacao
from pos_tagger import processar_lote_artigos
from wordcloud_gen import gerar_nuvem_palavras
from vocab_analysis import (
    analisar_vocabulario,
    plotar_distribuicao_pos,
    plotar_comparacao_frequencia,
)

ensure_dir(DIRETORIO_INPUT)
ensure_dir(DIRETORIO_SAIDA)

logger = inicializar_sistema_log("nlp_pipeline")


def _construir_caminho_saida(caminho_base, sufixo_metodo, nova_extensao=None):
    """Insere o sufixo do método antes da extensão do arquivo."""
    raiz, extensao = os.path.splitext(caminho_base)
    if nova_extensao is not None:
        extensao = nova_extensao
    return f"{raiz}{sufixo_metodo}{extensao}"


def executar_pipeline_por_metodo(artigos, metodo):
    """Executa a pipeline completa para um método de processamento de tokens."""
    sufixo = f"_{metodo}"

    logger.info("=" * 60)
    logger.info("METODO: %s", metodo)
    logger.info("=" * 60)

    logger.info("[ETAPA 2] POS Tagging com spaCy")
    dataframe = processar_lote_artigos(
        artigos,
        metodo_processamento=metodo,
        habilitar_stopwords=False,
        pos_permitidos=[],
    )

    dataframe_parquet = dataframe[
        dataframe["pos"].isin(POS_TAGS_PERMITIDOS) & ~dataframe["is_stop"]
    ].copy()
    caminho_parquet = _construir_caminho_saida(CAMINHO_PARQUET_SAIDA, sufixo)
    dataframe_parquet.to_parquet(caminho_parquet, index=False)
    logger.info("DataFrame salvo: %s (Parquet)", caminho_parquet)

    logger.info("[ETAPA 3] Analise de vocabulario (stopwords)")
    stopwords_atuais = obter_stopwords()
    logger.info("Stopwords carregadas: %d palavras", len(stopwords_atuais))

    todos_tokens = dataframe.to_dict("records")

    def eh_nao_stopword(registro_token):
        texto = str(registro_token.get("processado", registro_token.get("lema", ""))).lower()
        pos = str(registro_token.get("pos", ""))
        return texto not in stopwords_atuais and pos != "PUNCT" and texto.strip()

    tokens_brutos_info = [
        {
            "texto": str(token.get("token", "")),
            "lema": str(token.get("lema", token.get("processado", ""))),
            "processado": str(token.get("processado", token.get("lema", token.get("token", "")))),
            "pos": str(token.get("pos", "")),
        }
        for token in todos_tokens
    ]
    tokens_filtrados = [
        {
            "texto": str(token.get("token", "")),
            "lema": str(token.get("lema", token.get("processado", ""))),
            "processado": str(token.get("processado", token.get("lema", token.get("token", "")))),
            "pos": str(token.get("pos", "")),
        }
        for token in todos_tokens
        if eh_nao_stopword(token)
    ]

    contador_bruto = Counter(token["processado"].lower() for token in tokens_brutos_info if token["processado"].strip())
    contador_filtrado = Counter(token["processado"].lower() for token in tokens_filtrados if token["processado"].strip())

    caminho_vocabulario = _construir_caminho_saida(
        CAMINHO_ANALISE_VOCABULARIO, sufixo
    )
    analisar_vocabulario(tokens_brutos_info, tokens_filtrados, caminho_saida=caminho_vocabulario)

    # Step 4: Distribuição de POS Tags
    logger.info("[ETAPA 4] Distribuicao de POS Tags")
    caminho_saida_pos = _construir_caminho_saida(
        os.path.join(DIRETORIO_SAIDA, "pos_distribution.png"), sufixo
    )
    plotar_distribuicao_pos(dataframe_parquet, caminho_saida=caminho_saida_pos)

    # Step 5: Gráfico comparativo de frequência
    logger.info("[ETAPA 5] Grafico comparativo de frequencia")
    caminho_saida_frequencia = _construir_caminho_saida(
        os.path.join(DIRETORIO_SAIDA, "freq_comparison.png"), sufixo
    )
    plotar_comparacao_frequencia(contador_bruto, contador_filtrado, caminho_saida=caminho_saida_frequencia)

    # Step 6: WordCloud
    logger.info("[ETAPA 6] Geracao de WordCloud")
    caminho_saida_nuvem = _construir_caminho_saida(CAMINHO_NUVEM_PALAVRAS, sufixo)
    caminho_saida_nuvem_filtrada = _construir_caminho_saida(CAMINHO_NUVEM_PALAVRAS, f"{sufixo}_filtered")
    gerar_nuvem_palavras(tokens_brutos_info, incluir_stopwords=True, caminho_saida=caminho_saida_nuvem)
    gerar_nuvem_palavras(tokens_filtrados, caminho_saida=caminho_saida_nuvem_filtrada)

    logger.info("Outputs gerados com sufixo '%s'", sufixo)


def executar_pipeline_principal():
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE NLP - Wikipedia Articles")
    logger.info("=" * 60)

    # Inicializar recursos NLTK (downloads automáticos se ausentes)
    inicializar_nltk()

    # Etapa 1: Carregar corpus
    logger.info("[ETAPA 1] Carregamento e inspecao do corpus")
    artigos = carregar_artigos()
    estatisticas = obter_estatisticas_corpus(artigos)
    logger.info("Estatisticas iniciais: %s", estatisticas)

    # Etapa 1b: Filtrar artigos muito curtos
    logger.info("[ETAPA 1b] Filtrando artigos por tamanho minimo de palavras")
    artigos, artigos_removidos = filtrar_artigos_por_tamanho(artigos)
    logger.info("Artigos para processamento: %d", len(artigos))

    # Executar pipeline para cada método de processamento
    logger.info("Metodos de processamento: %s", METODOS_PROCESSAMENTO_TOKENS)

    # Etapa de Comparação: tabela stemming vs lematização (todos os artigos)
    logger.info("[ETAPA COMPARACAO] Gerando tabela stemming vs lematizacao para todos os artigos")
    fragmentos = [
        gerar_tabela_comparacao_stemming_lematizacao(artigo["conteudo"])
        for artigo in artigos
    ]
    tabela_comparacao = pd.concat(fragmentos, ignore_index=True)
    tabela_comparacao.to_csv(CAMINHO_TABELA_COMPARACAO, index=False, encoding='utf-8')
    logger.info("Tabela de comparacao salva: %s (%d linhas)", CAMINHO_TABELA_COMPARACAO, len(tabela_comparacao))

    for metodo in METODOS_PROCESSAMENTO_TOKENS:
        executar_pipeline_por_metodo(artigos, metodo)

    logger.info("=" * 60)
    logger.info("PIPELINE CONCLUIDO COM SUCESSO")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        executar_pipeline_principal()
    except FileNotFoundError as exc:
        logger.error("Erro de arquivo nao encontrado: %s", exc)
        sys.exit(1)
    except OSError as exc:
        logger.error("Erro de sistema de arquivos: %s", exc)
        sys.exit(1)
