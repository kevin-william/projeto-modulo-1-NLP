import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from preprocessing import (
    obter_instancia_nlp,
    obter_stopwords,
    obter_stopwords_nltk,
    adicionar_stopwords_personalizadas,
    aplicar_stemming,
    normalizar_texto,
    tokenizar_artigo,
    tokenizar_por_tipo,
    remover_stopwords_dos_tokens,
    tokenizar_sentencas,
    tokenizar_palavras_nltk,
    tokenizar_casual,
    gerar_tabela_comparacao_stemming_lematizacao,
    inicializar_nltk,
)


def test_obter_instancia_nlp_retorna_modelo():
    nlp = obter_instancia_nlp()
    assert nlp is not None
    assert nlp.lang == "pt"


def test_obter_stopwords_nao_vazio():
    stopwords_atuais = obter_stopwords()
    assert len(stopwords_atuais) > 0
    assert "de" in stopwords_atuais or "o" in stopwords_atuais or "a" in stopwords_atuais


def test_adicionar_stopwords_personalizadas():
    adicionar_stopwords_personalizadas(["palavrateste"])
    stopwords_atuais = obter_stopwords()
    assert "palavrateste" in stopwords_atuais


def test_aplicar_stemming_snowball():
    resultado = aplicar_stemming("processamento", metodo='snowball')
    assert isinstance(resultado, str)
    assert len(resultado) > 0
    assert len(resultado) <= len("processamento")


def test_aplicar_stemming_porter():
    resultado = aplicar_stemming("processamento", metodo='porter')
    assert isinstance(resultado, str)
    assert len(resultado) > 0


def test_tokenizar_artigo_none():
    texto = "O processamento de linguagem natural é fascinante."
    resultado = tokenizar_artigo(texto, metodo_processamento='none')
    assert "tokens" in resultado
    assert len(resultado["tokens"]) > 0
    chaves_token = [
        "texto", "lema", "processado", "pos", "tag", "dependencia",
        "eh_stopword", "eh_pontuacao", "eh_alfabetico", "formato"
    ]
    for chave in chaves_token:
        assert chave in resultado["tokens"][0], f"Campo '{chave}' ausente"
    primeiro_token = resultado["tokens"][0]
    assert primeiro_token["processado"] == primeiro_token["texto"]


def test_tokenizar_artigo_lemmatizacao():
    texto = "Os computadores processam dados rapidamente."
    resultado = tokenizar_artigo(texto, metodo_processamento='lemmatizacao')
    tokens = resultado["tokens"]
    for token in tokens:
        assert token["processado"] == token["lema"]


def test_tokenizar_artigo_stemming():
    texto = "O processamento de linguagem natural é fascinante."
    resultado = tokenizar_artigo(texto, metodo_processamento='stemming')
    tokens = resultado["tokens"]
    for token in tokens:
        assert isinstance(token["processado"], str)
        assert len(token["processado"]) > 0
    houve_mudanca = any(
        token["processado"] != token["lema"] for token in tokens if token["eh_alfabetico"]
    )
    assert houve_mudanca, "Stemming deveria alterar ao menos um token"


def test_tokenizar_artigo_compatibilidade_retroativa():
    """tokenizar_artigo sem argumento deve funcionar como antes (metodo_processamento='none')."""
    texto = "O processamento de linguagem natural é fascinante."
    resultado = tokenizar_artigo(texto)
    assert "tokens" in resultado


def test_remover_stopwords_dos_tokens():
    texto = "O processamento de linguagem natural é fascinante."
    resultado = tokenizar_artigo(texto, metodo_processamento='lemmatizacao')
    tokens_filtrados = remover_stopwords_dos_tokens(resultado["tokens"])
    processados = [token["processado"].lower() for token in tokens_filtrados]
    assert "o" not in processados
    assert "de" not in processados
    assert "processamento" in processados or "processar" in processados


# ---------------------------------------------------------------------------
# Testes de normalização textual
# ---------------------------------------------------------------------------

def test_normalizar_texto_lowercase():
    assert normalizar_texto("Texto COM MAIUSCULAS") == "texto com maiusculas"


def test_normalizar_texto_acentos_lowercase():
    resultado = normalizar_texto("Ção Ação")
    assert resultado == "ção ação"


def test_normalizar_texto_remove_caracteres_especiais():
    resultado = normalizar_texto("texto! com. caracteres? especiais# e (parênteses)")
    for char in "!.?#()":
        assert char not in resultado


def test_normalizar_texto_preserva_acentos():
    resultado = normalizar_texto("ação não está correta")
    assert "ação" in resultado
    assert "não" in resultado


# ---------------------------------------------------------------------------
# Testes de token.shape_ (campo "formato")
# ---------------------------------------------------------------------------

def test_tokenizar_por_tipo_campo_formato_presente():
    texto = "O gato preto."
    resultado = tokenizar_por_tipo(texto)
    assert "formato" in resultado["tokens"][0], "Campo 'formato' ausente nos tokens"


def test_tokenizar_por_tipo_campo_formato_tipo_str():
    texto = "O gato preto."
    resultado = tokenizar_por_tipo(texto)
    for token in resultado["tokens"]:
        assert isinstance(token["formato"], str), "Campo 'formato' deve ser string"


# ---------------------------------------------------------------------------
# Testes de stopwords NLTK
# ---------------------------------------------------------------------------

def test_obter_stopwords_nltk_nao_vazio():
    inicializar_nltk()
    sw = obter_stopwords_nltk()
    assert len(sw) > 0
    assert "de" in sw or "o" in sw or "a" in sw


def test_obter_stopwords_fonte_nltk():
    inicializar_nltk()
    sw = obter_stopwords(fonte='nltk')
    assert len(sw) > 0
    assert isinstance(sw, set)


def test_obter_stopwords_fonte_ambas_maior_que_spacy():
    inicializar_nltk()
    sw_spacy = obter_stopwords(fonte='spacy')
    sw_ambas = obter_stopwords(fonte='ambas')
    assert len(sw_ambas) >= len(sw_spacy)


def test_obter_stopwords_fonte_padrao_preserva_comportamento():
    """Sem argumento, deve se comportar como fonte='spacy' (retrocompatibilidade)."""
    sw_default = obter_stopwords()
    sw_spacy = obter_stopwords(fonte='spacy')
    assert sw_default == sw_spacy


# ---------------------------------------------------------------------------
# Testes dos tokenizadores NLTK (demonstração didática)
# ---------------------------------------------------------------------------

def test_tokenizar_sentencas_retorna_lista():
    inicializar_nltk()
    texto = "O gato dormiu. O cachorro acordou."
    resultado = tokenizar_sentencas(texto)
    assert isinstance(resultado, list)
    assert len(resultado) >= 1
    assert all(isinstance(s, str) for s in resultado)


def test_tokenizar_sentencas_segmenta_corretamente():
    inicializar_nltk()
    texto = "O gato dormiu. O cachorro acordou."
    resultado = tokenizar_sentencas(texto)
    assert len(resultado) == 2


def test_tokenizar_palavras_nltk_retorna_lista():
    inicializar_nltk()
    texto = "O processamento de linguagem natural."
    resultado = tokenizar_palavras_nltk(texto)
    assert isinstance(resultado, list)
    assert len(resultado) > 0
    assert all(isinstance(t, str) for t in resultado)


def test_tokenizar_casual_retorna_lista():
    texto = "Olá! Tudo bem? :)"
    resultado = tokenizar_casual(texto)
    assert isinstance(resultado, list)
    assert len(resultado) > 0
    assert all(isinstance(t, str) for t in resultado)


# ---------------------------------------------------------------------------
# Testes da tabela comparação stemming vs lematização
# ---------------------------------------------------------------------------

def test_gerar_tabela_comparacao_colunas():
    texto = "O gato preto correu pelo jardim."
    df = gerar_tabela_comparacao_stemming_lematizacao(texto)
    for coluna in ("palavra", "stemming", "lematização", "classe_gramatical"):
        assert coluna in df.columns, f"Coluna '{coluna}' ausente"


def test_gerar_tabela_comparacao_apenas_palavras_alfabeticas():
    texto = "O gato. Correu!"
    df = gerar_tabela_comparacao_stemming_lematizacao(texto)
    assert all(df["palavra"].str.isalpha()), "Tabela deve conter apenas palavras alfabéticas"


def test_gerar_tabela_comparacao_stemming_menor_ou_igual():
    texto = "Os computadores processam informações linguísticas."
    df = gerar_tabela_comparacao_stemming_lematizacao(texto)
    for _, linha in df.iterrows():
        assert len(linha["stemming"]) <= len(linha["palavra"]) + 1, (
            f"Stemming '{linha['stemming']}' não deveria ser maior que a palavra '{linha['palavra']}'"
        )


def test_gerar_tabela_comparacao_nao_vazia():
    texto = "O processamento de linguagem natural é fascinante."
    df = gerar_tabela_comparacao_stemming_lematizacao(texto)
    assert len(df) > 0


def test_normalizar_texto_excesso_espacos():
    resultado = normalizar_texto("texto   com    muito    espaço")
    assert "  " not in resultado
    assert resultado == resultado.strip()


def test_normalizar_texto_retorna_string():
    assert isinstance(normalizar_texto("qualquer texto"), str)


def test_normalizar_texto_remove_urls():
    resultado = normalizar_texto("acesse https://example.com/pagina e www.site.com.br hoje")
    assert "http" not in resultado
    assert "www" not in resultado
    assert "example" not in resultado
    assert "acesse" in resultado
    assert "hoje" in resultado


def test_normalizar_texto_remove_underscore():
    resultado = normalizar_texto("palavra_composta")
    assert "_" not in resultado


# ---------------------------------------------------------------------------
# Testes de tokenização por tipo
# ---------------------------------------------------------------------------

def test_tokenizar_por_tipo_palavra_retrocompativel():
    texto = "O processamento de linguagem natural é fascinante."
    resultado_tipo = tokenizar_por_tipo(texto)
    resultado_antigo = tokenizar_artigo(texto)
    assert len(resultado_tipo["tokens"]) == len(resultado_antigo["tokens"])


