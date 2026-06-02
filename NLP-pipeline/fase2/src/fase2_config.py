import os
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as _sw_mod
STOPWORDS_BOW_TFIDF = list(_sw_mod.words("portuguese"))

# Diretorio raiz da fase2 (usado para construir caminhos relativos do projeto).
DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminho do parquet de entrada gerado pela fase1 para o recorte atual.
CAMINHO_PARQUET_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "artigos_wikipedia_100_formatado-v001_lemmatizacao.parquet")

# Pasta padrao para salvar logs e artefatos da fase2.
DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")

# Caminho do arquivo de log principal da fase2.
CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "artigos_wikipedia_100_formatado-v001_lemmatizacao.log")

# Ordem dos metodos de embedding a treinar e disponibilizar na busca.
METODOS_EMBEDDING = ["tfidf", "word2vec", "bow"]  # Pode ser uma lista com um ou mais metodos (ex.: ["bow"], ["tfidf", "word2vec"], etc.)

# Quantidade padrao de resultados retornados por consulta textual.
TOP_K_RESULTADOS = 5

# Parametros do CountVectorizer (Bag-of-Words).
PARAMS_BOW = {"max_features": 5000, "min_df": 1, "max_df": 0.8, "stop_words": STOPWORDS_BOW_TFIDF}

# Parametros do TfidfVectorizer.
PARAMS_TFIDF = {"max_features": 5000, "min_df": 1, "max_df": 0.8, "norm": "l2", "stop_words": STOPWORDS_BOW_TFIDF}

# Parametros de treino do Word2Vec (Gensim).
PARAMS_WORD2VEC = {"vector_size": 100, "window": 5, "min_count": 5, "epochs": 50, "seed": 42, "workers": 2}

# Habilita/desabilita a remocao de stopwords portuguesas durante o pre-processamento.
HABILITAR_REMOCAO_STOPWORDS = True

# POS tags permitidos na filtragem de tokens (whitelist). None ou lista vazia desabilita o filtro de POS.
POS_TAGS_PERMITIDOS = ["NOUN", "VERB", "ADJ", "ADV"]

# Habilita/desabilita a geracao do grafico t-SNE apos o treino.
HABILITAR_TSNE = True

# Parametros de reducao de dimensionalidade do t-SNE.
PARAMS_TSNE = {"n_components": 2, "perplexity": 5, "n_iter": 2000, "init": "pca", "random_state": 0}

# Parametros visuais do plot t-SNE salvo em arquivo.
PARAMS_PLOT_TSNE = {
    "figsize": (24, 18),
    "dpi": 600,
    "marker_size": 50,
    "annotate_fontsize": 7,
}

# Caminho final da imagem t-SNE gerada no pipeline.
CAMINHO_SAIDA_TSNE = os.path.join(DIRETORIO_SAIDA, "artigos_wikipedia_100_formatado-v001_lemmatizacao.png")

# Habilita/desabilita o calculo de c-TF-IDF e geracao de WordCloud por categoria.
HABILITAR_CTFIDF = True

# Categoria padrao para artigos sem match nas regras de categorizacao.
CATEGORIA_PADRAO = "geral"

# Regras de categorizacao por palavra-chave no titulo (case-insensitive).
# Primeira regra que casar define a categoria do documento.
REGRAS_CATEGORIAS = [
    (
    ["linguagem", "lingua", "idioma", "fala", "escrita",
     "alfabeto", "fonema", "morfema", "sintaxe", "semântica",
     "pragmatica", "lexico", "gramatica", "ortografia", "pontuacao",
     "verbo", "substantivo", "adjetivo", "advérbio", "preposicao",
     "conjuncao", "interjeicao", "pronome", "artigo", "numeral",
     "frase", "oracao", "periodo", "sujeito", "predicado",
     "complemento", "adjunto", "agente_passiva", "voz_ativa", "voz_passiva",
     "indicativo", "subjuntivo", "imperativo", "presente", "preterito",
     "futuro", "flexao", "derivacao", "composicao", "radical",
     "afixo", "prefixo", "sufixo", "desinencia", "vogal_tematica",
     "tema", "ditongo", "tritongo", "hiato", "encontro_consonantal",
     "silaba", "monossilabo", "dissilabo", "trissilabo", "polissilabo",
     "oxitona", "paroxitona", "proparoxitona", "acentuacao", "crase",
     "apostrofo", "hifen", "paronimos", "homonimos", "sinonimos",
     "antonimos", "hiperonimo", "hiponimo", "polissemia", "denotacao",
     "conotacao", "metafora", "metonimia", "catacrese", "ironia",
     "eufemismo", "hipérbole", "prosopopeia", "pleonasmo", "elipse",
     "zeugma", "anafora", "catafora", "coesao", "coerencia",
     "dialogismo", "intertextualidade", "genero_textual", "discurso_direto", "discurso_indireto",
     "discurso_indireto_livre", "linguagem_verbal", "linguagem_nao_verbal", "lingua_materna", "lingua_estrangeira",
     "traducao", "dialeto", "idioleto", "registro_formal", "registro_informal"],
    "linguagem",
),
    (
    ["tecnologia", "tech", "computador", "software", "hardware",
     "programa", "aplicativo", "sistema_operacional", "processador", "memoria_ram",
     "disco_rigido", "ssd", "placa_mae", "placa_video", "fonte_alimentacao",
     "monitor", "teclado", "mouse", "impressora", "scanner",
     "webcam", "alto_falante", "microfone", "roteador", "modem",
     "cabo_rede", "wifi", "bluetooth", "usb", "hdmi",
     "nuvem", "servidor", "rede", "internet", "intranet",
     "navegador", "buscador", "site", "portal", "blog",
     "forum", "rede_social", "email", "chat", "videoconferencia",
     "streaming", "download", "upload", "backup", "firewall",
     "antivirus", "criptografia", "senha", "autenticacao", "biometria",
     "algoritmo", "inteligencia_artificial", "machine_learning", "deep_learning", "rede_neural",
     "big_data", "data_mining", "computacao_nuvem", "internet_coisas", "realidade_virtual",
     "realidade_aumentada", "blockchain", "criptomoeda", "bitcoin", "token",
     "smart_contract", "metaverso", "robotica", "automacao", "drone",
     "sensor", "microcontrolador", "arduino", "raspberry_pi", "linguagem_programacao",
     "codigo_fonte", "compilador", "interprete", "debug", "teste_software",
     "interface", "front_end", "back_end", "banco_dados", "sql",
     "api", "framework", "biblioteca", "repositorio", "versionamento",
     "git", "devops", "agile", "scrum", "waterfall",
     "prototipo", "manutencao"],
    "tecnologia",
),
    (
    ["calendario", "calendario_gregoriano", "ano", "mes", "semana",
     "dia", "hora", "minuto", "segundo", "milissegundo",
     "ano_bissexto", "decada", "seculo", "milenio", "lustro",
     "janeiro", "fevereiro", "marco", "abril", "maio",
     "junho", "julho", "agosto", "setembro", "outubro",
     "novembro", "dezembro", "segunda", "terca", "quarta",
     "quinta", "sexta", "sabado", "domingo", "fim_de_semana",
     "dias_uteis", "feriado", "ponto_facultativo", "data", "era_comum",
     "antes_de_cristo", "depois_de_cristo", "ano_zero", "primavera", "verao",
     "outono", "inverno", "solsticio", "equinocio", "hemisferio_norte",
     "hemisferio_sul", "tropico_cancer", "tropico_capricornio", "linha_equador", "natal",
     "ano_novo", "carnaval", "pascoa", "corpus_christi", "finados",
     "proclamacao_republica", "independencia", "tiradentes", "dia_trabalho", "dia_maes",
     "dia_pais", "dia_criancas", "dia_mulher", "dia_amigo", "aniversario",
     "bodas", "jubileu", "cronograma", "agenda", "planejamento_anual",
     "trimestre", "quadrimestre", "semestre", "bienio", "trienio",
     "quinquenio", "semana_santa", "advento", "quaresma", "lua_nova",
     "lua_crescente", "lua_cheia", "lua_minguante", "calendario_juliano", "calendario_lunar",
     "calendario_solar", "calendario_islamico", "calendario_judaico", "calendario_chines", "calendario_maia",
     "calendario_egipcio", "calendario_romano", "data_juliana"],
    "calendario",
),
]

# Caminho para salvar o grid de WordClouds por categoria (c-TF-IDF).
CAMINHO_SAIDA_WORDCLOUD_CTFIDF = os.path.join(DIRETORIO_SAIDA, "artigos_wikipedia_100_formatado-v001_lemmatizacao_ctfidf_wordclouds.png")
