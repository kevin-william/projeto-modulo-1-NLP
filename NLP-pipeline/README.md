# NLP Pipeline — Wikipedia Articles (Português)

Pipeline modular de Processamento de Linguagem Natural sobre artigos da Wikipedia em português, construído com spaCy, scikit-learn, Gensim e NetworkX. O projeto é organizado em **fases independentes e encadeadas**: o output de cada fase alimenta o input da próxima.

```
artigos_wikipedia.txt
        │
        ▼
  [ FASE 1 ] ── Pré-processamento, POS Tagging, NER, WordCloud
        │
        │  artigos_wikipedia_100_formatado-v001_{metodo}.parquet
        ▼
  [ FASE 2 ] ── BoW / TF-IDF / Word2Vec / c-TF-IDF + Busca por Similaridade
        │
        │  fase2_artifact.lpf2 (BoW + TF-IDF + documentos + tokens)
        ▼
  [ FASE 3 ] ── Modelagem de Tópicos (LSA, LDA, NMF) + EDA + pyLDAvis
        │
        │  Parquet da Fase 1 + artefato .lpf2 da Fase 2
        ▼
  [ FASE 4 ] ── Extração de Informação + Grafo de Conhecimento
```

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Instalação](#instalação)
3. [Formato do Corpus](#formato-do-corpus)
4. [Fase 1 — Pré-processamento e POS Tagging](#fase-1--pré-processamento-e-pos-tagging)
5. [Fase 2 — Embeddings e Busca por Similaridade](#fase-2--embeddings-e-busca-por-similaridade)
6. [Fase 3 — Modelagem de Tópicos](#fase-3--modelagem-de-tópicos)
7. [Fase 4 — Extração de Informação e Grafo de Conhecimento](#fase-4--extração-de-informação-e-grafo-de-conhecimento)
8. [Testes](#testes)
9. [Estrutura do Projeto](#estrutura-do-projeto)

---

## Visão Geral

Cada fase implementa uma etapa do pipeline e produz artefatos consumidos pela fase seguinte. A comunicação entre fases é feita por **arquivos no disco** (parquet, `.lpf2`, CSVs) e serializada por utilitários em `shared/`.

| Fase | Entrada | Processamento | Saída |
|------|---------|---------------|-------|
| **1** | `artigos_wikipedia.txt` | spaCy (`pt_core_news_lg`): tokenização, POS, NER, lematização/stemming, stopwords, WordCloud | `*.parquet` token-a-token + gráficos + JSON de vocabulário |
| **2** | parquet da Fase 1 | scikit-learn (BoW, TF-IDF) + Gensim (Word2Vec) + TF-IDF class-based (c-TF-IDF) + busca por cosseno | t-SNE, grid de WordClouds c-TF-IDF, **artefato `.lpf2`** |
| **3** | artefato `.lpf2` da Fase 2 | TruncatedSVD (LSA) + Gensim LDA + sklearn NMF + EDA textual + métricas (coerência, perplexidade) | CSVs comparativos, gráficos, visualização `pyLDAvis` |
| **4** | parquet da Fase 1 + `.lpf2` da Fase 2 | spaCy (NER, SVO) + regex + Levenshtein (fuzzy) + NetworkX (grafo + centralidade) | CSVs de entidades/relações, JSON de métricas, grafo PNG, relatório interpretativo |

Módulo `shared/`:

- `shared/utils.py` — utilitários genéricos (garantia de diretórios).
- `shared/artifacts.py` — `ArtifactFase2`, dataclass serializado via `joblib` que carrega matrizes BoW/TF-IDF, vectorizers, tokens e metadados da Fase 2 para a Fase 3.

---

## Instalação

### Pré-requisitos

- Python 3.8+
- pip

### Instalando as dependências

```bash
# Dependências globais (comuns a todas as fases)
pip install -r requirements.txt

# Dependências específicas de cada fase
pip install -r fase1/requirements.txt
pip install -r fase2/requirements.txt
pip install -r fase3/requirements.txt
pip install -r fase4/requirements.txt
```

### Baixar o modelo spaCy

As Fases 1 e 4 requerem o modelo grande de português do spaCy (a Fase 4 tem fallback para `pt_core_news_sm`):

```bash
python -m spacy download pt_core_news_lg
```

---

## Formato do Corpus

O arquivo de entrada da Fase 1 (`fase1/input/artigos_wikipedia_100_formatado.txt` por padrão) deve seguir o formato delimitado abaixo. Cada artigo é envolvido por marcadores de início e fim:

```
===== ARTICLE START =====
Title: Inteligência Artificial
URL: https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial
=========================
A inteligência artificial (IA) é a inteligência similar à humana...
===== ARTICLE END =====

===== ARTICLE START =====
Title: Processamento de Linguagem Natural
URL: https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural
=========================
O processamento de linguagem natural (PLN) é uma subárea...
===== ARTICLE END =====
```

> O parser extrai automaticamente `Title`, `URL` e o corpo do artigo.

---

## Fase 1 — Pré-processamento e POS Tagging

**Localização:** `fase1/`

### O que faz

A Fase 1 processa o corpus bruto e produz um DataFrame anotado com informações linguísticas para cada token:

| Etapa | Descrição |
|-------|-----------|
| **1. Carregamento** | Lê e parseia o arquivo de artigos, extrai `Title`, `URL` e conteúdo de cada artigo |
| **1b. Filtro de tamanho** | Remove artigos com menos de `MINIMO_PALAVRAS_ARTIGO` palavras (padrão: 100); log indica quantos foram removidos |
| **2. Normalização** | Aplica lowercase, remove caracteres especiais e excesso de espaços (acentos preservados) |
| **3. POS Tagging** | Aplica o modelo `pt_core_news_lg` via `nlp.pipe()` em lotes, com tokenização por palavra |
| **4. Tabela comparativa** | Gera CSV com stemming × lematização para todos os artigos |
| **5. Análise de Vocabulário** | Compara o vocabulário antes e depois da remoção de stopwords e filtra por `POS_TAGS_PERMITIDOS` |
| **6. Distribuição POS** | Gera gráfico de barras com a frequência de cada classe gramatical |
| **7. Comparativo de Frequência** | Plota as top-N palavras com/sem stopwords |
| **8. WordCloud** | Gera nuvens de palavras (com e sem stopwords) a partir do `processado` |

A pipeline é executada para **cada método** em `METODOS_PROCESSAMENTO_TOKENS`. Cada execução gera um conjunto de artefatos com sufixo `_{metodo}`.

### Como executar

```bash
cd fase1/src
python main.py
```

> Os caminhos são resolvidos relativamente ao diretório `fase1/`, então **não é necessário ajustar nenhum path** antes de rodar.

### Configuração

Edite `fase1/src/fase1_config.py` para ajustar o comportamento:

```python
SEED_ALEATORIO = 42                          # reprodutibilidade
MODELO_SPACY = "pt_core_news_lg"             # modelo spaCy a usar
TAMANHO_LOTE = 5                             # artigos por lote (ajuste conforme RAM)
MINIMO_PALAVRAS_ARTIGO = 100                 # artigos com menos palavras são removidos

# Métodos executados em sequência: 'lemmatizacao', 'stemming' ou 'none'
METODOS_PROCESSAMENTO_TOKENS = ['lemmatizacao', 'none']
METODO_STEMMING = 'snowball'                 # 'snowball' (produção) ou 'porter' (didático)
FONTE_STOPWORDS = 'spacy'                    # 'spacy', 'nltk' ou 'ambas'
STOPWORDS_EXTRAS = []                        # stopwords adicionais
HABILITAR_REMOCAO_STOPWORDS = True

# Whitelist de POS tags no parquet de saída. Lista vazia desabilita o filtro.
POS_TAGS_PERMITIDOS = ["NOUN", "VERB", "ADJ", "ADV"]

# WordCloud
LARGURA_NUVEM_PALAVRAS = 1200
ALTURA_NUVEM_PALAVRAS = 600
MAXIMO_PALAVRAS_NUVEM = 70
PALETA_CORES_NUVEM = "magma"
COR_FUNDO_NUVEM = "white"
TAMANHO_MINIMO_FONTE_NUVEM = 20
TAMANHO_MAXIMO_FONTE_NUVEM = 160
SEMENTE_NUVEM_PALAVRAS = SEED_ALEATORIO
```

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase1/input/artigos_wikipedia_100_formatado.txt` | Corpus de artigos no formato descrito acima |

### Output

Todos os arquivos são salvos em `fase1/output/` com sufixo `_{metodo}` por método:

| Arquivo | Descrição |
|---------|-----------|
| `artigos_wikipedia_100_formatado-v001_{metodo}.parquet` | DataFrame com uma linha por token, filtrado por `POS_TAGS_PERMITIDOS` e stopwords |
| `artigos_wikipedia_100_formatado-v001_{metodo}.png` | Nuvem de palavras com stopwords |
| `artigos_wikipedia_100_formatado-v001_{metodo}_filtered.png` | Nuvem de palavras sem stopwords |
| `pos_distribution_{metodo}.png` | Gráfico de barras com a distribuição das POS tags |
| `freq_comparison_{metodo}.png` | Comparativo das palavras mais frequentes antes e depois da filtragem |
| `artigos_wikipedia_100_formatado-v001_{metodo}.json` | Métricas de vocabulário (totais, redução percentual, top-20 bruto/filtrado) |
| `artigos_wikipedia_100_formatado_comparacao_stemming_lematizacao.csv` | Tabela token × stem × lema para todos os artigos |
| `nlp_artigos_wikipedia_100_formatado-v001.log` | Log completo de execução com timestamps |

### Colunas do Parquet gerado

```
id_artigo            – índice do artigo (1-based)
id_token             – posição do token no documento
token                – texto original do token
pos                  – classe gramatical universal (NOUN, VERB, ADJ, ...)
tag                  – etiqueta morfológica detalhada
lema                 – forma lematizada do token
processado           – token após método configurado (`none`, `lemmatizacao`, `stemming`)
is_stop              – bool indicando se o token é stopword
relacao_dependencia  – relação de dependência sintática (nsubj, obj, ...)
token_cabeca         – token cabeça na árvore de dependência
entidade             – texto da entidade nomeada (vazio se não for entidade)
rotulo_entidade      – tipo da entidade (PER, ORG, LOC, ...)
tipo_tokenizacao     – tipo de tokenização aplicada
titulo               – título do artigo de origem
url                  – URL do artigo de origem
formato              – metadado de formato do token
```

### Stopwords customizadas

Para adicionar palavras à lista de stopwords do modelo em tempo de execução, importe `adicionar_stopwords_personalizadas` de `fase1/src/preprocessing.py`:

```python
from preprocessing import adicionar_stopwords_personalizadas
adicionar_stopwords_personalizadas(["exemplo", "palavra", "123"])
```

---

## Fase 2 — Embeddings e Busca por Similaridade

**Localização:** `fase2/`

### O que faz

A Fase 2 consome o parquet produzido pela Fase 1, treina representações vetoriais dos documentos, calcula c-TF-IDF por categoria, gera visualização t-SNE e oferece uma interface de busca interativa via terminal:

| Método | Implementação | Descrição |
|--------|--------------|-----------|
| **BOW** | `CountVectorizer` (scikit-learn) | Vetores esparsos de contagem de ocorrências por termo |
| **TF-IDF** | `TfidfVectorizer` (scikit-learn) | Pesos TF-IDF normalizados (L2) |
| **Word2Vec** | `Word2Vec` (Gensim) | Embeddings densos — vetor de documento = média dos vetores de palavras |
| **c-TF-IDF** | `CalculadoraCTfIdf` (custom) | TF-IDF por classe reutilizando o IDF já treinado; gera grid de WordClouds por categoria |

Após o treinamento, o sistema gera uma visualização **t-SNE** dos embeddings, calcula **c-TF-IDF** por categoria e abre a interface de busca por **similaridade de cosseno**. Se `bow` ou `tfidf` estiverem configurados, o pipeline exporta o **artefato `.lpf2`** consumido pela Fase 3.

### Pré-requisito

A Fase 2 depende do parquet gerado pela Fase 1:

```bash
# Caminho padrão: fase2/input/artigos_wikipedia_100_formatado-v001_lemmatizacao.parquet
# Gere-o executando a Fase 1 com METODOS_PROCESSAMENTO_TOKENS contendo 'lemmatizacao'
# e copie/mova o arquivo para a pasta de input da Fase 2.
```

### Como executar

```bash
cd fase2/src
python main.py
```

O pipeline executa automaticamente:
1. Carregamento e agrupamento do parquet por documento
2. Filtro por `POS_TAGS_PERMITIDOS`, remoção de stopwords, lowercase
3. Treinamento dos vetorizadores configurados em `METODOS_EMBEDDING`
4. Cálculo de c-TF-IDF por categoria (se `HABILITAR_CTFIDF = True` e houver ≥ 2 categorias distintas)
5. Geração do gráfico t-SNE (se `HABILITAR_TSNE = True`)
6. Exportação do artefato `fase2_artifact.lpf2` (se houver bow ou tfidf)
7. Abertura da interface de busca interativa

### Configuração

Edite `fase2/src/fase2_config.py` para controlar todos os aspectos da fase:

```python
METODOS_EMBEDDING = ["tfidf", "word2vec", "bow"]   # ordem de treino e busca
TOP_K_RESULTADOS = 5                                # resultados por busca

PARAMS_BOW = {"max_features": 5000, "min_df": 1, "max_df": 0.8, "stop_words": STOPWORDS_BOW_TFIDF}
PARAMS_TFIDF = {"max_features": 5000, "min_df": 1, "max_df": 0.8, "norm": "l2", "stop_words": STOPWORDS_BOW_TFIDF}
PARAMS_WORD2VEC = {"vector_size": 100, "window": 5, "min_count": 5, "epochs": 50, "seed": 42, "workers": 2}

HABILITAR_REMOCAO_STOPWORDS = True
POS_TAGS_PERMITIDOS = ["NOUN", "VERB", "ADJ", "ADV"]   # None/[] desabilita o filtro

HABILITAR_TSNE = True
PARAMS_TSNE = {"n_components": 2, "perplexity": 5, "n_iter": 2000, "init": "pca", "random_state": 0}
PARAMS_PLOT_TSNE = {"figsize": (24, 18), "dpi": 600, "marker_size": 50, "annotate_fontsize": 7}

HABILITAR_CTFIDF = True
CATEGORIA_PADRAO = "geral"
REGRAS_CATEGORIAS = [                                  # palavras-chave no título → categoria
    (["linguagem", "lingua", "idioma", "..."], "linguagem"),
    (["tecnologia", "tech", "computador", "..."], "tecnologia"),
    (["calendario", "ano", "mes", "..."], "calendario"),
]
```

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase2/input/artigos_wikipedia_100_formatado-v001_lemmatizacao.parquet` | Parquet token-a-token gerado pela Fase 1 (método `lemmatizacao`) |

### Output

Todos os arquivos são salvos em `fase2/output/`:

| Arquivo | Descrição |
|---------|-----------|
| `artigos_wikipedia_100_formatado-v001_lemmatizacao.png` | Visualização 2D via t-SNE com rótulos de documento |
| `artigos_wikipedia_100_formatado-v001_lemmatizacao_ctfidf_wordclouds.png` | Grid de WordClouds por categoria (c-TF-IDF) |
| `artifacts/fase2_artifact.lpf2` | Artefato serializado (joblib) com matrizes BoW/TF-IDF, vectorizers, tokens, títulos, parâmetros — consumido pela Fase 3 |
| `artigos_wikipedia_100_formatado-v001_lemmatizacao.log` | Log detalhado do treinamento e buscas |

### Interface de Busca (CLI)

Após o treinamento, o terminal exibe um prompt interativo:

```
============================================================
  SISTEMA DE BUSCA TEXTUAL POR SIMILARIDADE
============================================================
  Metodos disponiveis: tfidf, word2vec, bow
  Comandos:
    <consulta>           - busca usando o primeiro metodo
    <metodo> <consulta>  - busca com metodo especifico
    sair                 - encerra
============================================================

> inteligencia artificial
Buscando com [tfidf]: 'inteligencia artificial'
--------------------------------------------------
  #1 [Score: 0.8721] Doc #2
      inteligência artificial sistemas especialistas...
  ...
```

**Sintaxe dos comandos:**

| Comando | Comportamento |
|---------|--------------|
| `<consulta>` | Busca com o primeiro método da lista (`METODOS_EMBEDDING[0]`) |
| `<metodo> <consulta>` | Busca com o método especificado (`bow`, `tfidf` ou `word2vec`) |
| `sair` | Encerra a interface |

O score exibido é a **similaridade de cosseno** entre o vetor da consulta e o vetor de cada documento (0 a 1, quanto maior melhor).

---

## Fase 3 — Modelagem de Tópicos

**Localização:** `fase3/`

### O que faz

A Fase 3 consome o artefato `.lpf2` gerado pela Fase 2 e aplica técnicas de modelagem de tópicos para descobrir temas latentes no corpus:

| Método | Implementação | Matriz de entrada | Métrica |
|--------|--------------|-------------------|---------|
| **LSA** | `TruncatedSVD` (scikit-learn) | TF-IDF | coerência c_v (Gensim) |
| **LDA** | Gensim | Tokens do artefato (Dictionary) | coerência c_v, perplexidade |
| **NMF** | `NMF` (scikit-learn, `nndsvda`) | TF-IDF | coerência c_v |

O pipeline também realiza uma **análise exploratória textual (EDA)** e gera uma visualização interativa com **pyLDAvis** para o LDA.

### Pré-requisito

A Fase 3 depende do artefato `.lpf2` gerado pela Fase 2:

```bash
# O artefato é gerado em fase2/output/artifacts/fase2_artifact.lpf2
# Copie-o para fase3/input/ antes de executar
copy fase2\output\artifacts\fase2_artifact.lpf2 fase3\input\
```

### Como executar

```bash
cd fase3/src
python main.py
```

O pipeline executa automaticamente:
1. Carregamento e validação do artefato `.lpf2`
2. EDA textual (comprimento de docs, vocabulário, Zipf, top termos)
3. Treinamento dos três modelos de tópicos
4. Cálculo de coerência (Gensim `c_v`) e perplexidade (LDA)
5. Geração de gráficos, heatmaps, wordclouds por tópico e `pyLDAvis`
6. Exportação de `comparacao_modelos.csv` e sumário no log

### Configuração

Edite `fase3/src/fase3_config.py` para controlar todos os aspectos da fase:

```python
NUM_TOPICOS = 10
TOP_N_PALAVRAS = 10

PARAMS_LSA = {"random_state": 42}

PARAMS_LDA_GENSIM = {
    "passes": 20,
    "iterations": 1000,
    "alpha": "auto",
    "eta": "auto",
    "random_state": 42,
}
PARAMS_DICTIONARY = {"no_below": 3, "no_above": 0.80, "keep_n": 1000}

PARAMS_NMF = {"init": "nndsvda", "random_state": 42, "max_iter": 500}
```

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase3/input/fase2_artifact.lpf2` | Artefato da Fase 2 (matrizes BoW/TF-IDF, vectorizers, tokens, títulos) |

### Output

Todos os arquivos são salvos em `fase3/output/` (gráficos em `fase3/output/plots/`):

| Arquivo | Descrição |
|---------|-----------|
| `comparacao_modelos.csv` | Tabela LSA × LDA × NMF com coerência e perplexidade |
| `fase3_pipeline-v001.log` | Log detalhado da execução |
| `plots/eda_*.png` | Gráficos da EDA textual (comprimento, top termos, Zipf, boxplot) |
| `plots/{lsa,lda,nmf}_topicos.png` | Top palavras por tópico com pesos |
| `plots/{lsa,lda,nmf}_heatmap.png` | Heatmap documento × tópico |
| `plots/{lsa,lda,nmf}_wordcloud.png` | WordCloud dos tópicos |
| `plots/comparacao_coerencia.png` | Gráfico comparativo de coerência entre modelos |
| `plots/pyldavis.html` | Visualização interativa `pyLDAvis` do LDA |

---

## Fase 4 — Extração de Informação e Grafo de Conhecimento

**Localização:** `fase4/`

### O que faz

A Fase 4 realiza extração de informação, normalização de entidades e construção de um **grafo de conhecimento** sobre o corpus, respondendo à pergunta analítica:

> *"Quais entidades são mais centrais no corpus e por quê?"*

| Etapa | Componente | Descrição |
|-------|-----------|-----------|
| **1** | `TextExtractor` + `pd.read_parquet` | Carrega o corpus do parquet da Fase 1 (reconstruindo documentos e títulos a partir do `.lpf2` da Fase 2, se disponível) |
| **2** | NER spaCy (`pt_core_news_lg`) | Extrai entidades nomeadas (PER, ORG, LOC, GPE, MISC) com visualização displaCy |
| **3** | Regex (`re`) | Extrai padrões: emails, URLs, datas, CPFs, valores monetários (R$), códigos |
| **4** | Dependências sintáticas spaCy | Extrai triplas SVO (sujeito-verbo-objeto) por janela de sentenças |
| **5** | `EntityNormalizer` (Levenshtein) | Agrupa variações ortográficas das entidades em formas canônicas |
| **6** | `KnowledgeGraphBuilder` (NetworkX) | Constrói grafo (≥ 20 nós) e calcula centralidade (betweenness, degree, eigenvector, closeness) |
| **7** | `visualizacao_grafo.py` | Gera PNG do grafo, distribuição de centralidade, frequência por tipo, comunidades, infográfico resumo, e relatório interpretativo |

### Pré-requisito

A Fase 4 depende dos artefatos das Fases 1 e 2:

```bash
# 1. Executar Fase 1
cd fase1/src && python main.py

# 2. Executar Fase 2
cd ../../fase2/src && python main.py

# 3. Copiar entradas para fase4/input/ (a config já procura primeiro local, depois em ../faseX/output/)
copy fase1\output\*_lemmatizacao_palavra.parquet fase4\input\
copy fase2\output\artifacts\fase2_artifact.lpf2 fase4\input\
```

### Como executar

```bash
cd fase4/src
python main.py
```

> Há também um notebook `fase4/notebook_fase4.ipynb` com a mesma análise em modo interativo.

### Configuração

Edite `fase4/src/fase4_config.py`:

```python
# Regex de extração de padrões
REGEX_EMAILS = r"[^\s]+@[a-zA-Z0-9\.]+\.[a-zA-Z]+"
REGEX_URLS   = r"https?://(?:www\.)?[a-zA-Z0-9\-_.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"
REGEX_DATAS  = r"\b\d{2}/\d{2}/\d{4}\b"
REGEX_CPFS   = r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"
# ... REGEX_VALORES, REGEX_CODIGOS

# spaCy
MODELO_SPACY = "pt_core_news_lg"
MODELO_SPACY_FALLBACK = "pt_core_news_sm"
AMOSTRA_DISPLACY = 5
ENTIDADES_VALIDAS_GRAFO = {"PER", "ORG", "LOC", "GPE", "MISC", ...}

# Fuzzy matching (Levenshtein)
MAX_LEVENSHTEIN_DISTANCE = 2
NORMALIZAR_CASE = True
REMOVER_ACENTOS = True

# Relações SVO
ENTIDADES_RELACAO = {"ORG", "PERSON", "GPE", "LOC", "PER"}
WINDOW_MAX_SENTENCAS = 5

# Grafo
MINIMO_FREQUENCIA_ENTIDADE = 2
NUMERO_MINIMO_NOS_GRAFO = 20
TOP_CENTRALIDADE = 10
METRICAS_CENTRALIDADE = ["betweenness", "degree", "eigenvector", "closeness"]
```

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase4/input/100-artigos_anotacao_lg_lemmatizacao_palavra.parquet` | Parquet da Fase 1 (procura local, depois `../fase1/output/`) |
| `fase4/input/fase2_artifact.lpf2` | Artefato da Fase 2 (procura local, depois `../fase2/output/artifacts/`) |

### Output

Todos os arquivos são salvos em `fase4/output/` (gráficos em `fase4/output/plots/`, renderizações displaCy em `fase4/output/displacy/`):

| Arquivo | Descrição |
|---------|-----------|
| `entidades_extraidas.csv` | Entidades NER extraídas do corpus |
| `padroes_extraidos.csv` | Padrões regex extraídos (emails, URLs, datas, CPFs, ...) |
| `relacoes_extraidas.csv` | Triplas SVO extraídas |
| `fuzzy_matches.csv` | Mapeamento de variações → forma canônica (Levenshtein) |
| `nos_grafo.csv` | Nós do grafo (entidade, tipo, frequência, centralidade) |
| `relacoes_grafo.csv` | Arestas do grafo (relações com peso) |
| `resultados.json` | Métricas consolidadas e resposta à pergunta analítica |
| `relatorio_interpretativo.txt` | Relatório textual com top entidades e justificativa |
| `fase4_pipeline.log` | Log detalhado da execução |
| `plots/grafo_conhecimento.png` | Visualização do grafo com matplotlib |
| `plots/centralidade_entidades.png` | Distribuição de centralidade de betweenness |
| `plots/comparacao_centralidades.png` | Comparação entre métricas de centralidade |
| `plots/frequencia_entidades.png` | Top-N entidades por frequência |
| `plots/comunidades_grafo.png` | Comunidades detectadas no grafo |
| `plots/infografico_resumo.png` | Infográfico-resumo da execução |
| `displacy/amostra_*.html` | Renderizações HTML de NER com displaCy |

---

## Testes

Cada fase possui uma suíte de testes independente com pytest:

```bash
# Apenas Fase 1
pytest fase1/tests/ -v

# Apenas Fase 2
pytest fase2/tests/ -v

# Apenas Fase 3
pytest fase3/tests/ -v

# Apenas Fase 4
pytest fase4/tests/ -v

# Todas as fases
pytest fase1/tests/ fase2/tests/ fase3/tests/ fase4/tests/ -v
```

**Cobertura de testes:**

| Fase | Módulos testados |
|------|-----------------|
| Fase 1 | `corpus_loader`, `preprocessing`, `pos_tagger`, `wordcloud_gen` + configuração da WordCloud + pipeline completo |
| Fase 2 | `bow_vectorizer`, `tfidf_vectorizer`, `ctfidf_vectorizer`, `word2vec_vectorizer`, `cosine_search`, `search_interface`, pipeline completo, exportação de artefato |
| Fase 3 | `carregador_artefato`, `eda`, `lsa`, `lda`, `nmf`, `avaliacao` + pipeline completo |
| Fase 4 | `extracao_padroes`, `fuzzy_matching`, `ner_analysis`, `relacoes`, `grafo_conhecimento` + pipeline completo |

---

## Estrutura do Projeto

```
NLP-pipeline/
│
├── requirements.txt              # dependências globais
├── README.md
│
├── shared/
│   ├── __init__.py
│   ├── utils.py                  # utilitários compartilhados entre fases
│   └── artifacts.py              # ArtifactFase2 — serialização de artefatos entre fases (joblib)
│
├── fase1/                        # Fase 1: Pré-processamento e POS Tagging
│   ├── requirements.txt
│   ├── input/
│   │   └── artigos_wikipedia_100_formatado.txt
│   ├── output/                   # gerado após execução
│   │   ├── artigos_wikipedia_100_formatado-v001_{metodo}.parquet
│   │   ├── *.png                 # nuvens e gráficos POS/frequência
│   │   ├── *.json                # métricas de vocabulário
│   │   ├── *_comparacao_stemming_lematizacao.csv
│   │   └── *.log
│   ├── src/
│   │   ├── main.py               # ponto de entrada
│   │   ├── fase1_config.py       # configurações e paths
│   │   ├── corpus_loader.py      # parsing do corpus
│   │   ├── preprocessing.py      # stopwords, stemming, lematização
│   │   ├── pos_tagger.py         # POS tagging e NER com spaCy
│   │   ├── vocab_analysis.py     # análise e gráficos de vocabulário
│   │   ├── wordcloud_gen.py      # geração de nuvem de palavras
│   │   └── logger.py
│   └── tests/
│       ├── test_corpus_loader.py
│       ├── test_preprocessing.py
│       ├── test_pos_tagger.py
│       ├── test_wordcloud_gen.py
│       ├── test_wordcloud_config.py
│       └── test_pipeline.py
│
├── fase2/                        # Fase 2: Embeddings e Busca por Similaridade
│   ├── requirements.txt
│   ├── input/
│   │   └── artigos_wikipedia_100_formatado-v001_lemmatizacao.parquet
│   ├── output/                   # gerado após execução
│   │   ├── artifacts/
│   │   │   └── fase2_artifact.lpf2
│   │   ├── *_lemmatizacao.png           # t-SNE
│   │   ├── *_lemmatizacao_ctfidf_wordclouds.png
│   │   └── *.log
│   ├── src/
│   │   ├── main.py                  # ponto de entrada
│   │   ├── fase2_config.py          # configurações e parâmetros
│   │   ├── embedding_pipeline.py    # orquestra vetorizadores, c-TF-IDF, t-SNE e busca
│   │   ├── search_interface.py      # interface CLI interativa
│   │   ├── logger.py
│   │   ├── vectorizers/
│   │   │   ├── bow_vectorizer.py        # Bag-of-Words
│   │   │   ├── tfidf_vectorizer.py      # TF-IDF
│   │   │   ├── ctfidf_vectorizer.py     # c-TF-IDF (class-based)
│   │   │   └── word2vec_vectorizer.py   # Word2Vec (Gensim)
│   │   ├── similarity/
│   │   │   └── cosine_search.py         # busca por similaridade de cosseno
│   │   └── visualization/
│   │       ├── tsne_plot.py             # projeção t-SNE
│   │       └── wordcloud_ctfidf.py      # grid de WordClouds por categoria
│   └── tests/
│       ├── test_bow_vectorizer.py
│       ├── test_tfidf_vectorizer.py
│       ├── test_ctfidf_vectorizer.py
│       ├── test_word2vec_vectorizer.py
│       ├── test_cosine_search.py
│       ├── test_search_interface.py
│       ├── test_embedding_pipeline.py
│       └── test_artifact_export.py
│
├── fase3/                        # Fase 3: Modelagem de Tópicos
│   ├── requirements.txt
│   ├── input/
│   │   └── fase2_artifact.lpf2   # artefato da Fase 2
│   ├── output/
│   │   ├── plots/                # EDA, topicos, heatmaps, wordclouds, pyLDAvis
│   │   ├── comparacao_modelos.csv
│   │   └── *.log
│   ├── src/
│   │   ├── main.py
│   │   ├── fase3_config.py
│   │   ├── carregador_artefato.py    # carrega e valida ArtifactFase2
│   │   ├── eda.py                    # análise exploratória textual
│   │   ├── avaliacao.py              # coerência (Gensim) e comparação de modelos
│   │   ├── visualizacao.py           # gráficos (topicos, heatmaps, wordclouds, pyLDAvis)
│   │   ├── logger.py
│   │   └── modelos_topicos/
│   │       ├── lsa_modelo.py         # LSA (TruncatedSVD)
│   │       ├── lda_modelo.py         # LDA (Gensim)
│   │       └── nmf_modelo.py         # NMF (sklearn)
│   └── tests/
│       ├── test_carregador_artefato.py
│       ├── test_eda.py
│       ├── test_lsa.py
│       ├── test_lda.py
│       ├── test_nmf.py
│       ├── test_avaliacao.py
│       └── test_pipeline.py
│
└── fase4/                        # Fase 4: Extração de Informação e Grafo de Conhecimento
    ├── requirements.txt
    ├── notebook_fase4.ipynb          # versão interativa em Jupyter
    ├── input/                        # parquet da Fase 1 + .lpf2 da Fase 2
    ├── output/                       # gerado após execução
    │   ├── plots/                    # grafo, centralidade, comunidades, infográfico
    │   ├── displacy/                 # renderizações HTML de NER
    │   ├── *.csv                     # entidades, relações, fuzzy, grafo
    │   ├── resultados.json
    │   ├── relatorio_interpretativo.txt
    │   └── *.log
    ├── src/
    │   ├── main.py                       # executar_fase4_principal() — pipeline de 7 etapas
    │   ├── fase4_config.py               # constantes e caminhos
    │   ├── extractor.py                  # TextExtractor (NER + regex + SVO consolidados)
    │   ├── normalizer.py                 # EntityNormalizer (Levenshtein)
    │   ├── graph_builder.py              # KnowledgeGraphBuilder (NetworkX + centralidade)
    │   ├── extracao_padroes.py           # regex: emails, URLs, datas, CPFs, valores, códigos
    │   ├── fuzzy_matching.py             # Levenshtein para entidades
    │   ├── ner_analysis.py               # NER spaCy + displaCy
    │   ├── relacoes.py                   # extração de relações SVO
    │   ├── grafo_conhecimento.py         # NetworkX: construção e centralidade
    │   ├── visualizacao_grafo.py         # matplotlib para grafo
    │   └── logger.py
    └── tests/
        ├── test_extracao_padroes.py
        ├── test_fuzzy_matching.py
        ├── test_ner_analysis.py
        ├── test_relacoes.py
        ├── test_grafo_conhecimento.py
        └── test_pipeline.py
```
