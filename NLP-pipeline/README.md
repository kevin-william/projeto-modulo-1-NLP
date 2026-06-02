# NLP Pipeline — Wikipedia Articles (Português)

Pipeline modular de Processamento de Linguagem Natural sobre artigos da Wikipedia em português, construído com spaCy, scikit-learn e Gensim. O projeto é organizado em **fases independentes e encadeadas**: o output de cada fase alimenta o input da próxima.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Instalação](#instalação)
3. [Formato do Corpus](#formato-do-corpus)
4. [Fase 1 — Pré-processamento e POS Tagging](#fase-1--pré-processamento-e-pos-tagging)
5. [Fase 2 — Embeddings e Busca por Similaridade](#fase-2--embeddings-e-busca-por-similaridade)
6. [Fase 3 — Modelagem de Tópicos](#fase-3--modelagem-de-tópicos)
7. [Testes](#testes)
8. [Estrutura do Projeto](#estrutura-do-projeto)

---

## Visão Geral

```
artigos_wikipedia.txt
        │
        ▼
  [ FASE 1 ] ── POS Tagging, NER, WordCloud, Análise de Vocabulário
        │
      │  100-artigos_anotacao_lg.parquet
        ▼
  [ FASE 2 ] ── BOW / TF-IDF / Word2Vec + Busca por Similaridade (CLI)
        │
      │  fase2_artifact.lpf2 (BOW + TF-IDF + documentos)
        ▼
  [ FASE 3 ] ── Modelagem de Tópicos (LDA, LSA, NMF)
        │
        ▼
  [ FASE N ] ── (futuro)
```

---

## Instalação

### Pré-requisitos

- Python 3.8+
- pip

### Instalando as dependências

Você pode instalar as dependências de cada fase individualmente ou tudo de uma vez:

```bash
# Dependências globais (inclui todas as fases)
pip install -r requirements.txt

# Apenas Fase 1
pip install -r fase1/requirements.txt

# Apenas Fase 2
pip install -r fase2/requirements.txt

# Apenas Fase 3
pip install -r fase3/requirements.txt
```

### Baixar o modelo spaCy

A Fase 1 requer o modelo de português grande do spaCy:

```bash
python -m spacy download pt_core_news_lg
```

---

## Formato do Corpus

O arquivo de entrada da Fase 1 (`fase1/input/artigos_wikipedia.txt`) deve seguir o formato delimitado abaixo. Cada artigo é envolvido por marcadores de início e fim:

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
| **1. Carregamento** | Lê e parseia `artigos_wikipedia.txt`, extrai título, URL e conteúdo de cada artigo |
| **1b. Filtro de tamanho** | Remove artigos com menos de `MINIMO_PALAVRAS_ARTIGO` palavras (padrão: 40); log indica quantos foram removidos |
| **2. Normalização** | Aplica lowercase, remove caracteres especiais e excesso de espaços (acentos preservados) |
| **3. POS Tagging** | Aplica o modelo `pt_core_news_lg` via `nlp.pipe()` em lotes, com tokenização por tipo configurado |
| **4. Análise de Vocabulário** | Compara o vocabulário antes e depois da remoção de stopwords |
| **5. Distribuição POS** | Gera gráfico de barras com a frequência de cada classe gramatical |
| **6. Comparativo de Frequência** | Plota as top-N palavras com/sem stopwords |
| **7. WordCloud** | Gera nuvem de palavras a partir dos lemas filtrados |

A pipeline é executada para **cada método** em `METODOS_PROCESSAMENTO_TOKENS`. Uma execução completa é realizada para cada método e os artefatos são salvos com sufixo `_{metodo}`.

### Como executar

```bash
cd fase1/src
python main.py
```

> Os caminhos são resolvidos relativamente ao diretório `fase1/`, então **não é necessário ajustar nenhum path** antes de rodar.

### Configuração

Edite `fase1/src/fase1_config.py` para ajustar o comportamento:

```python
MODELO_SPACY = "pt_core_news_lg"            # modelo spaCy a usar
TAMANHO_LOTE = 5                             # artigos por lote (ajuste conforme RAM disponível)
SEED_ALEATORIO = 42                          # reprodutibilidade
METODOS_PROCESSAMENTO_TOKENS = ["lemmatizacao", "stemming", "none"]
MINIMO_PALAVRAS_ARTIGO = 40                  # artigos com menos palavras são removidos
STOPWORDS_EXTRAS = []                        # stopwords adicionais além das do spaCy

# WordCloud (customizavel via fase1_config.py)
LARGURA_NUVEM_PALAVRAS = 1200
ALTURA_NUVEM_PALAVRAS = 600
MAXIMO_PALAVRAS_NUVEM = 200
PALETA_CORES_NUVEM = "viridis"
COR_FUNDO_NUVEM = "white"
TAMANHO_MINIMO_FONTE_NUVEM = 8
TAMANHO_MAXIMO_FONTE_NUVEM = 150
SEMENTE_NUVEM_PALAVRAS = SEED_ALEATORIO
```

Esses parametros controlam a aparencia da nuvem de palavras da fase1 e sao aplicados automaticamente pela funcao de geracao quando nenhum override e passado na chamada.

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase1/input/artigos_wikipedia.txt` | Corpus de artigos no formato descrito acima |

### Output

Todos os arquivos são salvos em `fase1/output/` com sufixo `_{metodo}` por método:

| Arquivo | Descrição |
|---------|-----------|
| `1100-artigos_wikipedia-formatados-v001_{metodo}.parquet` | DataFrame com uma linha por token, incluindo coluna `processado` |
| `1100-artigos_wikipedia-formatados-v001_{metodo}.png` | Nuvem de palavras gerada a partir dos lemas (sem stopwords) |
| `pos_distribution_{metodo}.png` | Gráfico de barras com a distribuição das POS tags no corpus |
| `freq_comparison_{metodo}.png` | Comparativo das palavras mais frequentes antes e depois da filtragem de stopwords |
| `1100-artigos_wikipedia-formatados-v001_{metodo}.json` | Métricas de vocabulário: total de tokens, tokens únicos, redução percentual após filtragem |
| `nlp_1100-artigos_wikipedia-formatados-v001.log` | Log completo de execução com timestamps |

### Colunas do Parquet gerado

```
id_artigo         – índice do artigo (1-based)
id_token          – posição do token no documento
token             – texto original do token (ou n-grama/sentença)
pos               – classe gramatical universal (NOUN, VERB, ADJ, ...)
tag               – etiqueta morfológica detalhada
lema              – forma lematizada do token
processado        – token após método configurado (`none`, `lemmatizacao`, `stemming`)
relacao_dependencia – relação de dependência sintática (nsubj, obj, ...)
token_cabeca      – token cabeça na árvore de dependência
entidade    – texto da entidade nomeada (vazio se não for entidade)
rotulo_entidade – tipo da entidade (PER, ORG, LOC, DATE, ...)
titulo      – título do artigo de origem
url         – URL do artigo de origem
```

### Stopwords customizadas

Para adicionar palavras à lista de stopwords do modelo, use a função `adicionar_stopwords_personalizadas` de `fase1/src/preprocessing.py`:

```python
from preprocessing import adicionar_stopwords_personalizadas
adicionar_stopwords_personalizadas(["exemplo", "palavra", "123"])
```

---

## Fase 2 — Embeddings e Busca por Similaridade

**Localização:** `fase2/`

### O que faz

A Fase 2 consome o CSV produzido pela Fase 1 e treina três representações vetoriais dos documentos, oferecendo em seguida uma interface de busca interativa via terminal:

| Método | Implementação | Descrição |
|--------|--------------|-----------|
| **BOW** | `CountVectorizer` (scikit-learn) | Vetores de contagem de ocorrências por termo |
| **TF-IDF** | `TfidfVectorizer` (scikit-learn) | Pesos TF-IDF normalizados (L2) |
| **Word2Vec** | `Word2Vec` (Gensim) | Média dos vetores de palavras do documento |

Após o treinamento, o sistema gera uma visualização **t-SNE** dos embeddings e inicia a interface de busca por **similaridade de cosseno**.

### Pré-requisito

A Fase 2 depende do output da Fase 1. Copie (ou mova) o Parquet gerado:

```bash
# Opção A: copiar manualmente
copy fase1\output\100-artigos_anotacao_lg.parquet fase2\input\artigos_anotacao_lg.parquet

# Opção B: o caminho padrão já aponta para fase2/input/artigos_anotacao_lg.parquet
# — basta garantir que o arquivo está lá antes de rodar a Fase 2
```

### Como executar

```bash
cd fase2/src
python main.py
```

O pipeline executa automaticamente:
1. Carregamento e agrupamento do Parquet por documento
2. Treinamento dos vetorizadores configurados em `METODOS_EMBEDDING`
3. Geração do gráfico t-SNE (se `HABILITAR_TSNE = True`)
4. Abertura da interface de busca interativa

### Configuração

Edite `fase2/src/fase2_config.py` para controlar todos os aspectos da fase:

```python
# Métodos a treinar — remova ou reordene conforme necessário
METODOS_EMBEDDING = ["bow", "tfidf", "word2vec"]

# Número de resultados retornados por busca
TOP_K_RESULTADOS = 10

# Parâmetros do Bag-of-Words
PARAMS_BOW = {
    "max_features": 5000,
    "min_df": 1,
}

# Parâmetros do TF-IDF
PARAMS_TFIDF = {
    "max_features": 5000,
    "min_df": 1,
    "norm": "l2",
}

# Parâmetros do Word2Vec (Gensim)
PARAMS_WORD2VEC = {
    "vector_size": 100,
    "window": 5,
    "min_count": 1,
    "epochs": 30,
    "seed": 42,
}

# Visualização t-SNE
HABILITAR_TSNE = True
PARAMS_TSNE = {
    "n_components": 2,
    "perplexity": 5,
    "n_iter": 1000,
    "random_state": 42,
}
```

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase2/input/artigos_anotacao_lg.parquet` | Parquet com anotações token-a-token gerado pela Fase 1 |

### Output

Todos os arquivos são salvos em `fase2/output/`:

| Arquivo | Descrição |
|---------|-----------|
| `tsne_plot.png` | Visualização 2D dos embeddings Word2Vec via t-SNE, com rótulos de documento |
| `fase2_pipeline.log` | Log detalhado do treinamento e buscas realizadas |

### Interface de Busca (CLI)

Após o treinamento, o terminal exibe um prompt interativo:

```
============================================================
  SISTEMA DE BUSCA TEXTUAL POR SIMILARIDADE
============================================================
  Metodos disponiveis: bow, tfidf, word2vec
  Comandos:
    <consulta>           - busca usando o primeiro metodo (bow)
    <metodo> <consulta>  - busca com metodo especifico
    sair                 - encerra
============================================================

> inteligencia artificial
Buscando com [bow]: 'inteligencia artificial'
--------------------------------------------------
  #1 [Score: 0.8721] Doc #2
      inteligência artificial sistemas especialistas...

  #2 [Score: 0.7340] Doc #5
      aprendizado de máquina redes neurais...
--------------------------------------------------
  2 resultados encontrados.

> tfidf redes neurais profundas
Buscando com [tfidf]: 'redes neurais profundas'
--------------------------------------------------
  #1 [Score: 0.9102] Doc #5
      deep learning redes convolucionais...
--------------------------------------------------

> word2vec processamento linguagem
Buscando com [word2vec]: 'processamento linguagem'
--------------------------------------------------
  ...

> sair
Encerrando busca...
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

A Fase 3 consome o artefato `.lpf2` gerado pela Fase 2 e aplica técnicas de modelagem de tópicos para descobrir temas latentes no corpus de artigos da Wikipedia:

| Método | Implementação | Descrição |
|--------|--------------|-----------|
| **LDA** | `LatentDirichletAllocation` (scikit-learn) | Modelagem probabilística — retorna proporção documento→tópico |
| **LSA** | `TruncatedSVD` (scikit-learn) | Modelagem por dimensionalidade — baseada em TF-IDF |
| **NMF** | `NMF` (scikit-learn) | Fatoração matricial — baseada em TF-IDF |

### Pré-requisito

A Fase 3 depende do artefato `.lpf2` gerado pela Fase 2:

```bash
# O artefato é gerado automaticamente pela Fase 2 em fase2/output/artifacts/
# Basta garantir que o arquivo existe antes de rodar a Fase 3
```

O artefato `.lpf2` (Language Processor Fase 2) contém:
- Matriz BoW (contagens) — usada pelo LDA
- Matriz TF-IDF — usada pelo LSA e NMF
- Vectorizers fitados — para transformar novos documentos
- Lista de documentos e títulos
- Parâmetros utilizados

### Como executar

```bash
cd fase3/src
python main.py
```

O pipeline executa automaticamente:
1. Carregamento do artefato `.lpf2` da Fase 2
2. Treinamento dos modelos de tópicos configurados
3. Geração de visualizações e métricas de avaliação

### Configuração

Edite `fase3/src/fase3_config.py` para controlar todos os aspectos da fase:

```python
# Métodos de modelagem a executar
METODOS_TOPICOS = ["lda", "lsa", "nmf"]

# Número de tópicos
NUM_TOPICOS = 10

# Parâmetros do LDA (usa matriz BoW)
PARAMS_LDA = {
    "n_components": 10,
    "max_iter": 20,
    "learning_method": "online",
    "random_state": 42,
}

# Parâmetros do LSA (usa matriz TF-IDF)
PARAMS_LSA = {
    "n_components": 10,
    "random_state": 42,
}

# Parâmetros do NMF (usa matriz TF-IDF)
PARAMS_NMF = {
    "n_components": 10,
    "random_state": 42,
}

# Número de palavras por tópico a exibir
TOP_N_PALAVRAS_TOPICO = 10
```

### Input

| Arquivo | Descrição |
|---------|-----------|
| `fase2/output/artifacts/fase2_artifact.lpf2` | Artefato da Fase 2 com matrizes BoW e TF-IDF |

### Output

Todos os arquivos são salvos em `fase3/output/`:

| Arquivo | Descrição |
|---------|-----------|
| `topicos_{metodo}.csv` | Tópicos encontrados com as top-N palavras e pesos |
| `distribuicao_documentos_{metodo}.png` | Distribuição de tópicos por documento |
| `distribuicao_topicos_{metodo}.png` | Distribuição geral de tópicos no corpus |
| `fase3_pipeline.log` | Log detalhado da execução |

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

# Todas as fases
pytest fase1/tests/ fase2/tests/ fase3/tests/ -v
```

**Cobertura de testes:**

| Fase | Módulos testados |
|------|-----------------|
| Fase 1 | `corpus_loader`, `preprocessing`, `pos_tagger`, pipeline completo |
| Fase 2 | `bow_vectorizer`, `tfidf_vectorizer`, `word2vec_vectorizer`, `cosine_search`, `search_interface`, pipeline completo, exportação de artefato |
| Fase 3 | `lda_model`, `lsa_model`, `nmf_model`, `topic_visualization`, pipeline completo |

---

## Estrutura do Projeto

```
python-NLP-pipeline-#1/
│
├── requirements.txt              # dependências globais
├── README.md
│
├── shared/
│   ├── __init__.py
│   ├── utils.py                  # utilitários compartilhados entre fases
│   └── artifacts.py              # ArtifactFase2 — serialização de artefatos entre fases
│
├── fase1/                        # Fase 1: Pré-processamento e POS Tagging
│   ├── requirements.txt
│   ├── input/
│   │   └── artigos_wikipedia.txt # corpus de entrada
│   ├── output/                   # gerado após execução
│   │   ├── 100-artigos_anotacao_lg.parquet
│   │   ├── wordcloud.png
│   │   ├── pos_distribution.png
│   │   ├── freq_comparison.png
│   │   ├── vocabulario_analise.json
│   │   └── nlp_pipeline.log
│   ├── src/
│   │   ├── main.py               # ponto de entrada
│   │   ├── fase1_config.py       # configurações e paths
│   │   ├── corpus_loader.py      # parsing do corpus
│   │   ├── preprocessing.py      # stopwords e filtragem
│   │   ├── pos_tagger.py         # POS tagging e NER com spaCy
│   │   ├── vocab_analysis.py     # análise e gráficos de vocabulário
│   │   ├── wordcloud_gen.py      # geração de nuvem de palavras
│   │   └── logger.py             # configuração de logging
│   └── tests/
│       ├── test_corpus_loader.py
│       ├── test_preprocessing.py
│       ├── test_pos_tagger.py
│       └── test_pipeline.py
│
├── fase2/                        # Fase 2: Embeddings e Busca por Similaridade
│   ├── requirements.txt
│   ├── input/
│   │   └── artigos_anotacao_lg.parquet  # Parquet da Fase 1
│   ├── output/                      # gerado após execução
│   │   ├── artifacts/
│   │   │   └── fase2_artifact.lpf2  # artefato para Fase 3
│   │   ├── tsne_plot.png
│   │   └── fase2_pipeline.log
│   ├── src/
│   │   ├── main.py                  # ponto de entrada
│   │   ├── fase2_config.py          # configurações e parâmetros
│   │   ├── embedding_pipeline.py    # orquestra vetorizadores e busca
│   │   ├── search_interface.py      # interface CLI interativa
│   │   ├── logger.py
│   │   ├── vectorizers/
│   │   │   ├── bow_vectorizer.py    # Bag-of-Words
│   │   │   ├── tfidf_vectorizer.py  # TF-IDF
│   │   │   └── word2vec_vectorizer.py # Word2Vec (Gensim)
│   │   ├── similarity/
│   │   │   └── cosine_search.py     # busca por similaridade de cosseno
│   │   └── visualization/
│   │       └── tsne_plot.py         # projeção t-SNE
│   └── tests/
│       ├── test_bow_vectorizer.py
│       ├── test_tfidf_vectorizer.py
│       ├── test_word2vec_vectorizer.py
│       ├── test_cosine_search.py
│       ├── test_search_interface.py
│       └── test_embedding_pipeline.py
│
├── fase3/                        # Fase 3: Modelagem de Tópicos
│   ├── requirements.txt
│   ├── input/
│   │   └── (artefato .lpf2 da Fase 2)
│   ├── output/                   # gerado após execução
│   ├── src/
│   │   ├── main.py               # ponto de entrada
│   │   ├── fase3_config.py       # configurações e parâmetros
│   │   ├── lda_model.py          # LDA (Latent Dirichlet Allocation)
│   │   ├── lsa_model.py          # LSA (Truncated SVD)
│   │   ├── nmf_model.py          # NMF (Non-negative Matrix Factorization)
│   │   ├── topic_visualization.py # gráficos de tópicos
│   │   └── logger.py
│   └── tests/
│       └── ...
│
├── fase4/                        # (futuro)
└── fase5/                        # (futuro)
```
