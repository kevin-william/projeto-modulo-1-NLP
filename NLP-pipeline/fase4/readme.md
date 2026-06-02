# Fase 4 — Extração de Informações e Grafo de Conhecimento

Pipeline de extração de informação, normalização de entidades e construção de
grafo de conhecimento sobre o corpus de artigos da Wikipedia. Consome o parquet
anotado da Fase 1 e o artefato `.lpf2` da Fase 2.

## Funcionalidades

- **Extração de padrões com regex** — emails, URLs, datas e CPFs extraídos com
  expressões regulares (mesmas da Aula 07 do professor)
- **NER com spaCy** — extração de entidades nomeadas (PERSON, ORG, GPE, DATE)
  com contagem por tipo e visualização com displaCy
- **Fuzzy matching com Levenshtein** — normalização de entidades para agrupar
  variações ortográficas da mesma entidade
- **Extração de relações SVO** — triplas sujeito-verbo-objeto a partir de
  dependências sintáticas do spaCy
- **Grafo de conhecimento com NetworkX** — grafo com ≥20 nós, cálculo de
  centralidade (betweenness, degree, eigenvector) e pergunta analítica
- **Visualizações** — grafo com matplotlib, distribuição de centralidade,
  top entidades por tipo, renderizações HTML com displaCy

## Estrutura

```
fase4/
├── input/                              # parquet da Fase 1 e .lpf2 da Fase 2
├── output/
│   ├── plots/                          # gráficos e visualizações
│   ├── displacy/                       # renderizações HTML do displaCy
│   ├── grafo_edges.csv                 # arestas do grafo exportadas
│   ├── nos_grafo.csv                   # nós do grafo com atributos
│   ├── entidades_extraidas.csv         # entidades NER extraídas do corpus
│   ├── fuzzy_matches.csv               # resultado do Levenshtein
│   ├── relacoes_extraidas.csv          # triplas SVO extraídas
│   ├── resumo_metricas.json            # métricas consolidadas da execução
│   ├── relatorio_interpretativo.txt    # resposta à pergunta analítica
│   └── fase4_pipeline.log
├── src/
│   ├── fase4_config.py                 # constantes e caminhos
│   ├── logger.py                       # sistema de log
│   ├── main.py                         # executar_fase4_principal()
│   ├── extracao_padroes.py             # regex: emails, URLs, datas, CPFs
│   ├── fuzzy_matching.py               # Levenshtein para entidades
│   ├── ner_analysis.py                 # NER spaCy + displaCy
│   ├── relacoes.py                     # extração de relações SVO
│   ├── grafo_conhecimento.py           # NetworkX: construção, centralidade
│   └── visualizacao_grafo.py           # matplotlib/PyVis para grafo
├── tests/
│   ├── test_extracao_padroes.py
│   ├── test_fuzzy_matching.py
│   ├── test_ner_analysis.py
│   ├── test_relacoes.py
│   ├── test_grafo_conhecimento.py
│   └── test_pipeline.py
├── requirements.txt
├── readme.md
└── plan.md
```

## Pré-requisitos

### 1. Executar as Fases Anteriores

A Fase 4 depende de arquivos gerados pelas Fases 1 e 2:

```bash
# 1. Executar Fase 1 (gera o parquet com anotações linguísticas)
cd fase1/src
python main.py

# 2. Executar Fase 2 (gera o artefato .lpf2)
cd fase2/src
python main.py

# 3. Copiar arquivos de entrada para fase4/input/
copy fase1\output\*_lemmatizacao.parquet fase4\input\
copy fase2\output\artifacts\fase2_artifact.lpf2 fase4\input\
```

### 2. Dependências

```bash
pip install -r fase4/requirements.txt
```

### 3. Modelo spaCy

```bash
python -m spacy download pt_core_news_lg
```

## Configuração

As configurações estão em `src/fase4_config.py`:

```python
# Regex
REGEX_EMAILS    # padrão para extração de emails
REGEX_URLS      # padrão para extração de URLs
REGEX_DATAS     # padrão para extração de datas (dd/mm/aaaa)
REGEX_CPFs      # padrão para extração de CPFs

# Fuzzy matching
MAX_LEVENSHTEIN_DISTANCE = 2

# Grafo
MINIMO_FREQUENCIA_ENTIDADE = 3
NUMERO_MINIMO_NOGRAFOS = 20
TOP_CENTRALIDADE = 10
```

## Como Executar

```bash
# A partir da raiz do projeto
python -m fase4.src.main

# Ou diretamente
cd fase4/src
python main.py
```

## Testes

```bash
python -m pytest fase4/tests -q
```

## Saídas Esperadas

| Arquivo | Descrição |
|---|---|
| Arquivo | Descrição |
|---|---|
| `output/fase4_pipeline.log` | log detalhado da execução |
| `output/entidades_extraidas.csv` | entidades NER extraídas do corpus |
| `output/fuzzy_matches.csv` | resultado da normalização fuzzy das entidades |
| `output/relacoes_extraidas.csv` | triplas SVO extraídas |
| `output/grafo_edges.csv` | arestas do grafo de conhecimento |
| `output/nos_grafo.csv` | nós do grafo com tipo, frequência e centralidade |
| `output/resumo_metricas.json` | métricas consolidadas da execução |
| `output/relatorio_interpretativo.txt` | resposta textual à pergunta analítica |
| `output/plots/grafo_conhecimento.png` | visualização do grafo com matplotlib |
| `output/plots/centralidade_entidades.png` | distribuição de centralidade das entidades |
| `output/plots/top_entidades_por_tipo.png` | top entidades frequentes por tipo |
| `output/displacy/amostra_*.html` | renderizações HTML de NER com displaCy |

## Decisões Técnicas

| Componente | Tecnologia | Justificativa |
|---|---|---|
| Extração de padrões | `re` (regex do Python) | Mesmo uso do professor na Aula 07 |
| Fuzzy matching | `python-Levenshtein` | Biblioteca usada pelo professor na Aula 07 |
| NER | `spaCy` (`pt_core_news_lg`) | Modelo já utilizado nas Fases 1-3 |
| Visualização NER | `displacy` | Ferramenta nativa do spaCy, mostrada na Aula 08 |
| Extração de relações | `spaCy` (dependências sintáticas) | Abordagem do professor (Aula 07, linhas 684-700) |
| Grafo de conhecimento | `NetworkX` | Biblioteca padrão para grafos em Python |
| Visualização do grafo | `matplotlib` + `PyVis` (opcional) | Estático + interativo |

## Pergunta Analítica Respondida

O pipeline responde à seguinte pergunta:

> **"Quais entidades são mais centrais no corpus e por quê?"**

A resposta é baseada no cálculo de **betweenness centrality** do grafo de
conhecimento. Entidades com maior centralidade de intermediação são aquelas
que conectam diferentes clusters do corpus — ou seja, entidades que aparecem
em contextos diversos e conectam temas distintos.
