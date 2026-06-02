# Fase 3 — Modelagem de Tópicos (LSA, LDA, NMF)

Pipeline de modelagem de tópicos sobre o corpus de artigos da Wikipedia.
Consome o artefato `.lpf2` exportado pela Fase 2 e executa análise exploratória
textual (EDA) seguida de três algoritmos de topic modeling com comparação
explícita de métricas.

## Funcionalidades

- **EDA Textual** — distribuição de comprimento dos documentos, tamanho do
  vocabulário, curva de Zipf, top-20 termos mais frequentes
- **LSA** — Latent Semantic Analysis via `TruncatedSVD` sobre matriz TF-IDF
- **LDA** — Latent Dirichlet Allocation via `sklearn` sobre matriz BoW
- **NMF** — Non-negative Matrix Factorization via `sklearn` sobre matriz TF-IDF
- **Comparação de modelos** — coerência top-N (co-ocorrência local) e
  perplexidade (LDA), exportadas em CSV
- **Visualizações** — barras horizontais com pesos reais por tópico, heatmaps
  documento×tópico e gráfico comparativo de coerência

## Estrutura

```
fase3/
├── input/                          # artefato .lpf2 copiado da fase2
├── output/
│   ├── plots/                      # gráficos EDA e visualizações de tópicos
│   ├── comparacao_modelos.csv      # tabela comparativa LSA × LDA × NMF
│   └── fase3_pipeline.log
├── src/
│   ├── fase3_config.py             # constantes e caminhos
│   ├── logger.py                   # sistema de log
│   ├── main.py                     # executar_fase3_principal()
│   ├── carregador_artefato.py      # carrega e valida ArtifactFase2 do .lpf2
│   ├── eda.py                      # análise exploratória textual
│   ├── avaliacao.py                # métricas de coerência e comparação
│   ├── visualizacao.py             # geração de gráficos
│   └── modelos_topicos/
│       ├── lsa_modelo.py
│       ├── lda_modelo.py
│       └── nmf_modelo.py
├── tests/
│   ├── test_carregador_artefato.py
│   ├── test_eda.py
│   ├── test_lsa.py
│   ├── test_lda.py
│   ├── test_nmf.py
│   ├── test_avaliacao.py
│   └── test_pipeline.py
└── requirements.txt
```

## Pré-requisitos

### 1. Executar a Fase 2 Primeiro

A Fase 3 depende do artefato `.lpf2` gerado pela Fase 2:

```bash
# 1. Executar Fase 2
cd fase2/src
python main.py

# 2. O artefato é gerado em output/artifacts/fase2_artifact.lpf2
#    Copiar para a entrada da Fase 3
copy output\artifacts\fase2_artifact.lpf2 fase3\input\fase2_artifact.lpf2
```

### 2. Dependências

```bash
pip install -r fase3/requirements.txt
```

## Configuração

As configurações estão em `src/fase3_config.py`:

```python
CAMINHO_ARTEFATO_FASE2  # caminho para o .lpf2 de entrada
NUM_TOPICOS             # número de tópicos (padrão: 10)
TOP_N_PALAVRAS          # palavras exibidas por tópico (padrão: 10)
PARAMS_LSA              # parâmetros do TruncatedSVD
PARAMS_LDA              # parâmetros do LatentDirichletAllocation
PARAMS_NMF              # parâmetros do NMF
```

## Como Executar

```bash
# A partir da raiz do projeto
python -m fase3.src.main

# Ou diretamente
cd fase3/src
python main.py
```

## Testes

```bash
python -m pytest fase3/tests -q
```

## Saídas Esperadas

| Arquivo | Descrição |
|---|---|
| `output/fase3_pipeline.log` | log detalhado da execução |
| `output/comparacao_modelos.csv` | tabela comparativa LSA × LDA × NMF |
| `output/plots/eda_comprimento.png` | histograma de comprimento dos documentos |
| `output/plots/eda_top_termos.png` | top-20 termos mais frequentes |
| `output/plots/eda_zipf.png` | curva de Zipf do corpus |
| `output/plots/eda_boxplot.png` | box plot de comprimento por documento |
| `output/plots/lsa_topicos.png` | top palavras por tópico com pesos — LSA |
| `output/plots/lda_topicos.png` | top palavras por tópico com pesos — LDA |
| `output/plots/nmf_topicos.png` | top palavras por tópico com pesos — NMF |
| `output/plots/lsa_heatmap.png` | heatmap documento×tópico — LSA |
| `output/plots/lda_heatmap.png` | heatmap documento×tópico — LDA |
| `output/plots/nmf_heatmap.png` | heatmap documento×tópico — NMF |
| `output/plots/comparacao_coerencia.png` | comparação de coerência entre modelos |

## Decisões Técnicas

| Algoritmo | Matriz de entrada | Métricas disponíveis |
|---|---|---|
| LSA | TF-IDF | coerência top-N |
| LDA | BoW (count) | coerência top-N, perplexidade |
| NMF | TF-IDF | coerência top-N |

- **LSA**: pesos diretos do SVD (sem valor absoluto) — seleciona palavras com
  maior carga positiva em cada componente.
- **LDA**: se o artefato não contiver `bow_matrix`, a matriz é reconstruída
  internamente via `CountVectorizer` com `PARAMS_BOW_RECONSTRUCAO`.
- **Coerência**: calculada localmente por co-ocorrência das top-N palavras de
  cada tópico nos documentos, sem dependência do Gensim.

