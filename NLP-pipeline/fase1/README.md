# Fase 1: Pré-processamento e POS Tagging

Pipeline de processamento de linguagem natural para artigos da Wikipedia em português utilizando spaCy.

## Funcionalidades

- **Carregamento de Corpus**: Leitura e parsing de artigos da Wikipedia com metadados (título, URL)
- **Filtro de Tamanho**: Remove artigos com menos de `MINIMO_PALAVRAS_ARTIGO` palavras antes do processamento; o log registra quantos foram removidos
- **Normalização Textual**: Etapa dedicada — lowercase, remoção de caracteres especiais e excesso de espaços (acentos preservados)
- **Tokenização por palavra**: Tokenização word-level com spaCy
- **Métodos de Processamento**: Lematização (spaCy), stemming (Snowball/Porter) ou sem normalização
- **Execução por método**: Pipeline rodada para cada método em `METODOS_PROCESSAMENTO_TOKENS`; artefatos separados por sufixo
- **POS Tagging**: Análise morfossintática completa com spaCy (`pt_core_news_lg`)
- **NER**: Reconhecimento de entidades nomeadas
- **Remoção de Stopwords**: Stopwords padrão do spaCy + customizadas em tempo de execução + extras via `STOPWORDS_EXTRAS`
- **WordCloud**: Geração de nuvens de palavras personalizáveis
- **Análise de Vocabulário**: Comparação antes/depois da remoção de stopwords, gráficos de frequência e distribuição de POS tags

## Estrutura de Arquivos

```
fase1/
├── input/
│   └── artigos_wikipedia.txt      # Dados de entrada
├── src/
│   ├── main.py                   # Script principal
│   ├── fase1_config.py           # Configurações
│   ├── logger.py                  # Sistema de logs
│   ├── corpus_loader.py          # Carregamento do corpus
│   ├── preprocessing.py          # Tokenização, stopwords, lemmatização
│   ├── pos_tagger.py             # POS tagging com spaCy
│   ├── vocab_analysis.py         # Análise de vocabulário
│   ├── wordcloud_gen.py          # Geração de wordcloud
│   └── vectorizers/              # (reservado para fase2)
├── output/                       # Resultados gerados
│   ├── 100-artigos_anotacao_lg.parquet   # DataFrame com anotações
│   ├── wordcloud.png            # Nuvem de palavras
│   ├── pos_distribution.png     # Distribuição de POS tags
│   ├── freq_comparison.png      # Comparativo de frequência
│   ├── vocabulario_analise.json # Métricas de vocabulário
│   └── nlp_pipeline.log         # Log completo
└── tests/                        # Testes unitários
    ├── test_corpus_loader.py
    ├── test_preprocessing.py
    ├── test_pos_tagger.py
    └── test_pipeline.py
```

## Requisitos

- Python 3.8+
- spaCy >= 3.7.0
- pandas >= 2.0.0
- wordcloud >= 1.9.0
- matplotlib >= 3.7.0
- pytest >= 7.0.0

## Instalação

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
```

## Como Executar

```bash
cd fase1/src
python main.py
```

## Saída do Pipeline

Ao executar `main.py`, o pipeline produz:

### Parâmetros do Parquet gerado
`output/1100-artigos_wikipedia-formatados-v001_{metodo}.parquet` — DataFrame contendo:
- `id_artigo`: ID único do artigo
- `id_token`: Posição do token no artigo
- `token`: Texto do token original
- `pos`: POS tag (NOUN, VERB, ADJ...)
- `tag`: Tag detalhada
- `lema`: Lema do token
- `processado`: Token após método configurado (`none`, `lemmatizacao`, `stemming`)
- `relacao_dependencia`: Relação de dependência
- `token_cabeca`: Token cabeça na árvore de dependência
- `entidade`: Entidade nomeada (se aplicável)
- `rotulo_entidade`: Tipo da entidade (PER, LOC, ORG, etc.)
- `titulo`: Título do artigo
- `url`: URL do artigo

### Visualizações
- `1100-artigos_wikipedia-formatados-v001_{metodo}.png` — Nuvem de palavras
- `pos_distribution_{metodo}.png` — Gráfico de barras com distribuição de POS tags
- `freq_comparison_{metodo}.png` — Comparativo de frequência antes/depois da remoção de stopwords

### Análise de Vocabulário
`output/1100-artigos_wikipedia-formatados-v001_{metodo}.json` — Métricas:
- `quantidade_vocabulario_bruto`: Total de palavras únicas (com stopwords)
- `quantidade_vocabulario_filtrado`: Total de palavras únicas (sem stopwords)
- `percentual_reducao_vocabulario`: Percentual de redução
- `top_20_bruto`: 20 palavras mais frequentes (com stopwords)
- `top_20_filtrado`: 20 palavras mais frequentes (sem stopwords)

## Configuração

As configurações estão em `src/fase1_config.py`:

```python
MODELO_SPACY = "pt_core_news_lg"              # Modelo spaCy
TAMANHO_LOTE = 5                               # Tamanho do lote de processamento
SEED_ALEATORIO = 42                            # Semente para reprodutibilidade

# Métodos de normalização do token (execução sequencial)
METODOS_PROCESSAMENTO_TOKENS = ['lemmatizacao', 'none']

# Filtro: artigos com menos de N palavras são removidos antes do processamento
MINIMO_PALAVRAS_ARTIGO = 40

# Stopwords adicionais além das padrão do spaCy
STOPWORDS_EXTRAS = []

# Customizacao da WordCloud (MVP)
LARGURA_NUVEM_PALAVRAS = 1200                  # resolucao horizontal da imagem
ALTURA_NUVEM_PALAVRAS = 600                    # resolucao vertical da imagem
MAXIMO_PALAVRAS_NUVEM = 200                    # limite de palavras na nuvem
PALETA_CORES_NUVEM = "viridis"                # mapa de cores (matplotlib)
COR_FUNDO_NUVEM = "white"                     # cor de fundo da imagem
TAMANHO_MINIMO_FONTE_NUVEM = 8                 # menor tamanho de fonte permitido
TAMANHO_MAXIMO_FONTE_NUVEM = 150               # maior tamanho de fonte permitido
SEMENTE_NUVEM_PALAVRAS = SEED_ALEATORIO        # layout reproduzivel
```

Os parametros da WordCloud sao lidos automaticamente por `wordcloud_gen.py`.
Quando voce quiser sobrescrever algo pontualmente via codigo, basta passar os argumentos na chamada de `gerar_nuvem_palavras(...)`; caso contrario, os valores acima sao usados como default.

> A pipeline é executada para **cada método** em `METODOS_PROCESSAMENTO_TOKENS`. Com `['lemmatizacao', 'none']`, são geradas 2 execuções com artefatos separados.

## Testes

```bash
cd fase1
pytest tests/ -v
```

## Logs

Os logs são salvos em:
- Arquivo: `output/nlp_pipeline.log` (detalhado, DEBUG)
- Terminal: informações principais (INFO)

Formato: `[TIMESTAMP] [LEVEL] message`

## Exemplo de Uso Programático

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from corpus_loader import carregar_artigos, filtrar_artigos_por_tamanho
from preprocessing import tokenizar_por_tipo, obter_stopwords
from pos_tagger import processar_lote_artigos

# Carregar e filtrar artigos
artigos = carregar_artigos()
artigos, removidos = filtrar_artigos_por_tamanho(artigos)  # remove artigos muito curtos

# Processar artigos com spaCy
dataframe = processar_lote_artigos(
    artigos,
    metodo_processamento="lemmatizacao",
)

# Inspecionar resultado
print(dataframe[["token", "lema", "processado", "pos"]].head())

# Salvar resultado
dataframe.to_parquet("output/resultado_lemmatizacao.parquet", index=False)
```

## Notas

- O modelo `pt_core_news_lg` é grande (~500MB). Na primeira execução, pode levar alguns minutos para baixar.
- O processamento em lote (`TAMANHO_LOTE`) otimiza o uso de memória.
- A saída desta fase (`100-artigos_anotacao_lg.parquet`) serve como entrada para a **Fase 2**.