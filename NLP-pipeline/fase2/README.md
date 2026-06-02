# Fase 2: Token Embedding e Busca por Similaridade

Pipeline para geração de embeddings de texto e busca textual por similaridade.

## Funcionalidades

- **Bag-of-Words (BoW)**: Representação vetorial usando `CountVectorizer`
- **TF-IDF**: Representação vetorial com ponderação por frequência
- **Word2Vec**: Embeddings densos usando Gensim
- **Busca por Similaridade**: Interface interativa CMD para consultas textuais
- **t-SNE**: Visualização 2D dos embeddings
- **Fila de Métodos Configurável**: Escolha quais métodos usar e em qual ordem

## Estrutura de Arquivos

```
fase2/
├── input/
│   └── artigos_anotacao_lg.parquet   # Saída da Fase 1 (obrigatório)
├── src/
│   ├── main.py                   # Script principal
│   ├── fase2_config.py           # Configurações
│   ├── logger.py                 # Sistema de logs
│   ├── embedding_pipeline.py     # Orquestrador do pipeline
│   ├── search_interface.py      # Interface CMD de busca
│   ├── vectorizers/              # Implementações de vetorização
│   │   ├── bow_vectorizer.py
│   │   ├── tfidf_vectorizer.py
│   │   └── word2vec_vectorizer.py
│   ├── similarity/               # Busca por similaridade
│   │   └── cosine_search.py
│   └── visualization/             # Visualizações
│       └── tsne_plot.py
├── output/                       # Resultados gerados
│   ├── fase2_pipeline.log        # Log detalhado
│   └── tsne_plot.png            # Visualização t-SNE
└── tests/                        # Testes unitários
    ├── test_bow_vectorizer.py
    ├── test_tfidf_vectorizer.py
    ├── test_word2vec_vectorizer.py
    ├── test_cosine_search.py
    ├── test_embedding_pipeline.py
    └── test_search_interface.py
```

## Pré-requisitos

### 1. Executar a Fase 1 Primeiro

A Fase 2 depende do arquivo Parquet gerado pela Fase 1:

```bash
# 1. Executar Fase 1
cd fase1/src
python main.py

# 2. Copiar o Parquet para a Fase 2 (se não foi feito automaticamente)
copy fase1\output\100-artigos_anotacao_lg.parquet fase2\input\artigos_anotacao_lg.parquet
```

### Dependências

- Python 3.8+
- scikit-learn >= 1.3.0
- gensim >= 4.3.0
- numpy >= 1.24.0

```bash
pip install scikit-learn gensim
```

## Configuração

As configurações estão em `src/fase2_config.py`:

```python
# Caminhos e diretórios
DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_PARQUET_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "100-artigos_anotacao_lg_lemmatizacao_palavra.parquet")
DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")
CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "100-artigos_anotacao_lg_lemmatizacao_palavra.log")

# Métodos a serem treinados (ordem importa)
METODOS_EMBEDDING = ["bow", "tfidf", "word2vec"]
TOP_K_RESULTADOS = 5

# Parâmetros de cada método
PARAMS_BOW = {"max_features": 5000, "min_df": 1}
PARAMS_TFIDF = {"max_features": 5000, "min_df": 1, "norm": "l2"}
PARAMS_WORD2VEC = {
    "vector_size": 100,
    "window": 5,
    "min_count": 1,
    "epochs": 30,
    "seed": 42
}

# Visualização t-SNE
HABILITAR_TSNE = True
PARAMS_TSNE = {"n_components": 2, "perplexity": 30, "n_iter": 2000, "random_state": 42}
PARAMS_PLOT_TSNE = {
    "figsize": (20, 16),
    "dpi": 600,
    "marker_size": 50,
    "annotate_fontsize": 7,
}

CAMINHO_SAIDA_TSNE = os.path.join(DIRETORIO_SAIDA, "tsne_plot-100-artigos_anotacao_lg_lemmatizacao_palavra.png")
```

### Referência completa das variáveis de configuração

| Variável | Tipo | Valor atual | Descrição |
|----------|------|-------------|-----------|
| `DIRETORIO_BASE` | `str` | diretório `fase2/` | Raiz da fase2 para resolver caminhos relativos. |
| `CAMINHO_PARQUET_ENTRADA` | `str` | `fase2/input/100-artigos_anotacao_lg_lemmatizacao_palavra.parquet` | Parquet de entrada usado para gerar os embeddings. |
| `DIRETORIO_SAIDA` | `str` | `fase2/output/` | Pasta onde logs e artefatos são salvos. |
| `CAMINHO_LOG` | `str` | `fase2/output/100-artigos_anotacao_lg_lemmatizacao_palavra.log` | Arquivo de log da execução da fase2. |
| `METODOS_EMBEDDING` | `list[str]` | `['bow', 'tfidf', 'word2vec']` | Ordem dos métodos treinados e disponíveis na busca. |
| `TOP_K_RESULTADOS` | `int` | `5` | Quantidade padrão de resultados por consulta. |
| `PARAMS_BOW` | `dict` | `{'max_features': 5000, 'min_df': 1}` | Hiperparâmetros do Bag-of-Words (`CountVectorizer`). |
| `PARAMS_TFIDF` | `dict` | `{'max_features': 5000, 'min_df': 1, 'norm': 'l2'}` | Hiperparâmetros do TF-IDF (`TfidfVectorizer`). |
| `PARAMS_WORD2VEC` | `dict` | `{'vector_size': 100, 'window': 5, 'min_count': 1, 'epochs': 30, 'seed': 42}` | Hiperparâmetros de treino do Word2Vec (Gensim). |
| `HABILITAR_TSNE` | `bool` | `True` | Liga/desliga a geração da visualização t-SNE ao final do pipeline. |
| `PARAMS_TSNE` | `dict` | `{'n_components': 2, 'perplexity': 30, 'n_iter': 2000, 'random_state': 42}` | Parâmetros do algoritmo t-SNE para reduzir embeddings para 2D. |
| `PARAMS_PLOT_TSNE` | `dict` | `{'figsize': (20, 16), 'dpi': 600, 'marker_size': 50, 'annotate_fontsize': 7}` | Parâmetros visuais do gráfico t-SNE salvo em arquivo. |
| `CAMINHO_SAIDA_TSNE` | `str` | `fase2/output/tsne_plot-100-artigos_anotacao_lg_lemmatizacao_palavra.png` | Caminho final da imagem t-SNE gerada. |

### Significado dos parâmetros internos dos dicionários

`PARAMS_BOW`
- `max_features`: limite máximo de termos no vocabulário final do `CountVectorizer`. Reduzir esse valor diminui uso de memória e tempo de treino, mas pode perder termos relevantes; aumentar melhora cobertura semântica do corpus, com custo computacional maior.
- `min_df`: frequência mínima de documentos para manter um termo no vocabulário. Valores maiores removem ruído (termos muito raros), mas podem descartar palavras específicas importantes para busca em nichos.

`PARAMS_TFIDF`
- `max_features`: limite máximo de termos no vocabulário do `TfidfVectorizer`. Em corpora grandes, esse controle evita explosão dimensional; em corpora pequenos, valores muito baixos podem empobrecer a representação.
- `min_df`: frequência mínima de documentos para manter um termo. Funciona como filtro de ruído lexical; quanto maior o valor, mais agressiva é a remoção de termos raros.
- `norm`: normalização dos vetores (`l1` ou `l2`). Com `l2` (padrão), a similaridade de cosseno tende a ficar mais estável entre documentos de tamanhos diferentes.

`PARAMS_WORD2VEC`
- `vector_size`: dimensão de cada embedding de palavra. Dimensões maiores capturam mais nuances semânticas, porém exigem mais memória e podem overfit em corpus pequeno.
- `window`: tamanho da janela de contexto (palavras antes/depois). Janelas pequenas favorecem relações sintáticas locais; janelas maiores capturam contexto semântico mais amplo.
- `min_count`: frequência mínima para incluir uma palavra no vocabulário. Aumentar remove palavras raras e ruído; reduzir preserva termos pouco frequentes, úteis em domínios específicos.
- `epochs`: número de passagens completas no corpus. Mais épocas normalmente melhoram convergência, mas aumentam tempo de treino e risco de ajuste excessivo em datasets pequenos.
- `seed`: semente aleatória para reprodutibilidade do treinamento e comparabilidade entre execuções.

`PARAMS_TSNE`
- `n_components`: dimensão da projeção final (tipicamente `2` para gráfico plano). Valores maiores podem ser úteis para análise exploratória, mas dificultam visualização direta.
- `perplexity`: controla o número efetivo de vizinhos considerados. Valores baixos destacam microclusters; valores altos tendem a preservar estrutura global. Regra prática: manter abaixo do número de amostras e testar variações.
- `n_iter`: número de iterações de otimização. Iterações insuficientes podem gerar projeções instáveis; valores mais altos aumentam qualidade visual, com maior custo de tempo.
- `random_state`: semente para tornar a projeção reproduzível entre execuções e facilitar comparação de experimentos.

`PARAMS_PLOT_TSNE`
- `figsize`: tamanho da figura em polegadas (`largura`, `altura`). Figuras maiores melhoram legibilidade de rótulos quando há muitos documentos.
- `dpi`: resolução de saída da imagem. DPI alto melhora nitidez para relatório/apresentação, mas aumenta tamanho de arquivo e tempo de renderização.
- `marker_size`: tamanho dos pontos no scatter plot. Ajuste fino importante para evitar sobreposição (muito grande) ou baixa visibilidade (muito pequeno).
- `annotate_fontsize`: tamanho da fonte das anotações dos pontos. Deve equilibrar legibilidade e poluição visual; em muitos pontos, fontes menores tendem a funcionar melhor.

### Exemplos de Configuração

**Usar apenas TF-IDF:**
```python
METODOS_EMBEDDING = ["tfidf"]
```

**Usar Word2Vec + TF-IDF:**
```python
METODOS_EMBEDDING = ["word2vec", "tfidf"]
```

**Aumentar dimensões do Word2Vec:**
```python
PARAMS_WORD2VEC = {"vector_size": 200, "window": 5, "min_count": 2, "epochs": 50}
```

### Parâmetros do t-SNE

| Parâmetro | Valor Padrão | Efeito Prático | Faixa sugerida |
|-----------|-------------|----------------|----------------|
| `perplexity` | 30 | Controla o balanço local vs global da projeção. Baixo destaca microgrupos; alto tende a mostrar organização mais ampla. | 5 a 50 (sempre menor que o número de amostras) |
| `n_iter` | 2000 | Define o esforço de otimização. Poucas iterações podem gerar mapa instável; mais iterações melhoram convergência com maior custo de tempo. | 1000 a 4000 |
| `n_components` | 2 | Número de dimensões da saída. Em 2D é ideal para visualização; em 3D serve para exploração adicional. | 2 (padrão) ou 3 |
| `random_state` | 42 | Mantém a projeção reproduzível entre execuções para comparação de experimentos. | inteiro fixo |

Sinais de ajuste:
- Se o gráfico ficar "embolado", aumente `n_iter` e teste `perplexity` maior.
- Se perder separação de grupos pequenos, reduza `perplexity`.
- Para comparação entre versões do pipeline, mantenha `random_state` fixo.

### Parâmetros do Plot (visualização)

| Parâmetro | Valor Padrão | Efeito Prático | Faixa sugerida |
|-----------|-------------|----------------|----------------|
| `figsize` | (20, 16) | Aumenta a área útil do gráfico e reduz sobreposição de rótulos. | (12, 9) a (24, 18) |
| `dpi` | 600 | Melhora nitidez para relatório e zoom, com arquivos maiores. | 150 a 600 |
| `marker_size` | 50 | Controla visibilidade dos pontos; muito alto gera sobreposição, muito baixo prejudica leitura. | 20 a 80 |
| `annotate_fontsize` | 7 | Define legibilidade dos rótulos no gráfico. | 6 a 10 |

Sinais de ajuste:
- Se houver muita sobreposição de texto, aumente `figsize` e reduza `annotate_fontsize`.
- Se os pontos estiverem pouco visíveis em tela/projeção, aumente `marker_size`.
- Para exportação acadêmica, priorize `dpi` alto (300+); para uso rápido local, `dpi` menor acelera geração.

### Valores Recomendados por Cenário

| Cenário | perplexity | n_iter | figsize |
|---------|------------|--------|---------|
| Poucos dados (<100 docs) | 5-10 | 1000 | (10, 8) |
| Dados médios (100-1000 docs) | 30-50 | 2000 | (14, 10) |
| Muitos dados (>1000 docs) | 50-100 | 3000 | (16, 12) |

### Exemplo: Visualização condensada

Se a visualização estiver com pontos muito aglomerados, aumente `perplexity` e `figsize`:
```python
PARAMS_TSNE = {"perplexity": 50, "n_iter": 3000, "random_state": 42}
PARAMS_PLOT_TSNE = {"figsize": (20, 14), "dpi": 200, "marker_size": 40}
```

## Como Executar

```bash
cd fase2/src
python main.py
```

### O que acontece:

1. **Carregamento**: O pipeline lê o Parquet da Fase 1 (`input/artigos_anotacao_lg.parquet`)
2. **Pré-processamento**: Constrói documentos a partir dos lemas (agrupados por id_artigo), remove pontuação
3. **Treinamento**: Treina cada método de embedding configurado
4. **t-SNE**: Gera visualização 2D dos embeddings (se habilitado)
5. **Interface de Busca**: Abre prompt interativo para consultas

## Interface de Busca CMD

Ao final do treinamento, o sistema entra em modo interativo:

```
============================================================
  SISTEMA DE BUSCA TEXTUAL POR SIMILARIDADE
============================================================
  Metodos disponiveis: bow, tfidf, word2vec
  Comandos:
    <consulta>           - busca usando o primeiro metodo
    <metodo> <consulta>  - busca com metodo especifico
    sair                 - encerra
============================================================
```

### Exemplos de Consulta

**Busca simples (usa primeiro método):**
```
> python processamento natural
```

**Busca com método específico:**
```
> bow aprendizado de maquina
> tfidf linguagem natural
> word2vec redes neurais
```

**Buscar com mais resultados:**
```
> 20 processamentos de linguagem natural
```

### Saída dos Resultados

```
Buscando com [tfidf]: 'processamento natural'
--------------------------------------------------
  #1 [Score: 0.8723] Doc #1
      processamento linguagem natural dados...

  #2 [Score: 0.6541] Doc #3
      linguagem natural processamento texto...

--------------------------------------------------
  2 resultados encontrados.
```

Cada resultado contém:
- `Score`: Similaridade de cosseno (0 a 1)
- `Doc #N`: ID do documento no corpus
- Preview: Primeiros 200 caracteres do documento

Para sair, digite `sair`.

## Uso Programático

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from embedding_pipeline import PipelineEmbeddings
from fase2_config import (
    CAMINHO_PARQUET_ENTRADA,
    METODOS_EMBEDDING,
    PARAMS_BOW,
    PARAMS_TFIDF,
    PARAMS_WORD2VEC,
    TOP_K_RESULTADOS,
    HABILITAR_TSNE,
    PARAMS_TSNE,
    PARAMS_PLOT_TSNE,
)

# Configuração
config = {
    "METODOS_EMBEDDING": METODOS_EMBEDDING,
    "PARAMS_BOW": PARAMS_BOW,
    "PARAMS_TFIDF": PARAMS_TFIDF,
    "PARAMS_WORD2VEC": PARAMS_WORD2VEC,
    "TOP_K_RESULTADOS": TOP_K_RESULTADOS,
    "HABILITAR_TSNE": HABILITAR_TSNE,
    "PARAMS_TSNE": PARAMS_TSNE,
    "PARAMS_PLOT_TSNE": PARAMS_PLOT_TSNE,
}

# Criar e executar pipeline
pipeline = PipelineEmbeddings(config, CAMINHO_PARQUET_ENTRADA)
search_engines = pipeline.executar()

# Consulta via código
results = pipeline.buscar_texto("bow", "inteligencia artificial", top_k=5)

for r in results:
    print(f"Score: {r['score']:.4f} | Doc #{r['index']}")
```

## Testes

```bash
cd fase2
pytest tests/ -v
```

### Cobertura

- `test_bow_vectorizer.py`: Fit, transform, edge cases
- `test_tfidf_vectorizer.py`: Fit, transform, valores positivos
- `test_word2vec_vectorizer.py`: Fit, sentence embeddings, palavras desconhecidas
- `test_cosine_search.py`: Busca, ordenação por score
- `test_embedding_pipeline.py`: Pipeline completo, múltiplos métodos
- `test_search_interface.py`: Imports e estrutura

## Diferenças entre Métodos

| Método | Tipo | Velocidade | Uso de Memória | Melhor para |
|--------|------|------------|----------------|-------------|
| **BoW** | Esparso | Rápido | Médio | Baseline, debug |
| **TF-IDF** | Esparso | Rápido | Médio | Busca textual, relevância |
| **Word2Vec** | Denso | Moderado | Baixo | Semântica, similaridade |

### Observações

- **BoW** e **TF-IDF** são métodos esparsos (matrizes com muitos zeros)
- **Word2Vec** gera vetores densos (representação contínua)
- A similaridade de cosseno funciona bem para todos
- t-SNE usa o primeiro método disponível para visualização

## Logs

Os logs são salvos em:
- Arquivo: `output/fase2_pipeline.log` (detalhado, DEBUG)
- Terminal: informações principais (INFO)

Formato: `[TIMESTAMP] [LEVEL] [MODULE] message`

## Fluxo de Dados

```
Fase 1 Output                    Fase 2 Input                Fase 2 Output
┌─────────────────┐             ┌─────────────────┐          ┌─────────────────┐
│ artigos_anotacao│ ─────────> │ artigos_anotacao│          │ tsne_plot.png   │
│ _lg.parquet     │            │ _lg.parquet     │          │ fase2.log       │
│ (tokens+POS)    │            │ (tokens+POS)    │          │                 │
└─────────────────┘            └─────────────────┘          └─────────────────┘
                                       │
                                       v
                              ┌─────────────────┐
                              │ EmbeddingPipeline
                              │ - tokenização   │
                              │ - vetorização   │
                              │ - busca         │
                              └─────────────────┘
```

## Notas

- O CSV de entrada deve conter as colunas: `artigo_id`, `lemma`, `pos`, `title`
- A busca por similaridade usa similaridade de cosseno entre vetores
- O Word2Vec precisa de ao menos uma ocorrência de cada palavra (`min_count=1` por padrão)
- Para textos pequenos, o TF-IDF geralmente oferece melhores resultados
- Para capturar relações semânticas, o Word2Vec é mais adequado