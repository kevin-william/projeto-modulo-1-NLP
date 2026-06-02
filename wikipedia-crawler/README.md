# Wiki Webscrap

Crawler recursivo de artigos da Wikipedia. A partir de uma URL inicial, coleta artigos via busca em largura (BFS), extrai o texto limpo de cada um e gera um arquivo consolidado.

O processamento e incremental: cada artigo e processado e escrito no arquivo de saida imediatamente apos o download, sem acumular HTML em memoria RAM.

## Requisitos

- Python 3.8+
- Dependencias em `requirements.txt`

## Instalacao

```bash
git clone <repositorio>
cd python-wiki-webscrap
pip install -r requirements.txt
```

## Configuracao

Edite o arquivo `config.json`:

| Chave | Tipo | Default | Descricao |
|---|---|---|---|
| `max_articles` | int | `1000` | Numero maximo de artigos a coletar |
| `delay` | float | `0.5` | Intervalo entre requisicoes (minimo 0.5s) |
| `links_per_page` | int | `10` | Quantidade de links extraidos por pagina |

Exemplo:

```json
{
    "max_articles": 500,
    "delay": 0.8,
    "links_per_page": 8
}
```

## Uso

```bash
# Crawl basico
python crawler.py "https://pt.wikipedia.org/wiki/Python"

# Com configuracao customizada e arquivo de log especifico
python crawler.py "https://pt.wikipedia.org/wiki/Python" --config meu_config.json --log meu_log.log

# Modo verbose (exibe mensagens de debug no console)
python crawler.py "https://pt.wikipedia.org/wiki/Python" -v
```

### Argumentos

| Argumento | Descricao |
|---|---|
| `url` | URL inicial do artigo (obrigatorio) |
| `--config` | Caminho para o arquivo de configuracao (default: `config.json`) |
| `--log` | Caminho para o arquivo de log (default: `crawler.log`) |
| `--verbose` / `-v` | Exibe mensagens de nivel DEBUG no console |

### Regras de coleta

- O crawler restringe a coleta ao **mesmo dominio** da URL inicial (ex: `https://pt.wikipedia.org`). Links para outras linguas (ex: `en.wikipedia.org`, `af.wikipedia.org`) sao ignorados.
- Usa busca em largura (BFS), respeita o delay entre requisicoes.
- Filtra links invalidos: namespaces com `:`, pagina principal (`Main_Page`).

### Processamento incremental

Cada artigo e processado assim que baixado: titulo extraido, texto limpo, bloco escrito em `artigos_wikipedia.txt`. O HTML bruto e descartado em seguida. Isso mantem o uso de RAM constante, independente do numero de artigos coletados.

## Logs

O arquivo de log (`crawler.log` por padrao) e o console exibem as mesmas informacoes:

- **INFO**: artigos obtidos, progresso da coleta, artigos escritos, finalizacao
- **WARNING**: problemas de configuracao, delay abaixo do minimo, falhas no crawl
- **ERROR**: falhas HTTP, erros de parse

Use `-v` para incluir mensagens de nivel DEBUG (diagnostico detalhado).

Exemplo de `crawler.log`:

```
2026-05-16 10:30:00 [INFO   ] Configuration loaded: {'max_articles': 500, ...}
2026-05-16 10:30:00 [INFO   ] Starting crawl from: https://pt.wikipedia.org/wiki/Python
2026-05-16 10:30:00 [INFO   ] [1/500] Fetching: https://pt.wikipedia.org/wiki/Python
2026-05-16 10:30:01 [INFO   ] Written article 1: Python (https://pt.wikipedia.org/wiki/Python)
2026-05-16 10:30:02 [INFO   ] [2/500] Fetching: https://pt.wikipedia.org/wiki/...
2026-05-16 10:30:05 [ERROR  ] HTTP error fetching https://pt.wikipedia.org/wiki/X: 404 ...
2026-05-16 10:32:00 [INFO   ] Crawl finished. 498 articles written, 2 errors.
2026-05-16 10:32:00 [WARNING] 2 URLs failed during crawl.
```

## Formato de saida

O arquivo `artigos_wikipedia.txt` e gerado com blocos delimitados:

```
===== ARTICLE START =====
Title: Python
URL: https://pt.wikipedia.org/wiki/Python
=========================
<texto limpo do artigo>
===== ARTICLE END =====
```

## Testes

```bash
pytest test_crawler.py -v
```

Os testes cobrem:

- Carregamento de configuracao (valida, ausente, invalida, delay abaixo do minimo)
- Extracao de links (caminhos wiki, exclusao de `:`, Main Page, URLs protocol-relative, limite, duplicatas, filtro de dominio diferente)
- Extracao de titulo (h1, fallback para title tag, fallback unknown)
- Limpeza de texto (remocao de scripts/styles, classes de referencia, selecao do container correto)
- Crawl (limite de artigos, tratamento de erros HTTP, duplicatas, delay, fila vazia, links por pagina)
- Arquivo de saida (blocos delimitados, titulo/URL presentes, escrita incremental)
