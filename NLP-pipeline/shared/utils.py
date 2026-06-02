import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def garantir_arquivo_entrada(caminho, nome_amigavel, instrucoes):
    """
    Verifica se um arquivo de entrada existe; caso contrario levanta
    FileNotFoundError com mensagem amigavel e multilinha.

    Parametros
    ----------
    caminho : str
        Caminho absoluto ou relativo ao arquivo esperado.
    nome_amigavel : str
        Descricao curta do arquivo (ex.: "corpus de artigos").
    instrucoes : list[str]
        Lista de passos sugeridos ao usuario para resolver o problema.

    Raises
    ------
    FileNotFoundError
        Sempre com mensagem formatada contendo caminho, nome amigavel e passos.
    """
    if os.path.exists(caminho):
        return caminho

    caminho_absoluto = os.path.abspath(caminho)
    linhas = [
        f"{nome_amigavel} nao encontrado.",
        f"Esperado em: {caminho_absoluto}",
        "",
        "Para corrigir:",
        *[f"  {i+1}. {passo}" for i, passo in enumerate(instrucoes)],
    ]
    raise FileNotFoundError("\n".join(linhas))
