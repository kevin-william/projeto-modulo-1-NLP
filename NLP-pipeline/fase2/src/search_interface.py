import re

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def _obter_modelo_word2vec(pipeline):
    """Retorna o wv do Word2Vec se disponivel, ou None com mensagem de erro."""
    if "word2vec" not in pipeline.vetorizadores:
        print("[AVISO] Word2Vec nao disponivel. Inclua 'word2vec' em METODOS_EMBEDDING.")
        return None
    modelo = pipeline.vetorizadores["word2vec"].model
    if modelo is None:
        print("[AVISO] Modelo Word2Vec nao foi treinado (dados insuficientes).")
        return None
    return modelo.wv


def iniciar_interface_busca(pipeline):
    metodos = list(pipeline.motores_busca.keys())
    if not metodos:
        print("[ERRO] Nenhum metodo de busca disponivel.")
        return

    print()
    print("=" * 60)
    print("  SISTEMA DE BUSCA TEXTUAL POR SIMILARIDADE")
    print("=" * 60)
    print(f"  Metodos disponiveis: {', '.join(metodos)}")
    print("  Comandos:")
    print("    <consulta>               - busca usando o primeiro metodo")
    print("    <metodo> <consulta>      - busca com metodo especifico")
    print("    similar <palavra>        - palavras similares (Word2Vec)")
    print("    vetor <palavra>          - vetor da palavra (Word2Vec)")
    print("    analog <a> - <b> + <c>  - analogia: a - b + c (Word2Vec)")
    print("    sair                     - encerra")
    print("=" * 60)

    while True:
        try:
            entrada_usuario = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando...")
            break

        if not entrada_usuario:
            continue

        if entrada_usuario.lower() == "sair":
            print("Encerrando busca...")
            break

        # --- Comandos Word2Vec ---
        if entrada_usuario.lower().startswith("similar "):
            palavra = entrada_usuario[8:].strip()
            vetores_palavras = _obter_modelo_word2vec(pipeline)
            if vetores_palavras is not None:
                if palavra not in vetores_palavras:
                    print(f"[AVISO] Palavra '{palavra}' nao esta no vocabulario Word2Vec.")
                else:
                    quantidade_maxima = pipeline.configuracoes.get("TOP_K_RESULTADOS", 5)
                    resultados_similares = vetores_palavras.most_similar(palavra, topn=quantidade_maxima)
                    print(f"\nPalavras mais similares a '{palavra}':")
                    for palavra_similar, pontuacao in resultados_similares:
                        print(f"  {palavra_similar:<20} {pontuacao:.4f}")
            continue

        if entrada_usuario.lower().startswith("vetor "):
            palavra = entrada_usuario[6:].strip()
            vetores_palavras = _obter_modelo_word2vec(pipeline)
            if vetores_palavras is not None:
                if palavra not in vetores_palavras:
                    print(f"[AVISO] Palavra '{palavra}' nao esta no vocabulario Word2Vec.")
                else:
                    vetor_palavra = vetores_palavras[palavra]
                    print(f"\nVetor de '{palavra}' (dim={len(vetor_palavra)}):")
                    print("  " + " ".join(f"{componente:.4f}" for componente in vetor_palavra))
            continue

        correspondencia_analogia = re.match(r'^analog\s+(.+?)\s+-\s+(.+?)\s+\+\s+(.+)$', entrada_usuario, re.IGNORECASE)
        if correspondencia_analogia:
            a, b, c = correspondencia_analogia.group(1).strip(), correspondencia_analogia.group(2).strip(), correspondencia_analogia.group(3).strip()
            vetores_palavras = _obter_modelo_word2vec(pipeline)
            if vetores_palavras is not None:
                palavras_ausentes = [palavra for palavra in (a, b, c) if palavra not in vetores_palavras]
                if palavras_ausentes:
                    print(f"[AVISO] Palavras fora do vocabulario: {', '.join(palavras_ausentes)}")
                else:
                    quantidade_maxima = pipeline.configuracoes.get("TOP_K_RESULTADOS", 5)
                    resultados_similares = vetores_palavras.most_similar(positive=[a, c], negative=[b], topn=quantidade_maxima)
                    print(f"\nAnalogia: '{a}' - '{b}' + '{c}':")
                    for palavra_similar, pontuacao in resultados_similares:
                        print(f"  {palavra_similar:<20} {pontuacao:.4f}")
            continue

        partes = entrada_usuario.split(" ", 1)

        if len(partes) == 2 and partes[0] in metodos:
            metodo = partes[0]
            consulta = partes[1]
        elif partes[0] in metodos:
            metodo = partes[0]
            consulta = input("Digite sua consulta: ").strip()
            if not consulta:
                continue
        else:
            metodo = metodos[0]
            consulta = entrada_usuario

        print(f"\nBuscando com [{metodo}]: '{consulta}'")
        print("-" * 50)

        resultados_busca = pipeline.buscar_texto(metodo, consulta, top_k=pipeline.configuracoes.get("TOP_K_RESULTADOS", 10))

        if not resultados_busca:
            print("  Nenhum resultado encontrado.")
            continue

        for posicao, resultado in enumerate(resultados_busca, 1):
            print(f"  #{posicao} [Score: {resultado['score']:.4f}] Doc #{resultado['index']}")
            print(f"      {resultado['preview']}")
            print()

        print("-" * 50)
        print(f"  {len(resultados_busca)} resultados encontrados.")
