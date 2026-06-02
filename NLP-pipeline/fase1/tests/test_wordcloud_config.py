import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fase1_config import (
    LARGURA_NUVEM_PALAVRAS,
    ALTURA_NUVEM_PALAVRAS,
    MAXIMO_PALAVRAS_NUVEM,
    PALETA_CORES_NUVEM,
    COR_FUNDO_NUVEM,
    TAMANHO_MINIMO_FONTE_NUVEM,
    TAMANHO_MAXIMO_FONTE_NUVEM,
    SEMENTE_NUVEM_PALAVRAS,
)


def test_config_wordcloud_possui_valores_validos():
    assert isinstance(LARGURA_NUVEM_PALAVRAS, int) and LARGURA_NUVEM_PALAVRAS > 0
    assert isinstance(ALTURA_NUVEM_PALAVRAS, int) and ALTURA_NUVEM_PALAVRAS > 0
    assert isinstance(MAXIMO_PALAVRAS_NUVEM, int) and MAXIMO_PALAVRAS_NUVEM > 0
    assert isinstance(PALETA_CORES_NUVEM, str) and PALETA_CORES_NUVEM.strip()
    assert isinstance(COR_FUNDO_NUVEM, str) and COR_FUNDO_NUVEM.strip()
    assert isinstance(TAMANHO_MINIMO_FONTE_NUVEM, int) and TAMANHO_MINIMO_FONTE_NUVEM > 0
    assert isinstance(TAMANHO_MAXIMO_FONTE_NUVEM, int) and TAMANHO_MAXIMO_FONTE_NUVEM >= TAMANHO_MINIMO_FONTE_NUVEM
    assert isinstance(SEMENTE_NUVEM_PALAVRAS, int)
