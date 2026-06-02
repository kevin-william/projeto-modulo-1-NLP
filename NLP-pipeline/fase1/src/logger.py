import io
import logging
import os
import sys
from fase1_config import CAMINHO_LOG


def inicializar_sistema_log(nome_logger="nlp_pipeline"):
    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Garante que o diretorio de logs existe
    diretorio_log = os.path.dirname(CAMINHO_LOG)
    if diretorio_log and not os.path.exists(diretorio_log):
        os.makedirs(diretorio_log, exist_ok=True)

    file_handler = logging.FileHandler(CAMINHO_LOG, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
