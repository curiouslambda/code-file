import logging
from pathlib import Path

def setup_logger(log_path: Path | None = None, logger_name: str = "kref_rag") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path is not None:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
