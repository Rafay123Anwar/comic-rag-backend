import logging
import sys


def setup_logger(name: str = "comic_rag", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a standardized application logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


logger = setup_logger()
