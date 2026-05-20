import logging
import sys


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Standardized application logger.

    Includes:
    - timestamp
    - log level
    - filename
    - function name
    - line number
    - message
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent double logging from root logger
    logger.propagate = False

    return logger
