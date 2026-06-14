import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d"
            " | %(funcName)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[handler],
    )
