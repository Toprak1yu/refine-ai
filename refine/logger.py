"""Central logging infrastructure for refine-ai."""

import logging
from pathlib import Path

_LOGGER_INITIALIZED = False


def setup_logger(log_file: str = "refine.log", level: int = logging.INFO) -> logging.Logger:
    """Configures the primary refine-ai logger.
    Logs detailed timestamped entries to file without cluttering terminal output.
    """
    global _LOGGER_INITIALIZED
    logger = logging.getLogger("refine")

    if not _LOGGER_INITIALIZED:
        logger.setLevel(level)
        logger.propagate = False

        # File handler for persistent execution logs
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setLevel(level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s:%(module)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            logger.addHandler(logging.NullHandler())

        _LOGGER_INITIALIZED = True

    return logger


def get_logger() -> logging.Logger:
    """Returns the central refine-ai logger."""
    return logging.getLogger("refine")
