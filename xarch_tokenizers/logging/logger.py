"""Defines logging utilities"""

import logging
from pathlib import Path

from xarch_tokenizers.config import Config


def setup_logger(config: "Config", name: str = "") -> logging.Logger:
    """Setup logger based on config."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name)

    # Add file handler
    file_handler = logging.FileHandler(Path(config.experiment_dir) / "logs.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    return logger
