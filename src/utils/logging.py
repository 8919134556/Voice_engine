"""
logging.py — a tiny wrapper around Python's standard `logging` module.

Library code (engine, conditioner, ...) only CREATES log messages:

    logger = get_logger(__name__)
    logger.info("Selected voice: %s", voice_id)

Nothing is printed until an application/script/notebook calls setup_logging()
once. That is standard Python practice: a library should never decide where
its logs go.

PRIVACY RULE: log voice_ids, shapes and counts — never raw embeddings, audio
samples or file contents.
"""

import logging
import sys

from src.config.settings import LOG_FORMAT, LOG_LEVEL

ROOT_LOGGER_NAME = "voice_engine"

# Silence by default (no "No handler found" noise) until setup_logging() is called.
logging.getLogger(ROOT_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """
    A logger inside the project's "voice_engine" family, e.g.
    get_logger("src.engine.voice_engine") -> "voice_engine.engine.voice_engine".
    """
    short = name.removeprefix("src.")
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{short}")


def setup_logging(level: str | int = LOG_LEVEL) -> logging.Logger:
    """
    Show the project's log messages on the console:  "INFO - Selected voice: voice_001".
    Safe to call more than once (it won't add duplicate handlers).
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level)
    if not any(getattr(h, "_voice_engine", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)  # stdout: stays in order with print() output
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._voice_engine = True  # mark it, so a second call doesn't add another
        root.addHandler(handler)
    return root
