"""Logging setup."""

import logging


def configure_logging(level: str) -> None:
    """Configure root logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s [%(name)s] %(message)s",
    )
