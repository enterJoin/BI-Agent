"""Load prompt templates from files."""

from pathlib import Path


def load_prompt(name: str) -> str:
    """Load a prompt template by filename."""
    path = Path(__file__).resolve().parent / name
    return path.read_text(encoding="utf-8")
