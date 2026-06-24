"""Database engine and session helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from springgraph.config import get_settings

SessionLocal: sessionmaker[Session] | None = None


def get_session_factory() -> sessionmaker[Session]:
    """Return a lazily initialized session factory."""
    global SessionLocal
    if SessionLocal is None:
        engine = create_engine(get_settings().database_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
    return SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional SQLAlchemy session."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
