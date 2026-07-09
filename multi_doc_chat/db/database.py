import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from multi_doc_chat.model.db_models import Base


load_dotenv()


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "rag_project_db.sqlite"


def normalize_database_url(database_url: str) -> str:
    """Make common database URLs work with SQLAlchemy."""
    database_url = database_url.strip()

    # Neon sometimes gives postgres://, but SQLAlchemy needs postgresql+psycopg://
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)

    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    if database_url.startswith("mysql://"):
        return database_url.replace("mysql://", "mysql+pymysql://", 1)

    return database_url


def get_database_url() -> str:
    database_url = (
        os.getenv("NEON_DATABASE_URL")
        or os.getenv("NEON_DB_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("MYSQL_URL")
    )

    if database_url:
        return normalize_database_url(database_url)

    return f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"


def create_db_engine(database_url: str | None = None):
    database_url = normalize_database_url(database_url or get_database_url())

    if database_url.startswith("sqlite"):
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )

    return create_engine(database_url, pool_pre_ping=True)


engine = create_db_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db(db_engine=engine):
    Base.metadata.create_all(bind=db_engine)


def check_database_connection(db_engine=engine) -> bool:
    with db_engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True
