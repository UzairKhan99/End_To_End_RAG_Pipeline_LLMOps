import logging
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"
DEFAULT_DB_PATH = ROOT_DIR / "data" / "app.db"

logger = logging.getLogger(__name__)


def get_database_path() -> Path:
    load_dotenv(ENV_PATH, override=True)

    db_path = os.getenv("DATABASE_PATH") or os.getenv("SQLITE_DB_PATH")
    if not db_path:
        return DEFAULT_DB_PATH

    return Path(db_path).expanduser()


def get_database_connection() -> sqlite3.Connection:
    db_path = get_database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to database: %s", db_path)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    logger.info("Database connected successfully")
    return connection


def close_database_connection(connection: sqlite3.Connection) -> None:
    if connection:
        connection.close()
        logger.info("Database connection closed")
