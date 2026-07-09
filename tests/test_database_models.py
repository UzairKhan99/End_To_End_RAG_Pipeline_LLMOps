import tempfile
import unittest
from pathlib import Path

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import sessionmaker

from multi_doc_chat.db.database import (
    create_db_engine,
    init_db,
    normalize_database_url,
)
from multi_doc_chat.model.db_models import (
    ChatSession,
    DocumentRecord,
    Message,
    User,
)


class DatabaseModelsTest(unittest.TestCase):
    def test_neon_postgres_url_uses_psycopg_driver(self):
        url = (
            "postgres://user:password@ep-test.us-east-1.aws.neon.tech/"
            "rag_project?sslmode=require"
        )

        normalized = normalize_database_url(url)

        self.assertEqual(
            normalized,
            (
                "postgresql+psycopg://user:password@"
                "ep-test.us-east-1.aws.neon.tech/rag_project?sslmode=require"
            ),
        )

    def test_models_create_tables_and_persist_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            engine = create_db_engine(f"sqlite:///{db_path.as_posix()}")
            init_db(engine)

            tables = set(inspect(engine).get_table_names())
            self.assertTrue(
                {
                    "users",
                    "chat_sessions",
                    "documents",
                    "messages",
                }.issubset(tables)
            )

            SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
            with SessionLocal() as session:
                user = User(
                    name="Test User",
                    email="test@example.com",
                    password_hash="hashed-password",
                )
                session.add(user)
                session.flush()

                chat_session = ChatSession(
                    session_id="session-test",
                    user_id=user.id,
                    title="Test Session",
                    faiss_index_path="faiss/session-test",
                )
                session.add(chat_session)
                session.flush()

                session.add(
                    DocumentRecord(
                        session_id=chat_session.session_id,
                        original_filename="sample.txt",
                        stored_path="data/sample.txt",
                        file_type="txt",
                    )
                )
                session.add(
                    Message(
                        session_id=chat_session.session_id,
                        role="user",
                        content="Hello database",
                    )
                )
                session.commit()

            with SessionLocal() as session:
                user_count = session.scalar(select(func.count(User.id)))
                session_count = session.scalar(select(func.count(ChatSession.id)))
                document_count = session.scalar(select(func.count(DocumentRecord.id)))
                message_count = session.scalar(select(func.count(Message.id)))

                self.assertEqual(user_count, 1)
                self.assertEqual(session_count, 1)
                self.assertEqual(document_count, 1)
                self.assertEqual(message_count, 1)

            engine.dispose()


if __name__ == "__main__":
    unittest.main()
