from datetime import datetime
from pathlib import Path
import uuid

from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from multi_doc_chat.utils.document_ops import load_documents
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.model_loader import ModelLoader


def generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


class ChatIngestor:
    def __init__(
        self,
        temp_base="data",
        faiss_base="faiss_index",
        use_session_dirs=True,
        session_id=None,
    ):
        self.model_loader = ModelLoader()
        self.use_session = use_session_dirs
        self.session_id = session_id or generate_session_id()
        self.temp_base = Path(temp_base)
        self.faiss_base = Path(faiss_base)
        self.temp_base.mkdir(parents=True, exist_ok=True)
        self.faiss_base.mkdir(parents=True, exist_ok=True)

    def _resolve_dir(self, base: Path) -> Path:
        if self.use_session:
            directory = base / self.session_id
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        return base

    def _split(self, docs, chunk_size=1000, chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(docs)

    def build_retriever(self, uploaded_files):
        self.temp_dir = self._resolve_dir(self.temp_base)
        paths = save_uploaded_files(uploaded_files, self.temp_dir)
        docs = load_documents(paths)

        if not docs:
            raise ValueError("No valid documents were provided")

        return self._split(docs)

    # Keep the old misspelled method working for existing callers.
    build_retreiver = build_retriever


# Keep the old misspelled class name working for existing callers.
ChateIngestor = ChatIngestor


class FaissManager:
    def __init__(self, index_dir, model_loader=None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_loader = model_loader or ModelLoader()
        self.embedding_model = self.model_loader.load_embeddings()
        self.vector_store = None

    def _index_exists(self):
        return (
            (self.index_dir / "index.faiss").exists()
            and (self.index_dir / "index.pkl").exists()
        )

    def load_or_create(self, documents):
        if self._index_exists():
            self.vector_store = FAISS.load_local(
                str(self.index_dir),
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            documents = list(documents)
            if not documents:
                raise ValueError("Cannot create a FAISS index without documents")
            self.vector_store = FAISS.from_documents(
                documents,
                self.embedding_model,
            )
            self.vector_store.save_local(str(self.index_dir))

        return self.vector_store

    def add_new_documents(self, documents):
        documents = list(documents)
        if not documents:
            return self.vector_store

        if self.vector_store is None:
            return self.load_or_create(documents)

        self.vector_store.add_documents(documents)
        self.vector_store.save_local(str(self.index_dir))
        return self.vector_store


