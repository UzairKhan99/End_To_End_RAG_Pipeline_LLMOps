from pathlib import Path

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader


def load_documents(paths):
    """Load supported document files into LangChain documents."""
    documents = []

    for path in paths:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        elif suffix == ".csv":
            loader = CSVLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix or path.name}")

        documents.extend(loader.load())

    return documents
