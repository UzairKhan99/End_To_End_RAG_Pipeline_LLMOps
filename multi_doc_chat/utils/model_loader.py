import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)

from multi_doc_chat.utils.config_loader import load_config


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH, override=True)


def configure_langsmith():
    """Enable LangSmith tracing when its API key is configured."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "RAG_PIPELINE")
    return True


LANGSMITH_ENABLED = configure_langsmith()


def load_llm():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

    model_config = load_config()["model"]
    endpoint = HuggingFaceEndpoint(
        repo_id=model_config["name"],
        task="text-generation",
        max_new_tokens=model_config["max_tokens"],
        temperature=model_config["temperature"],
        do_sample=True,
        provider="auto",
    )
    return ChatHuggingFace(llm=endpoint)


def load_embedding_model():
    embedding_config = load_config()["embedding"]
    return HuggingFaceEmbeddings(
        model_name=embedding_config["model_name"]
    )


class ModelLoader:
    """Load the configured chat and embedding models."""

    def load_llm(self):
        return load_llm()

    def load_embeddings(self):
        return load_embedding_model()
