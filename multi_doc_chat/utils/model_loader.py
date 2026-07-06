from multi_doc_chat.utils.mdeol_loader import (
    load_embedding_model,
    load_llm,
)


class ModelLoader:
    """Small wrapper used by the ingestion pipeline."""

    def load_llm(self):
        return load_llm()

    def load_embeddings(self):
        return load_embedding_model()
