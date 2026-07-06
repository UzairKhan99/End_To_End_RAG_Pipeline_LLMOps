import os

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings
)

from multi_doc_chat.utils.config_loader import load_config


load_dotenv(override=True)


def load_llm():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN not found in .env file"
        )

    model_config = load_config()["model"]

    llm = HuggingFaceEndpoint(
        repo_id=model_config["name"],
        task="text-generation",
        max_new_tokens=model_config["max_tokens"],
        temperature=model_config["temperature"],
        do_sample=True,
        provider="auto"
    )

    model = ChatHuggingFace(llm=llm)

    return model


def load_embedding_model():
    embedding_config = load_config()["embedding"]

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_config["model_name"]
    )

    return embedding_model
