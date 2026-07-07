import os
import sys
import logging

from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.faiss import FAISS

from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.prompts.prompts_library import PROMPT_REGISTRY
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.utils.model_loader import ModelLoader


log = logging.getLogger(__name__)


class ConversationalRAG:

    def __init__(
        self,
        session_id: Optional[str],
        retriever=None
    ):
        try:
            self.session_id = session_id

            # Load the LLM
            self.llm = self._load_llm()

            self.contextualize_prompt = PROMPT_REGISTRY[
                "contextualize_question"
            ]
            self.qa_prompt = PROMPT_REGISTRY["context_qa"]

            # Retriever may be loaded later
            self.retriever = retriever

            # RAG chain starts empty
            self.chain = None

            # If retriever was already provided,
            # create the RAG chain immediately
            if self.retriever is not None:
                self._build_chain()

            log.info("ConversationalRAG initialized: %s", self.session_id)

        except Exception as e:
            log.error("Failed to initialize ConversationalRAG: %s", e)

            raise DocumentPortalException(
                "Initialization error in ConversationalRAG",
                sys
            )


    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: Optional[int] = None,
        index_name: str = "index",
        search_type: Optional[str] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):

        try:
            retriever_config = load_config().get("retriever", {})
            k = k if k is not None else retriever_config.get("top_k", 3)
            search_type = search_type or retriever_config.get(
                "search_type", "mmr"
            )
            fetch_k = fetch_k if fetch_k is not None else retriever_config.get(
                "fetch_k", 20
            )
            lambda_mult = (
                lambda_mult
                if lambda_mult is not None
                else retriever_config.get("lambda_mult", 0.5)
            )

            # Check whether FAISS folder exists
            if not os.path.isdir(index_path):
                raise FileNotFoundError(
                    f"FAISS index directory not found: {index_path}"
                )

            # Load embedding model
            embeddings = ModelLoader().load_embeddings()

            # Load saved FAISS vector database
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            # Prepare search settings
            if search_kwargs is None:

                search_kwargs = {
                    "k": k
                }

                if search_type == "mmr":
                    search_kwargs["fetch_k"] = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult

            # Convert vectorstore into retriever
            self.retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )

            # Build the RAG chain
            self._build_chain()

            log.info(
                "FAISS retriever loaded successfully: %s (%s)",
                index_path,
                self.session_id,
            )

            return self.retriever

        except Exception as e:
            log.error("Failed to load FAISS retriever: %s", e)

            raise DocumentPortalException(
                "Loading error in ConversationalRAG",
                sys
            )


    def invoke(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None
    ) -> str:

        try:
            # Make sure chain exists
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Load the retriever first.",
                    sys
                )

            # Use empty history if no history is provided
            if chat_history is None:
                chat_history = []

            # Input sent to the LCEL chain
            payload = {
                "input": user_input,
                "chat_history": chat_history
            }

            # Run complete RAG pipeline
            answer = self.chain.invoke(
                payload,
                config={
                    "run_name": "conversational_rag",
                    "tags": ["rag", "document-retrieval"],
                    "metadata": {"session_id": self.session_id},
                },
            )

            if not answer:
                return "No answer generated."

            answer = str(answer)

            log.info(
                "Answer generated successfully (%s): %s",
                self.session_id,
                answer[:150],
            )

            return answer

        except Exception as e:

            log.error("Failed to invoke RAG pipeline: %s", e)

            raise DocumentPortalException(
                "Invocation error in ConversationalRAG",
                sys
            )


    def _load_llm(self):

        try:
            llm = ModelLoader().load_llm()

            if not llm:
                raise ValueError("LLM could not be loaded")

            return llm

        except Exception as e:

            log.error("Failed to load LLM: %s", e)

            raise DocumentPortalException(
                "LLM loading error",
                sys
            )


    @staticmethod
    def _format_docs(docs):

        formatted_docs = []

        for doc in docs:

            content = getattr(
                doc,
                "page_content",
                str(doc)
            )

            formatted_docs.append(content)

        return "\n\n".join(formatted_docs)


    def _build_chain(self):

        try:
            if self.retriever is None:

                raise DocumentPortalException(
                    "Retriever is not available",
                    sys
                )

            # --------------------------------
            # STEP 1: Rewrite the question
            # --------------------------------

            question_rewriter = (

                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")
                }

                | self.contextualize_prompt

                | self.llm

                | StrOutputParser()
            )


            # --------------------------------
            # STEP 2: Retrieve documents
            # --------------------------------

            retrieve_docs = (

                question_rewriter

                | self.retriever

                | self._format_docs
            )


            # --------------------------------
            # STEP 3: Generate final answer
            # --------------------------------

            self.chain = (

                {
                    "context": retrieve_docs,

                    "input": itemgetter("input"),

                    "chat_history": itemgetter(
                        "chat_history"
                    )
                }

                | self.qa_prompt

                | self.llm

                | StrOutputParser()
            )


            log.info("RAG chain created successfully: %s", self.session_id)

        except Exception as e:

            log.error("Failed to build RAG chain: %s", e)

            raise DocumentPortalException(
                "Failed to build RAG chain",
                sys
            )
