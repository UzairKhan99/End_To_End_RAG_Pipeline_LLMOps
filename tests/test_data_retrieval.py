import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import multi_doc_chat.src.document_chat.retreival as retrieval


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
HAS_HUGGINGFACE_TOKEN = bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
HAS_LANGSMITH_TOKEN = bool(os.getenv("LANGSMITH_API_KEY"))

DEFAULT_CONTEXT = (
    "Agentic AI systems can plan tasks, use external tools, and take actions "
    "autonomously to achieve a goal. A basic chatbot mainly responds to each "
    "user message without independently planning or taking actions."
)
DEFAULT_USER_INPUT = "How is agentic AI different from a basic chatbot?"


class FakeModelLoader:
    def load_llm(self):
        def respond(prompt):
            prompt_text = prompt.to_string()
            if prompt_text.rstrip().endswith("Standalone question:"):
                return DEFAULT_USER_INPUT
            return (
                "Agentic AI can plan, use tools, and take actions, while a "
                "basic chatbot mainly responds to messages."
            )

        return RunnableLambda(respond)

    def load_embeddings(self):
        return object()


class DataRetrievalTest(unittest.TestCase):
    def test_faiss_uses_mmr_retrieval(self):
        vectorstore = MagicMock()
        vectorstore.as_retriever.return_value = RunnableLambda(lambda query: [])

        with (
            patch.object(retrieval, "ModelLoader", FakeModelLoader),
            patch.object(retrieval.FAISS, "load_local", return_value=vectorstore),
        ):
            rag = retrieval.ConversationalRAG(session_id="mmr-test")
            rag.load_retriever_from_faiss("tests")

        vectorstore.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
        )

    def test_rag_chain_with_fake_llm(self):
        retrieved_questions = []

        def retrieve(question):
            retrieved_questions.append(question)
            return [Document(page_content=DEFAULT_CONTEXT)]

        retriever = RunnableLambda(retrieve)

        with patch.object(retrieval, "ModelLoader", FakeModelLoader):
            rag = retrieval.ConversationalRAG(
                session_id="test-session",
                retriever=retriever,
            )
            answer = rag.invoke(DEFAULT_USER_INPUT)

        print(f"\n[Retrieval] Query: {retrieved_questions[0]}")
        print(f"[Retrieval] Context: {DEFAULT_CONTEXT}")
        print(f"[Retrieval] Answer: {answer}")

        self.assertIn("plan", answer.lower())
        self.assertIn("chatbot", answer.lower())
        self.assertCountEqual(
            rag.contextualize_prompt.input_variables,
            ["chat_history", "input"],
        )
        self.assertCountEqual(
            rag.qa_prompt.input_variables,
            ["context", "chat_history", "input"],
        )

    @unittest.skipUnless(
        HAS_HUGGINGFACE_TOKEN,
        "HUGGINGFACEHUB_API_TOKEN is not available in .env",
    )
    def test_live_llm_response(self):
        context = os.getenv(
            "TEST_DOCUMENT_CONTEXT",
            DEFAULT_CONTEXT,
        )
        user_input = os.getenv("TEST_USER_INPUT", DEFAULT_USER_INPUT)
        retrieved_questions = []

        def retrieve(question):
            retrieved_questions.append(question)
            return [Document(page_content=context)]

        rag = retrieval.ConversationalRAG(
            session_id="live-test-session",
            retriever=RunnableLambda(retrieve),
        )
        answer = rag.invoke(user_input)

        print(f"\n[Live LLM] User input: {user_input}")
        print(f"\n[Live LLM] Rewritten query: {retrieved_questions[0]}")
        print(f"[Live LLM] Retrieved context: {context}")
        print(f"[Live LLM] Response: {answer}")
        if HAS_LANGSMITH_TOKEN:
            from langchain_core.tracers.langchain import wait_for_all_tracers

            wait_for_all_tracers()
            project = os.getenv("LANGSMITH_PROJECT", "RAG_PIPELINE")
            print(f"[LangSmith] Trace sent to project: {project}")
        else:
            print("[LangSmith] Trace disabled: LANGSMITH_API_KEY is not saved")

        self.assertTrue(answer.strip())
        self.assertTrue(retrieved_questions)


if __name__ == "__main__":
    unittest.main()
