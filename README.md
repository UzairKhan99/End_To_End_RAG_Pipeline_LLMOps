

<img src="assets/end_to_end_rag_pipeline.png" alt="End-to-End RAG Pipeline" width="100%">

# End-to-End RAG Pipeline

### A conversational document intelligence system built with FastAPI, LangChain and FAISS.

</div>

## About the Project

This project is an end-to-end Retrieval-Augmented Generation system that allows users to upload multiple documents and ask questions based on their content.

The system extracts document text, divides it into chunks, generates embeddings, stores them in a session-specific FAISS index and retrieves the most relevant information before generating an answer.

It also stores previous messages so users can ask context-aware follow-up questions.

## Main Features

- Multi-document upload
- PDF, DOCX, CSV, TXT and Markdown support
- Session-specific FAISS vector indexes
- Semantic search using embeddings
- Conversational chat history
- Context-aware follow-up questions
- FastAPI REST endpoints
- Structured LLM responses
- Database message storage
- LangSmith tracing and evaluation
- Modular and configurable architecture

## How It Works

```text
Upload Documents
        ↓
Extract Text
        ↓
Split Text into Chunks
        ↓
Generate Embeddings
        ↓
Store in FAISS
        ↓
User Asks a Question
        ↓
Retrieve Relevant Chunks
        ↓
Generate Context-Based Answer
        ↓
Save Conversation History
````

## Technology Stack

* Python
* FastAPI
* LangChain
* FAISS
* Hugging Face
* Qwen2.5-7B-Instruct
* Sentence Transformers
* SQLAlchemy
* LangSmith
* Pytest
* uv

## Installation

Clone the repository:

```bash
git clone https://github.com/UzairKhan99/End_To_End_RAG_Pipeline.git
cd End_To_End_RAG_Pipeline
```

Install the dependencies:

```bash
uv sync
```

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=RAG_PIPELINE
```

Run the application:

```bash
uv run uvicorn main:app --reload
```

Open the FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## API Workflow

1. Create a chat session
2. Upload one or multiple documents
3. Create the FAISS vector index
4. Ask questions about the documents
5. Ask follow-up questions using chat history
6. Retrieve stored conversation messages

## Use Cases

* Company knowledge assistants
* Research paper analysis
* Document question answering
* Policy and procedure chatbots
* Educational assistants
* Customer support knowledge bases
* Legal and financial document search

## Future Improvements

* Source citations in answers
* User authentication
* React frontend
* Docker deployment
* Hybrid search
* Reranking
* Streaming responses
* CI/CD pipeline

## Author

**Muhammad Uzair Uddin Khan**

GitHub: [https://github.com/UzairKhan99](https://github.com/UzairKhan99)

```
```
