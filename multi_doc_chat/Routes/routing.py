import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from multi_doc_chat.db.database import check_database_connection, get_session
from multi_doc_chat.model.db_models import ChatSession, DocumentRecord, Message, User
from multi_doc_chat.src.document_chat.retreival import ConversationalRAG
from multi_doc_chat.src.document_ingestion.data_ingestion import (
    ChatIngestor,
    FaissManager,
)


router = APIRouter()


class SessionCreate(BaseModel):
    user_id: int | None = None
    title: str = "New Chat"


class ChatRequest(BaseModel):
    question: str


def get_default_user(db: Session):
    """Use this simple user when you do not pass a user_id."""
    user = db.query(User).filter(User.email == "default@example.com").first()

    if user:
        return user

    user = User(
        name="Default User",
        email="default@example.com",
        password_hash="not-used",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_session_or_error(db: Session, session_id: str):
    """Find one session or return 404."""
    chat_session = (
        db.query(ChatSession)
        .filter(ChatSession.session_id == session_id)
        .first()
    )

    if not chat_session:
        raise HTTPException(status_code=404, detail="Session not found")

    return chat_session


def get_chat_history(db: Session, session_id: str):
    """Convert saved database messages into LangChain messages."""
    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at)
        .all()
    )

    chat_history = []

    for message in messages:
        if message.role == "user":
            chat_history.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            chat_history.append(AIMessage(content=message.content))

    return chat_history


@router.get("/health", summary="Check API and database")
def health():
    """Checks that the app can connect to the database."""
    try:
        check_database_connection()
        return {"status": "ok", "database": "connected"}
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Database is not connected")


@router.post("/sessions", summary="Create a chat session")
def create_session(
    data: SessionCreate | None = None,
    db: Session = Depends(get_session),
):
    """Creates a new chat session."""
    data = data or SessionCreate()

    if data.user_id:
        user = db.query(User).filter(User.id == data.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    else:
        user = get_default_user(db)

    chat_session = ChatSession(
        session_id=str(uuid.uuid4()),
        user_id=user.id,
        title=data.title,
    )
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)

    return {
        "session_id": chat_session.session_id,
        "title": chat_session.title,
        "user_id": chat_session.user_id,
    }


@router.get("/sessions", summary="Get all chat sessions")
def get_sessions(db: Session = Depends(get_session)):
    """Returns all saved chat sessions."""
    sessions = db.query(ChatSession).all()

    return [
        {
            "session_id": session.session_id,
            "title": session.title,
            "user_id": session.user_id,
            "faiss_index_path": session.faiss_index_path,
        }
        for session in sessions
    ]


@router.get("/sessions/{session_id}", summary="Get one chat session")
def get_one_session(session_id: str, db: Session = Depends(get_session)):
    """Returns one chat session by session_id."""
    session = get_session_or_error(db, session_id)

    return {
        "session_id": session.session_id,
        "title": session.title,
        "user_id": session.user_id,
        "faiss_index_path": session.faiss_index_path,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


@router.delete("/sessions/{session_id}", summary="Delete one chat session")
def delete_session(session_id: str, db: Session = Depends(get_session)):
    """Deletes one session, plus its messages and document records."""
    session = get_session_or_error(db, session_id)

    db.query(Message).filter(Message.session_id == session_id).delete()
    db.query(DocumentRecord).filter(DocumentRecord.session_id == session_id).delete()
    db.delete(session)
    db.commit()

    return {"message": "Session deleted"}


@router.post(
    "/sessions/{session_id}/documents",
    summary="Upload documents and build FAISS index",
)
async def upload_documents(
    session_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_session),
):
    """Saves files, creates chunks, and stores them in FAISS."""
    session = get_session_or_error(db, session_id)

    upload_folder = Path("data") / session_id
    faiss_folder = Path("faiss_index") / session_id
    upload_folder.mkdir(parents=True, exist_ok=True)
    faiss_folder.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for file in files:
        file_name = Path(file.filename).name
        file_path = upload_folder / file_name
        file_path.write_bytes(await file.read())
        saved_paths.append(file_path)

    try:
        ingestor = ChatIngestor(session_id=session_id)
        chunks = ingestor.build_retriever_from_paths(saved_paths)

        faiss_manager = FaissManager(faiss_folder)
        faiss_manager.add_new_documents(chunks)

        for file_path in saved_paths:
            document = DocumentRecord(
                session_id=session_id,
                original_filename=file_path.name,
                stored_path=str(file_path),
                file_type=file_path.suffix.replace(".", ""),
                status="indexed",
            )
            db.add(document)

        session.faiss_index_path = str(faiss_folder)
        db.commit()

        return {
            "message": "Documents uploaded and indexed",
            "session_id": session_id,
            "files": [path.name for path in saved_paths],
            "chunks": len(chunks),
            "faiss_index_path": str(faiss_folder),
        }

    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {error}",
        )


@router.post("/sessions/{session_id}/chat", summary="Ask question using RAG")
def chat_with_session(
    session_id: str,
    data: ChatRequest,
    db: Session = Depends(get_session),
):
    """Loads FAISS, asks the RAG chain, and saves the chat messages."""
    session = get_session_or_error(db, session_id)

    if not session.faiss_index_path:
        raise HTTPException(
            status_code=400,
            detail="Upload documents before asking questions.",
        )

    try:
        chat_history = get_chat_history(db, session_id)

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(session.faiss_index_path)
        answer = rag.invoke(data.question, chat_history=chat_history)

        db.add(Message(session_id=session_id, role="user", content=data.question))
        db.add(Message(session_id=session_id, role="assistant", content=answer))
        db.commit()

        return {
            "session_id": session_id,
            "question": data.question,
            "answer": answer,
        }

    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Chat failed: {error}")


@router.get("/sessions/{session_id}/messages", summary="Get chat messages")
def get_messages(session_id: str, db: Session = Depends(get_session)):
    """Returns all messages for one chat session."""
    get_session_or_error(db, session_id)

    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at)
        .all()
    )

    return [
        {
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at,
        }
        for message in messages
    ]
