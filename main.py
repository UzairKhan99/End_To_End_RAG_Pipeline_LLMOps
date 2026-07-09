from fastapi import FastAPI

from multi_doc_chat.Routes.routing import router
from multi_doc_chat.db.database import init_db


app = FastAPI(title="RAG Pipeline API")


@app.on_event("startup")
def start_app():
    init_db()


app.include_router(router)


@app.get("/")
def home():
    return {"message": "RAG Pipeline API is running"}
