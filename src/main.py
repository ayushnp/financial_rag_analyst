from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_engine import answer_financial_question, ingest_financial_document

app = FastAPI(title="Financial RAG Analyst")

# Define what a "Request" looks like
class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Financial RAG API is live!"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    result = answer_financial_question(request.question)
    # This will now return the answer + sources + chunk_count to the UI
    return result

# This allows you to upload new PDFs via API later
@app.post("/ingest")
def ingest_file(file_path: str):
    ingest_financial_document(file_path)
    return {"message": f"Successfully ingested {file_path}"}