from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_engine import answer_financial_question, ingest_financial_document
import os
import uvicorn

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

if __name__ == "__main__":
    # Render provides the port via the PORT environment variable
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)