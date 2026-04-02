from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_engine import answer_financial_question, ingest_financial_document
import uvicorn
from fastapi import FastAPI, UploadFile, File 
import shutil
import os

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
async def ingest_file(file: UploadFile = File(...)):
    # 1. Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # 2. Save the uploaded file locally on the Backend server
    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 3. Process the file now that it actually exists on this server
    ingest_financial_document(file_path)
    
    return {"message": f"Successfully processed {file.filename}"}

if __name__ == "__main__":
    # Render provides the port via the PORT environment variable
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)