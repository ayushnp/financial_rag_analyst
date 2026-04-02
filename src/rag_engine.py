import os
import voyageai
from groq import Groq
from src.document_processor import process_document
from src.supabase_client import supabase, insert_document_chunks

# 1. Initialize Voyage AI (Weightless Embeddings)
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# 2. Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ingest_financial_document(file_path: str):
    """Read -> Chunk -> Vectorize via API -> Upload."""
    chunks = process_document(file_path)
    
    print(f"Generating high-fidelity vectors for {len(chunks)} chunks via Voyage AI...")
    
    # Using 'voyage-3' which is state-of-the-art for retrieval
    # Note: Voyage handles batches automatically
    embeddings = vo.embed(chunks, model="voyage-3", input_type="document").embeddings
    
    document_name = os.path.basename(file_path)
    insert_document_chunks(document_name, chunks, embeddings)
    
    print(f"🚀 Ingestion Complete for: {document_name}")

def answer_financial_question(question: str):
    """The Hybrid RAG Pipeline: Voyage Math + Groq Reasoning."""
    
    # A. Vectorize question via API
    query_vector = vo.embed([question], model="voyage-3", input_type="query").embeddings[0]

    # B. Vector Search in Supabase
    try:
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.3,
            "match_count": 20
        }).execute()
    except Exception as e:
        return {"error": f"Database Error: {str(e)}"}

    context_chunks = [item['chunk_text'] for item in response.data]
    sources = list(set([item.get('document_name', 'Unknown') for item in response.data]))
    context_text = "\n\n".join(context_chunks)

    if not context_text:
        return {"answer": "I couldn't find any relevant information.", "sources": []}

    # C. Cloud Reasoning with Groq
    print(f"Asking Groq (Llama 3.3 70B)...")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a Senior Equity Research Analyst. 
                    Your goal is to provide a High-Fidelity Financial Report.
                    
                    STYLE GUIDE:
                    1. Use Markdown Tables for YoY comparisons.
                    2. Use Bold Headers (###) for Revenue, Segments, and Risks.
                    3. Format numbers clearly (e.g., $402.8B).
                    4. Add a '💡 Analyst Commentary' section at the end.
                    """
                },
                {
                    "role": "user",
                    "content": f"CONTEXT FROM 10-K:\n{context_text}\n\nQUESTION: {question}"
                }
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.1,
        )
        
        return {
            "answer": chat_completion.choices[0].message.content,
            "sources": sources,
            "chunk_count": len(context_chunks)
        }
    except Exception as e:
        return {"error": f"Groq API Error: {str(e)}"}