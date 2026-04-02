import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from src.document_processor import process_document
from src.supabase_client import supabase, insert_document_chunks

# 1. Setup Local Embedding Model
print("Loading Local AI Embedding Model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ingest_financial_document(file_path: str):
    """The master pipeline: Read -> Chunk -> Vectorize -> Upload."""
    chunks = process_document(file_path)
    
    print(f"Generating vectors for {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks).tolist() 
    
    # Extract the filename from the path
    document_name = os.path.basename(file_path)
    
    # Upload to Supabase
    insert_document_chunks(document_name, chunks, embeddings)
    
    # FIXED: Using the correct variable name for the success message
    print(f"🚀 Ingestion Complete for: {document_name}")

def answer_financial_question(question: str):
    """The Hybrid RAG Pipeline: Local Math + Groq Reasoning with Metadata."""
    
    # A. Local Vectorization
    query_vector = embedding_model.encode(question).tolist()

    # B. Vector Search in Supabase (Retrieving 20 chunks for depth)
    try:
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.3,
            "match_count": 20
        }).execute()
    except Exception as e:
        return {"error": f"Database Error: {str(e)}"}

    # Extract chunks and find unique sources
    context_chunks = [item['chunk_text'] for item in response.data]
    sources = list(set([item.get('document_name', 'Unknown') for item in response.data]))
    context_text = "\n\n".join(context_chunks)

    if not context_text:
        return {"answer": "I couldn't find any relevant information in the document.", "sources": []}

    # C. Cloud Reasoning with Groq
    print(f"Asking Groq (Llama 3.3 70B) with {len(context_chunks)} chunks...")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a Senior Equity Research Analyst. 
                    Your goal is to provide a High-Fidelity Financial Report based on the provided 10-K context.
                    
                    STYLE GUIDE:
                    1. Use Markdown Tables for all year-over-year (YoY) comparisons.
                    2. Use Bold Headers (###) for distinct sections like Revenue, Segments, and Risks.
                    3. Use Bullet Points for lists of specific items (e.g., Acquisitions or Geographic data).
                    4. If a specific dollar amount ($) exists, ALWAYS include it. 
                    5. Add a '💡 Analyst Commentary' section at the end with a 1-sentence strategic insight.
                    
                    STRICT RULES:
                    - If data is missing for a specific field, do not make it up; omit the section.
                    - Ensure clean spacing between sections using '---' horizontal rules.
                    - Format numbers clearly (e.g., $402.8B instead of 402836000000).
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