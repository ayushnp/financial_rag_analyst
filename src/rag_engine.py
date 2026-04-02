import os
import cohere
from groq import Groq
from src.document_processor import process_document
from src.supabase_client import supabase, insert_document_chunks

# 1. Initialize Cohere (Trial Key - 1024 dimensions)
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# 2. Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ingest_financial_document(file_path: str):
    """Read -> Chunk -> Vectorize via Cohere API -> Upload."""
    chunks = process_document(file_path)
    
    print(f"Generating high-fidelity vectors for {len(chunks)} chunks via Cohere...")
    
    all_embeddings = []
    # Cohere trial keys handle up to 96 texts per request. We'll use a safe batch of 90.
    batch_size = 90 
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        
        # embed-english-v3.0 is optimized for RAG retrieval
        response = co.embed(
            texts=batch,
            model="embed-english-v3.0",
            input_type="search_document", # Context chunks are 'search_document'
            embedding_types=["float"]
        )
        all_embeddings.extend(response.embeddings.float)
        print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")

    document_name = os.path.basename(file_path)
    
    # Upload to Supabase
    insert_document_chunks(document_name, chunks, all_embeddings)
    
    print(f"🚀 Ingestion Complete for: {document_name}")

def answer_financial_question(question: str):
    """The Hybrid RAG Pipeline: Cohere Math + Groq Reasoning."""
    
    # A. Vectorize question via Cohere API
    query_response = co.embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query", # Questions are 'search_query'
        embedding_types=["float"]
    )
    query_vector = query_response.embeddings.float[0]

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
        return {"answer": "I couldn't find any relevant information in the 10-K.", "sources": []}

    # C. Cloud Reasoning with Groq (Llama 3.3 70B)
    print(f"Asking Groq for financial analysis...")
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