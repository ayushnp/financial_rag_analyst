from supabase import create_client, Client
from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

# 1. Initialize the Supabase connection securely
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def insert_document_chunks(document_name: str, chunks: list[str], embeddings: list[list[float]]):
    """Pushes text chunks and their mathematical vectors to the cloud database."""
    print(f"Uploading {len(chunks)} chunks to Supabase...")
    
    # Package the data exactly how our SQL table expects it
    data_to_insert = []
    for i in range(len(chunks)):
        data_to_insert.append({
            "document_name": document_name,
            "chunk_text": chunks[i],
            "embedding": embeddings[i]
        })
        
    # Execute the insert command
    try:
        response = supabase.table("document_chunks").insert(data_to_insert).execute()
        print("✅ Successfully saved to Supabase!")
        return response
    except Exception as e:
        print(f"❌ Error uploading to Supabase: {e}")
        raise e