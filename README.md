# 🦈 DeepAudit.ai --- Enterprise Financial RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Supabase](https://img.shields.io/badge/Supabase-VectorDB-purple)
![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**DeepAudit.ai** is an **enterprise-grade financial intelligence
system** that automates the analysis of complex **SEC 10-K filings**
using a powerful **Retrieval Augmented Generation (RAG)** pipeline.

The system combines:

-   **Llama 3.3 (70B)** for deep financial reasoning\
-   **Cohere Embed v3** for high-precision semantic search\
-   **Supabase + pgvector** for scalable vector storage

This enables analysts, researchers, and developers to **extract insights
from dense financial documents in seconds instead of hours.**

------------------------------------------------------------------------

# 🚀 Features

### ⚡ Enterprise Financial RAG Pipeline

Processes **hundreds of pages of financial reports** and extracts
relevant insights using semantic search.

### 🧠 70B Parameter Reasoning

Uses **Llama 3.3 (70B) via Groq** for:

-   Financial trend analysis\
-   Year-over-Year comparisons\
-   Risk interpretation\
-   Structured financial summaries

### 🔍 High-Fidelity Retrieval

Uses **Cohere Embed v3 (1024 dimensions)** designed to handle:

-   Financial terminology\
-   Noisy PDF text\
-   Long documents

### 🗄️ Hybrid Vector Search

Powered by **Supabase pgvector** for fast similarity search with
configurable thresholds.

### ☁️ Lightweight Backend

The architecture offloads heavy computations to **external APIs**,
enabling deployment even on **low-memory cloud instances like Render**.

### 📄 Large Document Processing

Robust ingestion pipeline supports **500+ page filings** with chunking
and batch embedding.

------------------------------------------------------------------------

# 🛠 Tech Stack

  Layer                 Technology
  --------------------- ----------------------------------
  **Frontend**          Streamlit
  **Backend API**       FastAPI
  **LLM Reasoning**     Llama 3.3 70B (Groq)
  **Embeddings**        Cohere Embed English v3
  **Vector Database**   Supabase (PostgreSQL + pgvector)
  **PDF Processing**    Python
  **Deployment**        Render + Streamlit Cloud

------------------------------------------------------------------------

# 🏗 System Architecture

                     ┌─────────────────────┐
                     │     Streamlit UI    │
                     │  (User Interaction) │
                     └──────────┬──────────┘
                                │
                                ▼
                       ┌────────────────┐
                       │   FastAPI API  │
                       │  RAG Backend   │
                       └────────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                   ▼
     ┌───────────────┐                   ┌─────────────────┐
     │ Cohere Embed  │                   │  Groq LLM API   │
     │ 1024-dim vecs │                   │ Llama 3.3 70B   │
     └───────┬───────┘                   └─────────┬───────┘
             │                                     │
             ▼                                     ▼
                    ┌───────────────────────────┐
                    │    Supabase pgvector DB   │
                    │  Semantic Document Store  │
                    └───────────────────────────┘

------------------------------------------------------------------------

# 📂 Project Structure

    .
    ├── src/
    │   ├── main.py               # FastAPI entry point
    │   ├── rag_engine.py         # RAG pipeline logic
    │   ├── document_processor.py # PDF parsing & chunking
    │   ├── supabase_client.py    # Vector DB connection
    │   └── config.py             # Environment configuration
    ├── data/                     # Temporary document storage
    ├── requirements.txt          # Dependencies
    └── streamlit_app.py          # Streamlit frontend

------------------------------------------------------------------------

# ⚙️ Setup & Installation

## 1️⃣ Clone Repository

    git clone https://github.com/ayushnp/financial_rag_analyst.git
    cd financial_rag_analyst

------------------------------------------------------------------------

## 2️⃣ Configure Environment Variables

Create `.env` file:

    GROQ_API_KEY=your_groq_key
    COHERE_API_KEY=your_cohere_key
    SUPABASE_URL=your_supabase_url
    SUPABASE_SERVICE_KEY=your_supabase_key

------------------------------------------------------------------------

## 3️⃣ Setup Supabase Database

Run in Supabase SQL Editor:

``` sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
  id BIGSERIAL PRIMARY KEY,
  document_name TEXT,
  chunk_text TEXT,
  embedding VECTOR(1024)
);

CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(1024),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  id BIGINT,
  document_name TEXT,
  chunk_text TEXT,
  similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    document_chunks.id,
    document_chunks.document_name,
    document_chunks.chunk_text,
    1 - (document_chunks.embedding <=> query_embedding) AS similarity
  FROM document_chunks
  WHERE 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
  ORDER BY document_chunks.embedding <=> query_embedding
  LIMIT match_count;
$$;
```

------------------------------------------------------------------------

## 4️⃣ Run Locally

### Start Backend

    uvicorn src.main:app --reload

### Start Frontend

    streamlit run streamlit_app.py

------------------------------------------------------------------------

# 📈 Future Roadmap

-   [ ] Multi-document financial comparison
-   [ ] Async document ingestion pipeline
-   [ ] Source citation with PDF page references
-   [ ] Analyst-grade financial dashboards

------------------------------------------------------------------------

# 👤 Author

**Ayush N P**\
CSE Student --- Vidyavardhaka College of Engineering\
Aspiring **Data Scientist & AI Engineer**

GitHub: https://github.com/ayushnp
