import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_path: str) -> str:
    """Reads a PDF file and extracts all raw text."""
    print(f"Reading {file_path}...")
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            # Extract text and add a space to prevent words merging across pages
            extracted = page.extract_text()
            if extracted:
                text += extracted + " \n"
    return text


def chunk_text(text: str) -> list[str]:
    """Splits a massive string of text into smaller, AI-readable chunks."""
    print("Chunking text into manageable paragraphs...")

    # This splitter is the industry standard for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The max length of each chunk
        chunk_overlap=200,  # Overlap prevents context loss between chunks
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    print(f"Successfully created {len(chunks)} text chunks.")
    return chunks


def process_document(file_path: str) -> list[str]:
    """Master function to read and chunk a PDF."""
    raw_text = extract_text_from_pdf(file_path)

    if not raw_text.strip():
        raise ValueError("Error: Could not extract any text from the PDF.")

    chunks = chunk_text(raw_text)
    return chunks