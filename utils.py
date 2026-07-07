"""
Utility functions for the AI Resume Analyzer.
Handles file parsing (PDF / TXT) and small helper functions.
"""

import io
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file given as bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text_chunks = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_chunks.append(page_text)
    return "\n".join(text_chunks).strip()


def extract_text_from_upload(uploaded_file) -> str:
    """
    Given a Streamlit UploadedFile object, return plain text content.
    Supports .pdf and .txt files.
    """
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")


def truncate_text(text: str, max_chars: int = 12000) -> str:
    """Guard against extremely long resumes blowing up the context window."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"
