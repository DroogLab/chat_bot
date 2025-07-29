import os

def load_text(file_path):
    """
    Loads and extracts text from various file formats: .txt, .md, .pdf, .docx.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support. Install with pip install pypdf2")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif ext == ".docx":
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx is required for DOCX support. Install with pip install python-docx")
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: .txt, .md, .pdf, .docx")