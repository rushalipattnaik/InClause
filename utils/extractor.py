import os

from docx import Document
from pypdf import PdfReader

from utils.text_cleanup import clean_text

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == ".docx":
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            return "Error: Unsupported file type. Please upload a PDF, DOCX, or TXT file."

        if not text.strip():
            return ("Error: No selectable text found. If this is a scanned PDF or a photo, "
                    "it needs OCR, which is not supported yet.")
        return clean_text(text)
    except Exception as e:
        return f"Error: Could not read this file ({str(e)})."
