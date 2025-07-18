import fitz  # PyMuPDF
import docx
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os


def load_pdf(path):
    try:
        text = ""
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text()
        if len(text.strip()) < 50:
            return ocr_pdf(path)
        return text
    except Exception:
        return ocr_pdf(path)


def ocr_pdf(path):
    text = ""
    images = convert_from_path(path)
    for img in images:
        text += pytesseract.image_to_string(img)
    return text


def load_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


def load_document(file_path):
    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".docx"):
        return load_docx(file_path)
    else:
        return ""
