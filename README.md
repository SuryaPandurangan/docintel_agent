# 🧠 DocIntel Agent – Multi-Document Q&A with LLMs

A powerful GenAI-based document assistant that can read, chunk, embed, and answer questions across multiple PDF/DOCX files. Built with **LangChain**, **Gemini Pro**, **HuggingFace Embeddings**, and **Streamlit**.

> “Upload documents. Ask questions. Get answers with sources.”

---

## 🚀 Features

- 📄 Upload multiple PDF & DOCX documents
- 🧠 Extract text, OCR fallback for scanned docs
- 🧩 Chunk + embed using `all-MiniLM-L6-v2` (local)
- 🔍 RAG pipeline with Gemini LLM
- 🤖 Ask complex questions and get reliable answers
- 📚 See source snippets for each response
- 🖥️ Easy-to-use Streamlit UI

---

## 🛠️ Tech Stack

| Component    | Tool                                  |
|--------------|----------------------------------------|
| Frontend     | Streamlit                              |
| Backend      | Python + LangChain                     |
| LLM          | Gemini Pro (via `langchain-google-genai`) |
| Embedding    | HuggingFace (`all-MiniLM-L6-v2`)       |
| Vector Store | FAISS                                  |
| OCR          | Tesseract + pdf2image (for scanned PDFs) |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/docintel-agent.git
cd docintel-agent
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

GOOGLE_API_KEY=your_gemini_api_key_here

streamlit run app.py
