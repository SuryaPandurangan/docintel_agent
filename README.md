# 🧠 DocIntel Agent – Gemini-Powered Multi-Document Q&A with Evaluation

A fully integrated GenAI document assistant and **RAG evaluation framework**, powered by **LangChain**, **Gemini Pro**, **HuggingFace Embeddings**, and **Streamlit**.

> “Upload docs → Ask Questions → Get Answers + Evaluation”

---

## 🚀 Features

- 📄 Upload multiple PDF or DOCX files
- 🧠 Extract, chunk, and embed using local `MiniLM` model
- 🔍 RAG pipeline using Gemini 2.5 Flash LLM
- 💬 Memory: Keep Q&A history with full context trace
- 📚 Show **source snippets** for every answer
- 📈 **LLM-as-Critic** evaluation (Relevance, Groundedness, Fluency) using Gemini itself
- 📊 **Dashboard** for end-to-end RAG evaluation:
  - Retrieval: `Recall@K`, `Precision@K`, `MRR`, `Hit@K`
  - Generation: `BLEU`, `ROUGE`, `BERTScore`
  - Diagnostics: `Latency`, `Coverage`, `Robustness`

---

## 🛠️ Tech Stack

| Layer         | Tool                                       |
|---------------|--------------------------------------------|
| UI            | Streamlit                                 |
| LLM           | Gemini 2.5 Flash (via `langchain-google-genai`) |
| Embeddings    | HuggingFace `all-MiniLM-L6-v2`             |
| Vector Store  | FAISS                                      |
| Evaluation    | HuggingFace `evaluate`, Gemini as critic   |
| File Parsing  | PyMuPDF, Docx2txt                          |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/docintel-agent.git
cd docintel-agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
