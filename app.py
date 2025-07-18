import streamlit as st
from dotenv import load_dotenv
import os
from doc_loader import load_document
from qa_chain import build_qa_chain
from vector_store import create_vector_store
import json
from langchain.schema import Document
import tempfile
import shutil

MEMORY_FILE = "long_term_memory.json"


def serialize_doc(doc):
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def save_memory(chat_history):
    serializable_history = []
    for item in chat_history:
        serializable_item = {
            "question": item["question"],
            "answer": item["answer"],
            "sources": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in item["sources"]
            ],
        }
        serializable_history.append(serializable_item)

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp_file:
        json.dump(serializable_history, tmp_file, indent=2)
        tmp_path = tmp_file.name

    shutil.move(tmp_path, MEMORY_FILE)


def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                raw_history = json.load(f)

                history = []
                for item in raw_history:
                    docs = [
                        Document(
                            page_content=d["page_content"],
                            metadata=d.get("metadata", {}),
                        )
                        for d in item["sources"]
                    ]
                    history.append(
                        {
                            "question": item["question"],
                            "answer": item["answer"],
                            "sources": docs,
                        }
                    )
                return history
        except json.JSONDecodeError:
            print("⚠️ Warning: Memory file is corrupted. Starting fresh.")
            os.remove(MEMORY_FILE)  # Optional: delete corrupt file
            return []
    return []


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="DocIntel Agent", layout="wide")
st.title("📄 DocIntel Agent")

# uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
uploaded_files = st.file_uploader(
    "Upload one or more PDF/DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        temp_path = f"temp/{file.name}"
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        text = load_document(temp_path)
        st.success(f"{file.name} loaded.")
        all_text += f"\n\n---\n\n{text}"  # Keep docs separated with delimiter

    st.text_area("Combined Text Preview", all_text[:3000], height=300)

    with st.spinner("Embedding all documents..."):
        chunks = [all_text[i : i + 1000] for i in range(0, len(all_text), 1000)]
        vectorstore = create_vector_store(chunks)
        qa_chain = build_qa_chain(vectorstore)

        st.markdown("## 💬 Ask Questions About Your Documents")

        # Init session history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = load_memory()

        # Show previous Q&A using chat bubbles
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(f"**{message['question']}**")
            with st.chat_message("assistant"):
                st.markdown(f"{message['answer']}")
                with st.expander("📚 Source Snippets"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f"**🔹 Snippet {i+1}:**")
                        st.write(doc.page_content)

        # User input box
        if prompt := st.chat_input("Ask a question about the documents..."):
            with st.chat_message("user"):
                st.markdown(f"**{prompt}**")

            with st.spinner("🤖 Thinking..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]

            with st.chat_message("assistant"):
                st.markdown(f"{answer}")
                with st.expander("📚 Source Snippets"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**🔹 Snippet {i+1}:**")
                        st.write(doc.page_content)

            # Save to session
            st.session_state.chat_history.append(
                {
                    "question": prompt,
                    "answer": answer,
                    "sources": sources,
                }
            )
            save_memory(st.session_state.chat_history)

        # Optional clear chat button
        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
            st.rerun()
