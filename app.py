import streamlit as st
from dotenv import load_dotenv
import os
from doc_loader import load_document
from qa_chain import build_qa_chain
from vector_store import create_vector_store

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

        st.success("Ready to answer your questions!")
        user_q = st.text_input("Ask a question about the document:")
        if user_q:
            response = qa_chain.invoke({"query": user_q})
            st.markdown(f"**Answer:** {response['result']}")

            with st.expander("🔍 Source Snippets"):
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
