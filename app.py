import os
import json
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# Evaluation tools
from eval_tools import (
    plot_similarity_chart,
    save_eval_to_jsonl,
    llm_critic_eval,
)
from eval_tools.retrieval import precision_at_k, recall_at_k, mrr, hit_rate_at_k
from eval_tools.generation import compute_bleu, compute_rouge
from eval_tools.diagnostics import measure_latency, compute_coverage, robustness_score

# -----------------------------
# 🧠 Memory File I/O
# -----------------------------
MEMORY_FILE = "long_term_memory.json"
load_dotenv()


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
            os.remove(MEMORY_FILE)
            return []
    return []


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
    with open(MEMORY_FILE, "w") as f:
        json.dump(serializable_history, f, indent=2)


# -----------------------------
# ⚙️ Embeddings & LLM Setup
# -----------------------------
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
# -----------------------------
# 🎯 Streamlit UI
# -----------------------------
st.set_page_config(page_title="📄 DocIntel Agent", layout="wide")
st.title("📄 DocIntel Agent with Memory, Debugging & Evaluation")

uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True
)

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        suffix = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name

        if tmp_path.endswith(".pdf"):
            loader = PyMuPDFLoader(tmp_path)
        elif tmp_path.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.warning("Unsupported format.")
            continue

        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, embedder)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_memory()

    st.markdown("### 💬 Chat with your documents")

    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["question"])
        with st.chat_message("assistant"):
            st.markdown(message["answer"])
            with st.expander("📚 Source Snippets"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Snippet {i+1}:**")
                    st.write(doc.page_content)

    if prompt := st.chat_input("Ask something..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("🤖 Thinking..."):
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            sources = result["source_documents"]

        with st.chat_message("assistant"):
            st.markdown(answer)

            with st.expander("📚 Source Snippets"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Snippet {i+1}:**")
                    st.write(doc.page_content)

            with st.expander("🧠 Chunk Similarity"):
                plot_similarity_chart(prompt, sources, embedder)

            with st.expander("📈 LLM-as-Critic Evaluation (Gemini)"):
                try:
                    critic_scores = llm_critic_eval(prompt, answer, sources)
                    if isinstance(critic_scores, dict) and "score" in critic_scores:
                        # Flat structure
                        for k, v in critic_scores.items():
                            st.markdown(f"**{k}**: {v}")
                    else:
                        # Nested structure
                        for k, v in critic_scores.items():
                            st.markdown(f"**{k}**: {v['score']} — _{v['explanation']}_")
                    save_eval_to_jsonl(prompt, answer, sources, critic_scores)
                except Exception as e:
                    st.warning(f"Evaluation failed: {e}")

        st.session_state.chat_history.append(
            {"question": prompt, "answer": answer, "sources": sources}
        )
        save_memory(st.session_state.chat_history)

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        st.rerun()

    with st.sidebar:
        st.markdown("## 📊 Evaluation Dashboard")

        if "chat_history" in st.session_state and st.session_state.chat_history:
            last_chat = st.session_state.chat_history[-1]
            sources = last_chat["sources"]
            answer = last_chat["answer"]
            prompt = last_chat["question"]

            # Retrieval metrics
            retrieved_ids = [
                doc.metadata.get("source_id", str(i)) for i, doc in enumerate(sources)
            ]
            relevant_ids = retrieved_ids  # Placeholder until gold labels

            st.markdown("### 🔍 Retrieval")
            st.write(
                f"**Precision@5:** {precision_at_k(relevant_ids, retrieved_ids, k=5):.2f}"
            )
            st.write(
                f"**Recall@5:** {recall_at_k(relevant_ids, retrieved_ids, k=5):.2f}"
            )
            st.write(f"**MRR:** {mrr(relevant_ids, retrieved_ids):.2f}")
            st.write(f"**Hit@5:** {hit_rate_at_k(relevant_ids, retrieved_ids, k=5)}")

            # Generation metrics (compare with reference answer)
            references = ["<ground truth answer>"]  # Replace later
            predictions = [answer]

            st.markdown("### 🧠 Generation")
            bleu_score = compute_bleu(predictions, references)
            rouge_score = compute_rouge(predictions, references)
            # bert_score = compute_bertscore(predictions, references)

            st.write(f"**BLEU:** {bleu_score['bleu']:.2f}")
            st.write(f"**ROUGE-L:** {rouge_score['rougeL']:.2f}")
            # st.write(
            #     f"**BERTScore-F1:** {sum(bert_score['f1']) / len(bert_score['f1']):.2f}"
            # )

            # LLM-as-Critic
            st.markdown("### 🤖 LLM-as-Critic")
            critic = llm_critic_eval(prompt, answer, sources)
            st.json(critic)

            # Diagnostics
            st.markdown("### 🧪 Diagnostics")
            latency = measure_latency(qa_chain.invoke, {"query": prompt})
            st.write(f"**Latency:** {latency:.2f} seconds")
            st.write(
                f"**Coverage:** {compute_coverage(len(vectorstore.docstore._dict), len(sources)):.1f}%"
            )
            st.write(f"**Robustness:** {robustness_score(answer, answer)}")
        else:
            st.info("Ask a question to view evaluation metrics.")
