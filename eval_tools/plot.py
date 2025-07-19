import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


def plot_similarity_chart(query, docs, embedder):
    doc_texts = [doc.page_content for doc in docs]
    vectors = embedder.embed_documents(doc_texts)
    query_vec = embedder.embed_query(query)

    sims = cosine_similarity([query_vec], vectors)[0]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=[f"Chunk {i+1}" for i in range(len(sims))], y=sims, name="Similarity")
    )
    fig.update_layout(
        title="Cosine Similarity with Query", yaxis_title="Similarity", height=300
    )
    st.plotly_chart(fig, use_container_width=True)
