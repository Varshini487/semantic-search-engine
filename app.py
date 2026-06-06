from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import json

st.set_page_config(page_title="🔍 Semantic Search", layout="wide")
st.title("🔍 Semantic Search Engine")
st.markdown("Search by meaning, not just keywords")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Sample documents corpus
sample_docs = [
    "Python is a high-level programming language known for simplicity",
    "Machine learning enables computers to learn from data automatically",
    "Neural networks are inspired by the human brain structure",
    "Deep learning uses multiple layers to extract features from data",
    "Natural language processing helps computers understand human text",
    "Computer vision allows machines to interpret visual information",
    "Reinforcement learning trains agents through reward and punishment",
    "Transfer learning reuses pre-trained models for new tasks",
    "Data science combines statistics, programming, and domain expertise",
    "Artificial intelligence simulates human intelligence in machines",
]

st.sidebar.header("📚 Knowledge Base")
custom_docs = st.sidebar.text_area("Add your documents (one per line):", height=200)
docs = sample_docs + [d for d in custom_docs.split("\n") if d.strip()]

doc_embeddings = model.encode(docs, convert_to_numpy=True)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

query = st.text_input("🔎 Enter your search query:")
top_k = st.slider("Number of results:", 1, 10, 5)

if query:
    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, top_k)

    st.markdown("### 📋 Results")
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        st.markdown(f"**{i+1}.** {docs[idx]}")
        st.progress(float(score), text=f"Relevance: {score:.2%}")
