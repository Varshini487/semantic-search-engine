# 🔍 Semantic Search Engine

A **Semantic Search Engine** that understands the *meaning* behind your queries, not just keywords.

## 🧠 Semantic vs Keyword Search
| Feature | Keyword Search | Semantic Search |
|---------|---------------|-----------------|
| Matching | Exact words | Meaning/intent |
| "car" vs "automobile" | ❌ No match | ✅ Match |
| Context awareness | ❌ | ✅ |
| Multilingual | ❌ | ✅ |

## ⚙️ How It Works
1. Documents are encoded into **dense vector embeddings**
2. Query is encoded into the same space
3. **Cosine similarity** ranks results by semantic closeness
4. Top-k results returned with relevance scores

## 🛠️ Tech Stack
- **Sentence-Transformers** (all-MiniLM-L6-v2)
- **FAISS** – billion-scale vector search
- **FastAPI** – REST API
- **Streamlit** – search UI
- **Pinecone** (optional cloud vector DB)

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/semantic-search-engine
cd semantic-search-engine
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Use Cases
- Enterprise document search
- E-commerce product discovery
- Job listing matching
- Academic paper search
