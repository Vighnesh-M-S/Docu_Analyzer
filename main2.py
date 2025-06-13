from llama_cpp import Llama
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load the llm
llm = Llama(model_path="Models/mistral-7b-instruct-v0.1.Q2_K.gguf", n_ctx=2048)

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(chunks)


# Load FAISS + Chunks
index = faiss.read_index("embedding_index.faiss")
with open("document_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Define a query
query = "Can they terminate my account?"

# Embed the query
query_emb = embed_model.encode([query])
_, I = index.search(np.array(query_emb), k=2)

# Retrieve context
context = "\n".join([chunks[i] for i in I[0]])

# Generate prompt
prompt = f"""You are a legal assistant. Based on the following contract terms, answer the user's question in simple terms.

Context:
{context}

Question: {query}
Answer:"""

# Generate response
response = llm(prompt, max_tokens=300, stop=["\n\n"])
print(response['choices'][0]['text'].strip())