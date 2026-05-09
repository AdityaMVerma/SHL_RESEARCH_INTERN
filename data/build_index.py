import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load catalog
with open("data/shl_product_catalog_with_search.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract search texts
texts = [item["search_text"] for item in data]

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Convert to float32 for FAISS
embeddings = embeddings.astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

# Save index
faiss.write_index(index, "data/shl_index.faiss")

# Save embeddings optionally
np.save("data/shl_embeddings.npy", embeddings)

print(f"Indexed {len(data)} assessments.")