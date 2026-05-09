import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load catalog
with open("data/shl_product_catalog_with_search.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load FAISS index
index = faiss.read_index("data/shl_index.faiss")


def search_assessments(query, top_k=5):

    # Embed query
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        results.append(data[idx])

    return results


# Example
if __name__ == "__main__":

    query = "Need leadership and stakeholder management assessment"

    results = search_assessments(query)

    for r in results:
        print(r["name"])