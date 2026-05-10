import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================
# LOAD EMBEDDING MODEL
# =========================
model = None

def get_model():

    global model

    if model is None:

        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return model

# =========================
# LOAD FAISS INDEX
# =========================

index = faiss.read_index(
    "data/shl_index.faiss"
)


# =========================
# LOAD CATALOG METADATA
# =========================

with open(
    "data/shl_product_catalog_with_search.json",
    "r",
    encoding="utf-8"
) as f:

    catalog = json.load(f)


# =========================
# RETRIEVAL FUNCTION
# =========================

def retrieve_assessments(query, top_k=5):

    # Create embedding for query
    current_model = get_model()
    query_embedding = current_model.encode([query])


    # Convert to float32 for FAISS
    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    # Search FAISS index
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    # Retrieve matching catalog entries
    for idx in indices[0]:

        # Safety check
        if idx < 0 or idx >= len(catalog):
            continue

        item = catalog[idx]

        results.append({
            "name": item.get("name"),
            "url": item.get("link"),
            "description": item.get("description"),
            "keys": item.get("keys", []),
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "remote": item.get("remote"),
            "adaptive": item.get("adaptive")
        })

    return results


# =========================
# TEST
# =========================

if __name__ == "__main__":

    query = (
        "mid-level java developer "
        "stakeholder communication "
        "technical and behavioral assessments"
    )

    results = retrieve_assessments(
        query=query,
        top_k=5
    )

    print("\nTop Matches:\n")

    for i, result in enumerate(results, start=1):

        print(f"{i}. {result['name']}")
        print(f"URL: {result['url']}")
        print(f"Keys: {result['keys']}")
        print()