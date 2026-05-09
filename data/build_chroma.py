import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="shl_assessments"
)

# Load JSON
with open("data/shl_product_catalog_with_search.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Add documents
for item in data:

    embedding = model.encode(item["search_text"]).tolist()

    collection.add(
        ids=[item["entity_id"]],
        embeddings=[embedding],
        documents=[item["search_text"]],
        metadatas=[{
            "name": item["name"],
            "url": item["link"],
            "remote": item["remote"],
            "adaptive": item["adaptive"]
        }]
    )

print("Chroma index built successfully.")