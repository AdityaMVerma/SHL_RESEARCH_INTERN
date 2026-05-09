import json

INPUT_FILE = "data\shl_product_catalog.json"
OUTPUT_FILE = "data\shl_product_catalog_with_search.json"

def clean_text(text):
   
    if not text:
        return ""

    return " ".join(str(text).lower().split())


def build_search_text(item):
    """
    Build semantic search text using:
    - name
    - description
    - keys
    """

    parts = []

    # Name
    parts.append("Role :")
    parts.append(item.get("name", ""))

    # Description
    parts.append("Description :")
    parts.append(item.get("description", ""))

    # Keys
    parts.append("Tests available :")
    parts.append(", ".join(item.get("keys", [])))

    combined = " ".join(parts)

    return clean_text(combined)


# ===== LOAD JSON =====

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ===== ADD SEARCH TEXT =====

for item in data:
    item["search_text"] = build_search_text(item)

# ===== SAVE UPDATED JSON =====

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Done. Updated JSON saved as: {OUTPUT_FILE}")