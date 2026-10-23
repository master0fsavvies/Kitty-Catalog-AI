import sqlite3
from collections import Counter

DB_PATH = "catlib.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def normalize_term(term):
    return term.strip().lower()


def split_comma_field(value):
    if not value:
        return []

    return [
        normalize_term(part)
        for part in value.split(",")
        if normalize_term(part)
    ]


def get_all_tags():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT tags FROM media_items WHERE tags != '';")
    rows = cursor.fetchall()
    conn.close()

    counter = Counter()

    for (tags,) in rows:
        counter.update(split_comma_field(tags))

    return sorted(counter.keys())


def get_vocabulary():
    return {
        "tags": get_all_tags()
    }


def search_media(selected_tags=None, limit=20):
    selected_tags = [normalize_term(t) for t in (selected_tags or [])]

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, file_path, file_name, media_type, caption, tags, favorite
        FROM media_items;
    """)

    rows = cursor.fetchall()
    conn.close()

    results = []

    for row in rows:
        item_id, file_path, file_name, media_type, caption, tags, favorite = row

        item_tags = set(split_comma_field(tags))

        score = 0
        matched_tags = []

        for tag in selected_tags:
            if tag in item_tags:
                score += 5
                matched_tags.append(tag)

        if favorite:
            score += 1

        if score > 0:
            results.append({
                "id": item_id,
                "file_path": file_path,
                "file_name": file_name,
                "media_type": media_type,
                "caption": caption,
                "tags": tags,
                "favorite": bool(favorite),
                "score": score,
                "matched_tags": matched_tags
            })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:limit]


if __name__ == "__main__":
    vocab = get_vocabulary()

    print("Tags:")
    print(vocab["tags"])

    print("\nTest search:")
    results = search_media(
        selected_tags=["angry", "meows", "frustrated"],
        limit=10
    )

    for result in results:
        print(result["score"], result["id"], result["file_name"])