import sys
import json
import re
import sqlite3
import math
import ollama

DB_PATH = "catlib.db"

CHAT_MODEL = "llama3.2:latest"
EMBED_MODEL = "nomic-embed-text:latest"


def get_connection():
    return sqlite3.connect(DB_PATH)


def parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def get_embedding(text):
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    if mag_a == 0 or mag_b == 0:
        return 0

    return dot / (mag_a * mag_b)


def generate_search_tags(prompt):
    instruction = f"""
You are generating search tags for a cat meme/media database.

Every media item is already a cat, so do not use generic tags like:
cat, kitten, cute, meme, funny, animal, pet.

Your goal is retrieval quality.

Do not create compound words.
Bad tags:
pickuprequest, sisterpicksmeup, schoolrun, familymember

Good tags:
pickup, ride, car, driving, waiting, school, leaving, hurry, obligation, responsibility, annoyed

Texting situation:
{prompt}

Rules:
- Return ONLY valid JSON.
- Return exactly one JSON object, not a list.
- Do not explain your answer.
- Do not include markdown.
- Use lowercase.
- Use normal reusable search tags.
- Prefer single words.
- Short phrases are okay only if natural.
- Include actions, emotions, situations, reactions, and meme concepts.
- Return 12 tags.

Format:
{{
  "tags": ["pickup", "ride", "waiting"]
}}
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": instruction}]
    )

    raw = response["message"]["content"].strip()

    try:
        data = parse_json(raw)
    except Exception:
        print("Model returned invalid JSON:")
        print(raw)
        return []

    if isinstance(data, dict):
        tags = data.get("tags", [])
    elif isinstance(data, list):
        tags = []
        for group in data:
            if isinstance(group, dict):
                tags.extend(group.get("tags", []))
    else:
        tags = []

    clean_tags = []

    banned = {
        "cat", "cats", "kitten", "kittens",
        "cute", "meme", "funny", "animal", "pet"
    }

    for tag in tags:
        tag = str(tag).strip().lower()

        if not tag:
            continue

        if tag in banned:
            continue

        clean_tags.append(tag)

    return clean_tags


def get_media_items():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, file_path, file_name, media_type, caption, tags, favorite
        FROM media_items
        WHERE tags != ''
        ORDER BY id;
    """)

    rows = cursor.fetchall()
    conn.close()

    items = []

    for row in rows:
        item_id, file_path, file_name, media_type, caption, tags, favorite = row

        items.append({
            "id": item_id,
            "file_path": file_path,
            "file_name": file_name,
            "media_type": media_type,
            "caption": caption or "",
            "tags": tags or "",
            "favorite": bool(favorite),
        })

    return items


def item_search_text(item):
    return item["tags"].strip()


def rerank_top_candidates(prompt, candidates, max_picks=3):
    option_lines = []

    for index, item in enumerate(candidates, start=1):
        option_lines.append(
            f"{index}. ID: {item['id']}\n"
            f"Caption: {item['caption']}\n"
            f"Tags: {item['tags']}\n"
            f"Vector score: {item['score']:.3f}"
        )

    option_text = "\n\n".join(option_lines)

    instruction = f"""
You are choosing the best cat media for a texting situation.

The candidates were already selected by tag similarity.
Now choose the best final matches using the captions.

Texting situation:
{prompt}

Candidates:
{option_text}

Rules:
- Return ONLY valid JSON.
- Return exactly one JSON object, not a list.
- Do not include markdown.
- You MUST return exactly {max_picks} recommendations.
- Always choose the best available options.
- Never return fewer than {max_picks}.
- Use only index numbers from the candidate list.
- Prefer captions that match the situation, emotion, reaction, or joke.
- Do not pick based only on being a cat.
- Include a short explanation for why each selected item fits.

Format:
{{
  "recommendations": [
    {{
      "index": 1,
      "reason": "short explanation"
    }},
    {{
      "index": 2,
      "reason": "short explanation"
    }},
    {{
      "index": 3,
      "reason": "short explanation"
    }}
  ]
}}
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": instruction}]
    )

    raw = response["message"]["content"].strip()

    try:
        data = parse_json(raw)
    except Exception:
        print("Reranker returned invalid JSON:")
        print(raw)

        fallback = []
        for candidate in candidates[:max_picks]:
            item = dict(candidate)
            item["reason"] = "Strong semantic match from tag search."
            fallback.append(item)

        return fallback

    recommendations = data.get("recommendations", [])

    selected = []
    used_ids = set()

    for rec in recommendations:
        try:
            idx = int(rec.get("index"))
            reason = str(rec.get("reason", "")).strip()

            if 1 <= idx <= len(candidates):
                item = dict(candidates[idx - 1])

                if item["id"] in used_ids:
                    continue

                item["reason"] = reason or "Strong caption match."
                selected.append(item)
                used_ids.add(item["id"])

        except (ValueError, TypeError, AttributeError):
            pass

    for candidate in candidates:
        if len(selected) >= max_picks:
            break

        if candidate["id"] not in used_ids:
            item = dict(candidate)
            item["reason"] = "Strong semantic match from tag search."
            selected.append(item)
            used_ids.add(item["id"])

    return selected[:max_picks]


def find_media(prompt, limit=10):
    print(f"Prompt: {prompt!r}\n")

    items = get_media_items()
    print(f"Media options: {len(items)} tagged items")

    query_tags = generate_search_tags(prompt)

    if not query_tags:
        print("No search tags generated.")
        return []

    query_text = ", ".join(query_tags)

    print(f"Generated search tags: {query_text}\n")

    query_embedding = get_embedding(query_text)

    scored_items = []

    for item in items:
        tags_text = item_search_text(item)

        if not tags_text:
            continue

        item_embedding = get_embedding(tags_text)
        score = cosine_similarity(query_embedding, item_embedding)

        scored_items.append({
            **item,
            "score": score,
        })

    scored_items.sort(key=lambda item: item["score"], reverse=True)

    candidates = scored_items[:limit]

    print(f"Top {len(candidates)} candidate(s) from tag search:\n")

    for rank, item in enumerate(candidates, start=1):
        fav = "★" if item["favorite"] else " "
        print(f"{rank}. [{fav}] ID {item['id']} — {item['file_name']} | Score: {item['score']:.3f}")

    print()

    selected = rerank_top_candidates(prompt, candidates, max_picks=3)

    print(f"{len(selected)} final recommendation(s):\n")

    for rank, item in enumerate(selected, start=1):
        fav = "★" if item["favorite"] else " "

        print(f"{rank}. [{fav}] ID {item['id']} — {item['file_name']}")
        print(f"   Score: {item['score']:.3f}")
        print(f"   Reason: {item.get('reason', '')}")
        print(f"   Type: {item['media_type']}")
        print(f"   Path: {item['file_path']}")
        print(f"   Caption: {item['caption']}")
        print(f"   Tags: {item['tags']}")
        print()

    return selected


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python kitty_ai.py "your prompt here"')
        sys.exit(1)

    find_media(" ".join(sys.argv[1:]))