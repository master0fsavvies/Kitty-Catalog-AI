import argparse
import csv
import hashlib
import os
import re
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import kitty_ai


DB_PATH = kitty_ai.DB_PATH
DEFAULT_OUTPUT = TESTS_DIR / "test.csv"
DEFAULT_MODEL = os.environ.get("KITTY_TEXT_MODEL", "llama3.2:latest")
CSV_FIELDS = (
    "test_id",
    "expected_id",
    "expected_file_path",
    "expected_file_name",
    "context",
)
GENERATION_ATTEMPTS = 3


def load_photos():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return conn.execute(
            """
            SELECT id, file_path, file_name, caption, tags, example_context
            FROM media_items
            WHERE media_type = 'image'
            ORDER BY id
            """
        ).fetchall()


def _ollama_chat(model, prompt):
        import ollama

        return ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )


def normalize_context(context):
    return " ".join(re.sub(r"[^\w\s]", " ", context.casefold()).split())


def clean_generated_context(response):
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return ""
    sentence = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", lines[0])
    sentence = sentence.strip().strip("\"'`“”‘’ ")
    match = re.search(r"[.!?](?:\s|$)", sentence)
    if match:
        sentence = sentence[:match.start() + 1]
    return " ".join(sentence.split())


def generate_context(caption, tags, existing_contexts, model=DEFAULT_MODEL):
    existing_keys = {normalize_context(context) for context in existing_contexts}
    description = caption.strip() or "No caption is available."
    tag_text = tags.strip() or "No tags are available."
    prompt = f"""Create exactly one short, natural sentence that could be used as a
test query to find this image in a cat meme library.

Image description: {description}
Image tags: {tag_text}

Existing example contexts for this same image (do not repeat or closely paraphrase
any of them):
{chr(10).join('- ' + item for item in existing_contexts) or '- None'}

Write a distinct context grounded in the description or tags. Return only the
sentence, without quotes, numbering, or an explanation."""

    for _ in range(GENERATION_ATTEMPTS):
        response = _ollama_chat(model, prompt)
        context = clean_generated_context(
            response["message"]["content"].strip()
        )
        if context and normalize_context(context) not in existing_keys:
            return context

    raise RuntimeError(
        "Ollama did not produce a new context after "
        f"{GENERATION_ATTEMPTS} attempts."
    )


def make_test_id(image_id, context):
    digest = hashlib.sha256(
        f"{image_id}\0{normalize_context(context)}".encode("utf-8")
    ).hexdigest()[:12]
    return f"image-{image_id}-{digest}"


def write_tests(output_path, model=DEFAULT_MODEL):
    photos = load_photos()
    tests = []
    for index, (image_id, file_path, file_name, caption, tags, raw_contexts) in enumerate(
        photos, start=1
    ):
        existing_contexts = kitty_ai.parse_contexts(raw_contexts)
        context = generate_context(caption or "", tags or "", existing_contexts, model)
        tests.append(
            {
                "test_id": make_test_id(image_id, context),
                "expected_id": image_id,
                "expected_file_path": file_path,
                "expected_file_name": file_name,
                "context": context,
            }
        )
        print(f"Generated {index}/{len(photos)}: {file_name}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(tests)
    print(f"Wrote {len(tests)} tests to {output_path}")
    return len(tests)


def main():
    parser = argparse.ArgumentParser(
        description="Generate one novel retrieval-test context for every catalog image."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    write_tests(args.output, args.model)


if __name__ == "__main__":
    main()
