"""Context-only semantic retrieval for Kitty CATalog.

No tags, captions, or chat-model reranking are used to select media.
The public find_media(prompt, limit=10) interface remains compatible with the bot.
"""
import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = os.environ.get('KITTY_DB_PATH', str(Path(__file__).resolve().parent / 'catlib.db'))
EMBED_MODEL = os.environ.get('KITTY_EMBED_MODEL', 'nomic-embed-text:latest')


def get_connection():
    return sqlite3.connect(DB_PATH, timeout=30)


def parse_contexts(value):
    """Support JSON arrays, the generator's comma-separated JSON strings, and lines."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        raw = str(value).strip()
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
            values = decoded if isinstance(decoded, list) else [decoded]
        except (ValueError, TypeError):
            try:
                decoded = json.loads('[' + raw + ']')
                values = decoded if isinstance(decoded, list) else [raw]
            except (ValueError, TypeError):
                values = raw.splitlines()
    result, seen = [], set()
    for item in values:
        if not isinstance(item, str):
            continue
        item = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', item).strip()
        item = ' '.join(item.split())
        if item and item.casefold() not in seen:
            seen.add(item.casefold())
            result.append(item)
    return result


def get_embedding(text):
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError('Install Ollama Python package: pip install ollama') from exc
    try:
        if hasattr(ollama, 'embed'):
            response = ollama.embed(model=EMBED_MODEL, input=text)
            vector = response['embeddings'][0]
        else:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            vector = response['embedding']
        if not vector or not all(math.isfinite(float(v)) for v in vector):
            raise ValueError('Embedding contains invalid values')
        return [float(v) for v in vector]
    except Exception as exc:
        raise RuntimeError(f'Could not embed text using Ollama model {EMBED_MODEL}. '
                           'Ensure Ollama is running and the model is pulled.') from exc


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0.0


def _read_cache(value, contexts):
    if not value:
        return {}
    try:
        cache = json.loads(value)
        if (isinstance(cache, dict) and cache.get('model') == EMBED_MODEL
                and cache.get('contexts') == contexts and isinstance(cache.get('vectors'), list)
                and len(cache['vectors']) == len(contexts)):
            vectors = cache['vectors']
            if all(isinstance(v, list) and v and all(isinstance(x, (float, int)) and math.isfinite(x) for x in v)
                   for v in vectors) and len({len(v) for v in vectors}) == 1:
                return cache
    except (ValueError, TypeError):
        pass
    return {}


def resolve_media_path(stored_path):
    """Resolve Windows-style DB paths on Windows, Linux and macOS."""
    if not stored_path:
        return ''
    path = Path(str(stored_path).replace('\\', '/'))
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def warm_context_cache():
    """Precompute missing/stale context vectors once (requires Ollama)."""
    items = get_media_items()
    with get_connection() as conn:
        for index, item in enumerate(items, 1):
            _get_context_vectors(conn, item)
            if index % 10 == 0 or index == len(items):
                print(f'Cached {index}/{len(items)} media items')
    return len(items)


def get_media_items():
    """Only context fields are loaded for ranking; no tag/caption dependencies."""
    with get_connection() as conn:
        rows = conn.execute('''SELECT id, file_path, file_name, media_type,
                                     example_context, context_embedding, favorite
                              FROM media_items ORDER BY id''').fetchall()
    items = []
    for item_id, path, name, kind, raw_contexts, cache, favorite in rows:
        contexts = parse_contexts(raw_contexts)
        if contexts:
            items.append({'id': item_id, 'file_path': resolve_media_path(path), 'file_name': name,
                          'media_type': kind, 'favorite': bool(favorite),
                          'contexts': contexts, 'cache': cache})
    return items


def _get_context_vectors(conn, item):
    contexts = item['contexts']
    cached = _read_cache(item['cache'], contexts)
    if cached:
        return cached['vectors']
    vectors = [get_embedding(context) for context in contexts]
    # Cache is keyed to both model and exact context strings. Edits automatically invalidate it.
    cache = json.dumps({'model': EMBED_MODEL, 'contexts': contexts, 'vectors': vectors})
    conn.execute('UPDATE media_items SET context_embedding = ? WHERE id = ?', (cache, item['id']))
    return vectors


def find_media(prompt, limit=10):
    """Return top-ranked matches; the Telegram bot displays only the first three.

    Video score = maximum cosine similarity across its example contexts.
    A single relevant example context can therefore retrieve a video.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return []
    if not isinstance(limit, int) or limit < 1:
        raise ValueError('limit must be a positive integer')
    items = get_media_items()
    if not items:
        print('No media with example_context found. Generate metadata first.')
        return []
    query_vector = get_embedding(prompt.strip())
    ranked = []
    with get_connection() as conn:
        for item in items:
            vectors = _get_context_vectors(conn, item)
            scored = [(cosine_similarity(query_vector, vector), context)
                      for vector, context in zip(vectors, item['contexts'])
                      if len(vector) == len(query_vector)]
            if not scored:
                continue
            score, matched_context = max(scored, key=lambda entry: entry[0])
            ranked.append({'id': item['id'], 'file_path': item['file_path'],
                           'file_name': item['file_name'], 'media_type': item['media_type'],
                           'favorite': item['favorite'], 'score': score,
                           'matched_context': matched_context,
                           'reason': f'Context match: {matched_context}'})
    ranked.sort(key=lambda item: (-item['score'], item['id']))
    selected = ranked[:limit]
    for rank, item in enumerate(selected, 1):
        print(f"{rank}. ID {item['id']} — {item['file_name']} | "
              f"score={item['score']:.4f} | {item['matched_context']}")
    return selected


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--warm-cache':
        try:
            warm_context_cache()
        except (sqlite3.Error, RuntimeError) as exc:
            print(f'Error: {exc}', file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    if len(sys.argv) < 2:
        print('Usage: python kitty_ai.py "your message"')
        sys.exit(1)
    try:
        find_media(' '.join(sys.argv[1:]))
    except (sqlite3.Error, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
