# Kitty CATalog — Context-Only Search Update

## What changed

`kitty_ai.py` now reads only `example_context` for retrieval. The user message and each example context are embedded using Ollama `nomic-embed-text`; the best cosine similarity among an item's contexts determines its score. The three highest-scoring items are returned to the existing Telegram bot. No tags, captions, search-tag LLM, or LLM reranking participate in retrieval.

Context vectors are cached in `media_items.context_embedding` as JSON with model name, source contexts, and vectors. Changing example contexts or the embedding model automatically invalidates that item's cache. The first query can take time while context embeddings are generated. The DB is updated locally; keep a backup.

## Windows quick start (PowerShell, in project directory)

1. Install Python 3.11+ and Ollama. Start Ollama and run `ollama pull nomic-embed-text`.
2. Create environment: `py -m venv .venv`
3. Activate: `.\.venv\Scripts\Activate.ps1` (if blocked, use `.venv\Scripts\python.exe` directly).
4. Install retrieval dependency: `python -m pip install ollama`.
5. **Best:** copy your existing populated `catlib.db` into this folder. It is deliberately NOT included in the ZIP. Back it up first.
6. If you have no database: `python create_db.py` then `python scan_media.py`. This creates records but does **not** invent example contexts.
7. Populate example contexts using your existing Streamlit editor (`pip install streamlit` then `streamlit run app.py`) or your existing `generate_metadata.py` workflow. That generator has additional model/system dependencies; inspect its setup before running it. Contexts should accurately describe conversational situations.
8. Test: `python kitty_ai.py "I hate her"` and `python kitty_ai.py "I will be right there"`.
9. Automated offline tests: `python -m unittest discover -s tests -v`.
10. Optional labeled evaluation: create a CSV with `query,expected_id` and run `python evaluate_context.py labels.csv`.

## Required real-data validation

The ZIP did not include `catlib.db`, so the two target rankings **have not been verified on real video metadata**. Offline unit tests use a temporary database and mocked embeddings to verify the algorithm, not real-world ranking quality. Find the actual video IDs, create accurate context examples, run the real Ollama model, and record results. Do not hardcode desired IDs or report mock-test accuracy as experimental results.

## GitHub handover

Use your own GitHub account: create a branch, upload changed `.py` files, tests, and this README, then open a pull request. Never upload `token.txt`, private databases, secrets, or personal videos without authorization. `.gitignore` already excludes `catlib.db`. The original `token.txt` was deliberately omitted from this deliverable; replace/rotate any token that was previously shared or committed.

## Academic report status

The rubric requires measured experiments, baselines, citations, conclusions, and genuine individual contributions within eight pages. A final evidence-based report cannot be completed from source code alone without the populated database, human-labeled evaluation queries, and the team's actual contributions. The evaluation script provides a starting point for real measurements.
