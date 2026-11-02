# Kitty CATalog — Context-only retrieval

## Included
- The original project and media library, plus the client-provided 60-row SQLite database.
- `kitty_ai.py` retrieves only using `example_context` and `context_embedding`; captions and tags are not read for ranking.
- `tests/` contains offline unit tests; `evaluate_context.py` supports labeled-query evaluation.

## Windows setup
1. Install Python 3.10+ and Ollama from its official installer.
2. Extract the entire ZIP; open PowerShell **in the folder containing kitty_ai.py**.
3. `py -m venv .venv`
4. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
5. `ollama pull nomic-embed-text`
6. Ensure Ollama is running. Warm the context embedding cache once:
   `.\.venv\Scripts\python.exe kitty_ai.py --warm-cache`
7. Run searches:
   `.\.venv\Scripts\python.exe kitty_ai.py "I hate her"`
   `.\.venv\Scripts\python.exe kitty_ai.py "I will be right there"`
8. Run tests: `.\.venv\Scripts\python.exe -m pytest -q tests`
9. Run labeled evaluation: `.\.venv\Scripts\python.exe evaluate_context.py TEST_QUERIES.csv`

## Telegram bot (optional)
The bot code is unchanged. Put your own private Telegram bot token in `token.txt` locally; never commit or share it. Run `python message_model.py` on a supported desktop environment. Telegram displays the first three matches.

## Streamlit metadata editor (optional)
Run `streamlit run app.py` to review/edit contexts. After editing a context, its cached vector is invalidated automatically on the next search.

## Caveats and acceptance testing
- This package was tested offline for code behavior. Actual embedding rankings **require a running Ollama server** and cannot be certified from the included unit tests alone.
- The two expected IDs in `TEST_QUERIES.csv` are **provisional metadata-based candidates**, not client-confirmed video identities. Visually verify the diary/racing clips and correct the IDs before treating them as ground truth.
- The research paper is deliberately deferred pending actual model runs, confirmed labels, and user-provided screenshots.
- The database stores Windows-style paths; `kitty_ai.py` resolves them portably. The original Streamlit code may still expect Windows paths.
