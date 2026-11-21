import csv
import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import kitty_ai


CATALOG_DB_PATH = kitty_ai.DB_PATH
RESPONSE_DB_PATH = TESTS_DIR / "test_results.db"
TESTS_CSV = TESTS_DIR / "test.csv"
ANSWERS = (
    "some",
    "all make sense",
    "none make sense",
    "bad context",
)
CONTEXT_SEARCH_MODES = {
    "Best match across individual contexts": "individual",
    "Match against full context string": "whole",
}
REQUIRED_COLUMNS = {
    "test_id",
    "expected_id",
    "expected_file_path",
    "expected_file_name",
    "context",
}


def load_test_cases(csv_path=TESTS_CSV):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Test CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        tests = []
        seen_ids = set()
        for line_number, row in enumerate(reader, start=2):
            test_id = (row["test_id"] or "").strip()
            context = (row["context"] or "").strip()
            if not test_id or not context:
                raise ValueError(
                    f"Test CSV row {line_number} needs a test_id and context."
                )
            if test_id in seen_ids:
                raise ValueError(f"Duplicate test_id on CSV row {line_number}: {test_id}")
            seen_ids.add(test_id)
            try:
                expected_id = int(row["expected_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid expected_id on CSV row {line_number}."
                ) from exc
            tests.append(
                {
                    "test_id": test_id,
                    "expected_id": expected_id,
                    "expected_file_path": kitty_ai.resolve_media_path(
                        row["expected_file_path"]
                    ),
                    "expected_file_name": row["expected_file_name"],
                    "context": context,
                }
            )
    if not tests:
        raise ValueError("The test CSV contains no test cases.")
    return tests


def ensure_response_table(conn):
    schema = conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name = 'context_test_responses'"
    ).fetchone()
    if schema and "'some'" in schema[0]:
        return

    if schema:
        conn.execute("ALTER TABLE context_test_responses RENAME TO context_test_responses_old")

    conn.execute(
        """
        CREATE TABLE context_test_responses (
            test_id TEXT PRIMARY KEY,
            response TEXT NOT NULL CHECK (
                response IN (
                    'some',
                    'all make sense',
                    'none make sense',
                    'bad context'
                )
            ),
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    if schema:
        conn.execute(
            """
            INSERT INTO context_test_responses (test_id, response, updated_at)
            SELECT test_id,
                   CASE response
                       WHEN 'one makes sense' THEN 'some'
                       ELSE response
                   END,
                   updated_at
            FROM context_test_responses_old
            """
        )
        conn.execute("DROP TABLE context_test_responses_old")


def ensure_custom_query_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS custom_query_metrics (
            query_id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            context_mode TEXT NOT NULL,
            candidates_json TEXT NOT NULL DEFAULT '[]',
            response TEXT CHECK (
                response IS NULL OR response IN (
                    'some',
                    'all make sense',
                    'none make sense',
                    'bad context'
                )
            ),
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def make_custom_query_id(query_text, context_mode):
    normalized_query = " ".join(query_text.casefold().split())
    digest = hashlib.sha256(
        f"{context_mode}\0{normalized_query}".encode("utf-8")
    ).hexdigest()
    return f"custom-{digest}"


def save_custom_query_metric(query_id, query_text, context_mode, candidates):
    stored_candidates = [
        {
            "id": candidate["id"],
            "file_name": candidate["file_name"],
            "score": candidate["score"],
            "matched_context": candidate.get("matched_context", ""),
        }
        for candidate in candidates
    ]
    with closing(sqlite3.connect(RESPONSE_DB_PATH, timeout=30)) as conn:
        with conn:
            ensure_custom_query_table(conn)
            conn.execute(
                """
                INSERT INTO custom_query_metrics
                    (query_id, query_text, context_mode, candidates_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_id) DO UPDATE SET
                    query_text = excluded.query_text,
                    context_mode = excluded.context_mode,
                    candidates_json = excluded.candidates_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    query_id,
                    query_text,
                    context_mode,
                    json.dumps(stored_candidates),
                ),
            )


def save_custom_query_response(query_id, response):
    if response not in ANSWERS:
        raise ValueError(f"Unsupported test response: {response}")
    with closing(sqlite3.connect(RESPONSE_DB_PATH, timeout=30)) as conn:
        with conn:
            ensure_custom_query_table(conn)
            cursor = conn.execute(
                """
                UPDATE custom_query_metrics
                SET response = ?, updated_at = CURRENT_TIMESTAMP
                WHERE query_id = ?
                """,
                (response, query_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("Search this custom query before classifying it.")


def load_custom_query_metrics():
    with closing(sqlite3.connect(RESPONSE_DB_PATH, timeout=30)) as conn:
        with conn:
            ensure_custom_query_table(conn)
            rows = conn.execute(
                """
                SELECT query_id, query_text, context_mode, candidates_json,
                       response, updated_at
                FROM custom_query_metrics
                ORDER BY updated_at DESC
                """
            ).fetchall()
    return [
        {
            "query_id": query_id,
            "query_text": query_text,
            "context_mode": context_mode,
            "candidates": json.loads(candidates_json),
            "response": response,
            "updated_at": updated_at,
        }
        for query_id, query_text, context_mode, candidates_json, response, updated_at in rows
    ]


def migrate_legacy_responses():
    with closing(sqlite3.connect(RESPONSE_DB_PATH, timeout=30)) as destination:
        with destination:
            ensure_response_table(destination)
            with closing(sqlite3.connect(CATALOG_DB_PATH, timeout=30)) as source:
                exists = source.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'context_test_responses'"
                ).fetchone()
                if not exists:
                    return
                rows = source.execute(
                    "SELECT test_id, response, updated_at "
                    "FROM context_test_responses"
                ).fetchall()
            destination.executemany(
                """
                INSERT OR IGNORE INTO context_test_responses
                    (test_id, response, updated_at)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        test_id,
                        "some" if response == "one makes sense" else response,
                        updated_at,
                    )
                    for test_id, response, updated_at in rows
                ],
            )

    with closing(sqlite3.connect(CATALOG_DB_PATH, timeout=30)) as source:
        with source:
            source.execute("DROP TABLE context_test_responses")


def save_response(test_id, response):
    if response not in ANSWERS:
        raise ValueError(f"Unsupported test response: {response}")
    with closing(sqlite3.connect(RESPONSE_DB_PATH, timeout=30)) as conn:
        with conn:
            ensure_response_table(conn)
            conn.execute(
                """
                INSERT INTO context_test_responses (test_id, response)
                VALUES (?, ?)
                ON CONFLICT(test_id) DO UPDATE SET
                    response = excluded.response,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (test_id, response),
            )


def clear_all_responses():
    with closing(sqlite3.connect(RESPONSE_DB_PATH, timeout=30)) as conn:
        with conn:
            ensure_response_table(conn)
            ensure_custom_query_table(conn)
            conn.execute("DELETE FROM context_test_responses")
            conn.execute("DELETE FROM custom_query_metrics")


def clear_all_tests(test_ids):
    import streamlit as st

    clear_all_responses()
    for test_id in test_ids:
        answer_key = f"answer_{test_id}"
        if answer_key in st.session_state:
            del st.session_state[answer_key]
    for key in (
        "custom_query_results",
        "custom_query_id",
        "custom_query_input",
    ):
        if key in st.session_state:
            del st.session_state[key]
    for key in list(st.session_state):
        if key.startswith("custom_answer_"):
            del st.session_state[key]
    st.session_state.test_index = 0


def autosave_response(test_id, answer_key):
    import streamlit as st

    response = st.session_state[answer_key]
    if response in ANSWERS:
        save_response(test_id, response)


def autosave_custom_query_response(query_id, answer_key):
    import streamlit as st

    response = st.session_state[answer_key]
    if response in ANSWERS:
        save_custom_query_response(query_id, response)


def load_responses(test_ids):
    if not test_ids:
        return {}
    with closing(sqlite3.connect(RESPONSE_DB_PATH, timeout=30)) as conn:
        with conn:
            ensure_response_table(conn)
            saved = dict(
                conn.execute(
                    "SELECT test_id, response FROM context_test_responses"
                ).fetchall()
            )
    test_id_set = set(test_ids)
    return {test_id: response for test_id, response in saved.items() if test_id in test_id_set}


def load_context_metadata(media_id):
    with closing(sqlite3.connect(CATALOG_DB_PATH, timeout=30)) as conn:
        row = conn.execute(
            "SELECT example_context FROM media_items WHERE id = ?",
            (media_id,),
        ).fetchone()
    if row is None:
        return []
    return kitty_ai.split_context_sentences(kitty_ai.parse_contexts(row[0]))


def show_context_metadata(media_id):
    import streamlit as st

    try:
        contexts = load_context_metadata(media_id)
    except sqlite3.Error as exc:
        st.error(f"Could not load context metadata: {exc}")
        return
    metadata = " | ".join(contexts) if contexts else "No context metadata."
    st.caption(f"Context metadata: {metadata}")


def show_media(media):
    import streamlit as st

    path = media["file_path"]
    if not Path(path).is_file():
        st.warning(f"Media file not found: {path}")
    elif media["media_type"] == "image":
        st.image(path, width="stretch")
    elif media["media_type"] == "video":
        st.video(path)
    else:
        st.warning(f"Unsupported media type: {media['media_type']}")


def render_custom_query(st, custom_metrics):
    st.subheader("Custom query")
    with st.form("custom_query_form"):
        query_text = st.text_input(
            "Enter a custom prompt",
            key="custom_query_input",
            placeholder="e.g. When you need a little encouragement",
        )
        mode_label = st.selectbox(
            "Context matching mode",
            tuple(CONTEXT_SEARCH_MODES),
            key="custom_query_context_mode",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        normalized_query = query_text.strip()
        if not normalized_query:
            st.warning("Enter a prompt before searching.")
            return
        context_mode = CONTEXT_SEARCH_MODES[mode_label]
        try:
            candidates = kitty_ai.find_media(
                normalized_query,
                limit=3,
                context_mode=context_mode,
            )
            query_id = make_custom_query_id(normalized_query, context_mode)
            save_custom_query_metric(
                query_id,
                normalized_query,
                context_mode,
                candidates,
            )
            custom_metrics = load_custom_query_metrics()
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
            st.error(f"Could not run custom query: {exc}")
            return
        st.session_state.custom_query_id = query_id
        st.session_state.custom_query_results = {
            "query_id": query_id,
            "query_text": normalized_query,
            "context_mode": context_mode,
            "candidates": candidates,
        }

    result = st.session_state.get("custom_query_results")
    if result is None:
        st.info("Enter a custom prompt and select Search to evaluate it.")
        return

    query_id = result["query_id"]
    query_text = result["query_text"]
    context_mode = result["context_mode"]
    candidates = result["candidates"]
    saved = next(
        (metric for metric in custom_metrics if metric["query_id"] == query_id),
        None,
    )
    saved_response = saved["response"] if saved else None
    st.markdown(f"**Prompt:** {query_text}")
    st.caption(f"Matching mode: {context_mode}")

    if not candidates:
        st.info("No candidate media was returned for this prompt.")
    else:
        columns = st.columns(len(candidates))
        for rank, (column, candidate) in enumerate(
            zip(columns, candidates), start=1
        ):
            with column:
                st.markdown(f"**#{rank} · ID {candidate['id']}**")
                st.caption(
                    f"{candidate['file_name']} · Score {candidate['score']:.4f}"
                )
                show_media(candidate)
                show_context_metadata(candidate["id"])
                st.caption(candidate.get("matched_context", ""))

    answer_key = f"custom_answer_{query_id}"
    if st.session_state.get(answer_key) not in ANSWERS:
        st.session_state[answer_key] = saved_response
    st.radio(
        "Classify these results",
        ANSWERS,
        key=answer_key,
        on_change=autosave_custom_query_response,
        args=(query_id, answer_key),
    )

    st.subheader("Custom query metrics")
    st.caption(
        f"Saved queries: {len(custom_metrics)} · "
        f"Classified: {sum(metric['response'] is not None for metric in custom_metrics)}"
    )
    st.caption(
        "This query's saved classification: "
        f"{saved_response or 'not classified yet'}"
    )


def main():
    import streamlit as st

    st.set_page_config(page_title="Kitty Search Test Evaluation", layout="wide")
    st.title("Kitty Search Test Evaluation")
    try:
        tests = load_test_cases()
    except (OSError, ValueError) as exc:
        st.error(f"Could not load {TESTS_CSV}: {exc}")
        st.info("Generate the test CSV first with `python tests\\generate_tests.py`.")
        st.stop()

    test_ids = [test["test_id"] for test in tests]
    try:
        migrate_legacy_responses()
        responses = load_responses(test_ids)
        custom_metrics = load_custom_query_metrics()
    except sqlite3.Error as exc:
        st.error(f"Could not load saved classifications: {exc}")
        st.stop()

    evaluation_type = st.radio(
        "Evaluation type",
        ("Generated test", "Custom query"),
        horizontal=True,
        key="evaluation_type",
    )
    if evaluation_type == "Custom query":
        render_custom_query(st, custom_metrics)
        all_responses = list(responses.values()) + [
            metric["response"]
            for metric in custom_metrics
            if metric["response"] is not None
        ]
        counts = Counter(all_responses)
        st.divider()
        st.subheader("Saved totals (generated tests and custom queries)")
        metric_columns = st.columns(len(ANSWERS))
        for column, answer in zip(metric_columns, ANSWERS):
            column.metric(answer.title(), counts[answer])
        st.button(
            "Clear All Tests",
            width="stretch",
            on_click=clear_all_tests,
            args=(test_ids,),
        )
        return

    if "test_order" not in st.session_state or set(st.session_state.test_order) != set(test_ids):
        shuffled_ids = test_ids.copy()
        random.shuffle(shuffled_ids)
        st.session_state.test_order = shuffled_ids
    tests_by_id = {test["test_id"]: test for test in tests}
    tests = [tests_by_id[test_id] for test_id in st.session_state.test_order]

    if "test_index" not in st.session_state:
        first_unanswered = next(
            (index for index, test in enumerate(tests) if test["test_id"] not in responses),
            0,
        )
        st.session_state.test_index = first_unanswered
    st.session_state.test_index = min(
        max(int(st.session_state.test_index), 0), len(tests) - 1
    )

    index = st.session_state.test_index
    test = tests[index]
    current_answer = responses.get(test["test_id"])
    answer_key = f"answer_{test['test_id']}"
    if st.session_state.get(answer_key) == "one makes sense":
        st.session_state[answer_key] = "some"
    counts = Counter(responses.values())
    counts.update(
        metric["response"]
        for metric in custom_metrics
        if metric["response"] is not None
    )

    left, right = st.columns([3, 1])
    with right:
        search_mode_label = st.selectbox(
            "Context matching mode",
            tuple(CONTEXT_SEARCH_MODES),
            key="context_matching_mode",
        )
    context_mode = CONTEXT_SEARCH_MODES[search_mode_label]

    try:
        cache_key = f"{test['test_id']}:{context_mode}"
        if cache_key not in st.session_state.get("candidate_cache", {}):
            candidates = kitty_ai.find_media(
                test["context"],
                limit=3,
                context_mode=context_mode,
            )
            cache = dict(st.session_state.get("candidate_cache", {}))
            cache[cache_key] = candidates
            st.session_state.candidate_cache = cache
        candidates = st.session_state.candidate_cache[cache_key]
        search_error = None
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        candidates = []
        search_error = str(exc)

    with left:
        image_columns = st.columns(4)
        with image_columns[0]:
            st.markdown("**Expected photo**")
            st.caption(f"ID {test['expected_id']} · {test['expected_file_name']}")
            show_media(
                {
                    "file_path": test["expected_file_path"],
                    "media_type": "image",
                }
            )
            show_context_metadata(test["expected_id"])
        for rank, column in enumerate(image_columns[1:], start=1):
            with column:
                if rank <= len(candidates):
                    candidate = candidates[rank - 1]
                    st.markdown(f"**#{rank} · ID {candidate['id']}**")
                    st.caption(candidate["file_name"])
                    show_media(candidate)
                    show_context_metadata(candidate["id"])
                elif not search_error:
                    st.caption(f"#{rank} · No result")
        if search_error:
            st.error(f"Could not search for candidates: {search_error}")
        elif not candidates:
            st.info("No candidate media was returned for this context.")

    with right:
        st.subheader(f"Test {index + 1} of {len(tests)}")
        st.markdown("**Generated context**")
        st.info(test["context"])
        st.caption("Classify how well the returned results answer this context.")

        st.radio(
            "Classification",
            ANSWERS,
            index=ANSWERS.index(current_answer) if current_answer in ANSWERS else None,
            key=answer_key,
            on_change=autosave_response,
            args=(test["test_id"], answer_key),
        )

        navigation_left, navigation_right = st.columns(2)
        with navigation_left:
            if st.button("Previous", disabled=index == 0, width="stretch"):
                st.session_state.test_index = index - 1
                st.rerun()
        with navigation_right:
            if st.button(
                "Next",
                disabled=index == len(tests) - 1,
                width="stretch",
            ):
                st.session_state.test_index = index + 1
                st.rerun()

        st.divider()
        st.subheader("Saved totals")
        for answer in ANSWERS:
            st.metric(answer.title(), counts[answer])
        st.caption(f"{len(responses)} of {len(tests)} tests classified.")
        st.caption(
            f"{sum(metric['response'] is not None for metric in custom_metrics)} "
            "custom queries classified."
        )
        st.caption(f"Responses are saved in {RESPONSE_DB_PATH.name}.")
        st.button(
            "Clear All Tests",
            width="stretch",
            on_click=clear_all_tests,
            args=(test_ids,),
        )


if __name__ == "__main__":
    main()
