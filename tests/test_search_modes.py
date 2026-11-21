import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import kitty_ai


class ContextSearchModeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "catalog.db"
        with closing(sqlite3.connect(self.database)) as conn, conn:
            conn.execute(
                """CREATE TABLE media_items (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT,
                    file_name TEXT,
                    media_type TEXT,
                    example_context TEXT,
                    context_embedding TEXT DEFAULT '',
                    favorite INTEGER DEFAULT 0
                )"""
            )
            conn.executemany(
                """INSERT INTO media_items
                   (id, file_path, file_name, media_type, example_context)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        1,
                        "one.jpg",
                        "one.jpg",
                        "image",
                        json.dumps(["needle", "unrelated"]),
                    ),
                    (
                        2,
                        "two.jpg",
                        "two.jpg",
                        "image",
                        json.dumps(["near match", "other detail"]),
                    ),
                ],
            )
        self.db_patch = patch.object(kitty_ai, "DB_PATH", str(self.database))
        self.db_patch.start()
        self.embeddings = {
            "query": [1.0, 0.0],
            "needle": [1.0, 0.0],
            "unrelated": [0.0, 1.0],
            "near match": [0.8, 0.6],
            "other detail": [0.6, 0.8],
            "needle unrelated": [0.4, 0.916515],
            "near match other detail": [1.0, 0.0],
        }
        self.embedding_patch = patch.object(
            kitty_ai, "get_embedding", side_effect=self.embeddings.__getitem__
        )
        self.embedding_patch.start()

    def tearDown(self):
        self.embedding_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_individual_and_whole_context_modes_can_rank_differently(self):
        individual = kitty_ai.find_media("query", limit=2, context_mode="individual")
        whole = kitty_ai.find_media("query", limit=2, context_mode="whole")

        self.assertEqual(individual[0]["id"], 1)
        self.assertEqual(individual[0]["score"], 1.0)
        self.assertEqual(whole[0]["id"], 2)
        self.assertEqual(whole[0]["matched_context"], "near match other detail")

    def test_invalid_context_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            kitty_ai.find_media("query", context_mode="unknown")


if __name__ == "__main__":
    unittest.main()
