import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kitty_ai


class ContextSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / 'test.db')
        with sqlite3.connect(self.db) as conn:
            conn.execute('''CREATE TABLE media_items (id INTEGER PRIMARY KEY,
                file_path TEXT, file_name TEXT, media_type TEXT,
                example_context TEXT, context_embedding TEXT DEFAULT '', favorite INTEGER DEFAULT 0)''')
            conn.executemany('''INSERT INTO media_items
                (id,file_path,file_name,media_type,example_context) VALUES (?,?,?,?,?)''', [
                (1, 'diary.mp4', 'diary.mp4', 'video', '"I hate her", "writing in my diary"'),
                (2, 'racing.mp4', 'racing.mp4', 'video', json.dumps(['I will be right there', 'rushing over'])),
                (3, 'other.mp4', 'other.mp4', 'video', 'unrelated situation'),
                (4, 'empty.mp4', 'empty.mp4', 'video', ''),
            ])
        self.db_patch = patch.object(kitty_ai, 'DB_PATH', self.db)
        self.db_patch.start()
        self.calls = []
        self.vectors = {'I hate her': [1., 0., 0.], 'writing in my diary': [0.9, 0.1, 0.],
                        'I will be right there': [0., 1., 0.], 'rushing over': [0.1, 0.9, 0.],
                        'unrelated situation': [0., 0., 1.]}
        def embed(text):
            self.calls.append(text)
            return self.vectors[text]
        self.embed_patch = patch.object(kitty_ai, 'get_embedding', side_effect=embed)
        self.embed_patch.start()

    def tearDown(self):
        self.embed_patch.stop()
        self.db_patch.stop()
        self.tmp.cleanup()

    def test_diary_top(self):
        self.assertEqual(kitty_ai.find_media('I hate her')[0]['file_name'], 'diary.mp4')

    def test_racing_top(self):
        self.assertEqual(kitty_ai.find_media('I will be right there')[0]['file_name'], 'racing.mp4')

    def test_cache_reuse_and_context_edit_invalidation(self):
        kitty_ai.find_media('I hate her')
        self.calls.clear()
        kitty_ai.find_media('I hate her')
        self.assertEqual(self.calls, ['I hate her'])
        with sqlite3.connect(self.db) as conn:
            conn.execute('UPDATE media_items SET example_context = ? WHERE id = 3', ('rushing over',))
        self.calls.clear()
        kitty_ai.find_media('I hate her')
        self.assertEqual(self.calls, ['I hate her', 'rushing over'])

    def test_empty_and_limit(self):
        self.assertEqual(kitty_ai.find_media('  '), [])
        self.assertEqual(len(kitty_ai.find_media('I hate her', limit=1)), 1)

    def test_context_formats(self):
        self.assertEqual(kitty_ai.parse_contexts('"hello", "world"'), ['hello', 'world'])
        self.assertEqual(kitty_ai.parse_contexts('["hello", "world"]'), ['hello', 'world'])
        self.assertEqual(kitty_ai.parse_contexts('hello\nworld'), ['hello', 'world'])

    def test_cosine_invalid_dimension(self):
        self.assertEqual(kitty_ai.cosine_similarity([1, 2], [1]), 0)


if __name__ == '__main__':
    unittest.main()
