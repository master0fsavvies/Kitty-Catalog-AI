import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate_tests


class CustomQueryMetricTests(unittest.TestCase):
    def test_custom_query_metric_and_classification_are_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_db = Path(temp_dir) / "results.db"
            candidate_results = [
                {
                    "id": 8,
                    "file_name": "holiday-cat.jpg",
                    "score": 0.87,
                    "matched_context": "A holiday greeting",
                    "file_path": "ignored-path",
                }
            ]
            query_id = evaluate_tests.make_custom_query_id(
                "  A holiday greeting  ",
                "individual",
            )
            with patch.object(evaluate_tests, "RESPONSE_DB_PATH", results_db):
                evaluate_tests.save_custom_query_metric(
                    query_id,
                    "A holiday greeting",
                    "individual",
                    candidate_results,
                )
                evaluate_tests.save_custom_query_response(query_id, "some")
                metrics = evaluate_tests.load_custom_query_metrics()

        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["query_id"], query_id)
        self.assertEqual(metrics[0]["query_text"], "A holiday greeting")
        self.assertEqual(metrics[0]["context_mode"], "individual")
        self.assertEqual(metrics[0]["response"], "some")
        self.assertEqual(
            metrics[0]["candidates"],
            [
                {
                    "id": 8,
                    "file_name": "holiday-cat.jpg",
                    "score": 0.87,
                    "matched_context": "A holiday greeting",
                }
            ],
        )

    def test_same_query_updates_candidate_metrics_without_losing_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_db = Path(temp_dir) / "results.db"
            query_id = evaluate_tests.make_custom_query_id("prompt", "whole")
            with patch.object(evaluate_tests, "RESPONSE_DB_PATH", results_db):
                evaluate_tests.save_custom_query_metric(
                    query_id, "prompt", "whole", []
                )
                evaluate_tests.save_custom_query_response(query_id, "bad context")
                evaluate_tests.save_custom_query_metric(
                    query_id,
                    "prompt",
                    "whole",
                    [
                        {
                            "id": 9,
                            "file_name": "new-result.jpg",
                            "score": 0.6,
                        }
                    ],
                )
                metrics = evaluate_tests.load_custom_query_metrics()

        self.assertEqual(metrics[0]["response"], "bad context")
        self.assertEqual(metrics[0]["candidates"][0]["id"], 9)

    def test_clear_all_removes_custom_query_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_db = Path(temp_dir) / "results.db"
            query_id = evaluate_tests.make_custom_query_id("prompt", "individual")
            with patch.object(evaluate_tests, "RESPONSE_DB_PATH", results_db):
                evaluate_tests.save_custom_query_metric(query_id, "prompt", "individual", [])
                evaluate_tests.clear_all_responses()
                metrics = evaluate_tests.load_custom_query_metrics()

        self.assertEqual(metrics, [])


if __name__ == "__main__":
    unittest.main()
