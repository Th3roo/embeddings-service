import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.text_embedder import TextEmbedder


class TestTextEmbedder(unittest.TestCase):
    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
    EXPECTED_DIMENSION_TEXT = 384
    CACHE_DIR = "./test_cache/text_models_cache"

    def setUp(self):
        if os.path.exists(self.CACHE_DIR):
            shutil.rmtree(self.CACHE_DIR)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.CACHE_DIR):
            shutil.rmtree(self.CACHE_DIR)

    def test_text_model_loading(self):
        embedder = TextEmbedder(
            model_name=self.DEFAULT_MODEL_NAME, model_cache_dir=self.CACHE_DIR
        )
        self.assertIsNotNone(embedder.model, "Text model should be loaded.")
        self.assertEqual(
            embedder.dimension,
            self.EXPECTED_DIMENSION_TEXT,
            f"Dimension should be {self.EXPECTED_DIMENSION_TEXT} for {self.DEFAULT_MODEL_NAME}",
        )
        self.assertTrue(
            embedder.dimension > 0, "Text model dimension should be a positive integer."
        )

    def test_model_caching_for_text_embedder(self):
        TextEmbedder(model_name=self.DEFAULT_MODEL_NAME, model_cache_dir=self.CACHE_DIR)

        self.assertTrue(os.path.exists(self.CACHE_DIR), "Cache directory should exist.")
        self.assertTrue(
            len(os.listdir(self.CACHE_DIR)) > 0,
            "Cache directory should not be empty after model load.",
        )

        expected_model_path_segment = self.DEFAULT_MODEL_NAME.replace("/", "_")
        model_found_in_cache = False
        for item in os.listdir(self.CACHE_DIR):
            if expected_model_path_segment in item and os.path.isdir(
                os.path.join(self.CACHE_DIR, item)
            ):
                model_specific_dir = os.path.join(self.CACHE_DIR, item)
                self.assertTrue(
                    len(os.listdir(model_specific_dir)) > 0,
                    f"Model directory '{item}' in cache should not be empty.",
                )
                model_found_in_cache = True
                break

        self.assertTrue(
            model_found_in_cache,
            f"Cache directory should contain a folder related to '{expected_model_path_segment}'.",
        )


if __name__ == "__main__":
    unittest.main()
