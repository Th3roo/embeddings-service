import unittest
import os
import sys
import shutil

# Add project root to sys.path to allow importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.text_embedder import TextEmbedder

class TestTextEmbedder(unittest.TestCase):
    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2" # Default model for TextEmbedder
    # The dimension for all-MiniLM-L6-v2 is 384.
    # We can fetch this dynamically or hardcode if we are sure about the default model.
    # For this test, we'll check if it's a positive integer,
    # as the main goal is testing loading and caching.
    EXPECTED_DIMENSION_TEXT = 384 
    CACHE_DIR = "./test_cache/text_models_cache"

    def setUp(self):
        # Ensure cache directory exists and is empty before each test that uses it
        if os.path.exists(self.CACHE_DIR):
            shutil.rmtree(self.CACHE_DIR)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def tearDown(self):
        # Clean up cache directory after tests
        if os.path.exists(self.CACHE_DIR):
            shutil.rmtree(self.CACHE_DIR)

    def test_text_model_loading(self):
        # Test with default model name, specify cache
        embedder = TextEmbedder(model_name=self.DEFAULT_MODEL_NAME, model_cache_dir=self.CACHE_DIR)
        self.assertIsNotNone(embedder.model, "Text model should be loaded.")
        self.assertEqual(embedder.dimension, self.EXPECTED_DIMENSION_TEXT,
                         f"Dimension should be {self.EXPECTED_DIMENSION_TEXT} for {self.DEFAULT_MODEL_NAME}")
        self.assertTrue(embedder.dimension > 0, "Text model dimension should be a positive integer.")

    def test_model_caching_for_text_embedder(self):
        # Instantiate embedder, which should download and cache the model
        TextEmbedder(model_name=self.DEFAULT_MODEL_NAME, model_cache_dir=self.CACHE_DIR)
        
        # Check if the cache directory contains the model files.
        # Sentence-transformers typically creates a folder with the model name directly
        # (or a slightly modified version, e.g., replacing '/' with '_').
        # For "all-MiniLM-L6-v2", it should create a folder like "sentence-transformers_all-MiniLM-L6-v2".
        
        self.assertTrue(os.path.exists(self.CACHE_DIR), "Cache directory should exist.")
        self.assertTrue(len(os.listdir(self.CACHE_DIR)) > 0, "Cache directory should not be empty after model load.")
        
        # Construct the expected model directory name based on SentenceTransformer's convention.
        # It often prepends 'sentence-transformers_' and replaces slashes.
        # However, for "all-MiniLM-L6-v2", it might just be the model name or a slight variation.
        # A common pattern is 'user_model-name' or 'model-name' if no user.
        # Let's check for a directory that contains the model name.
        
        expected_model_path_segment = self.DEFAULT_MODEL_NAME.replace("/", "_") # e.g., "all-MiniLM-L6-v2"
        model_found_in_cache = False
        for item in os.listdir(self.CACHE_DIR):
            # Sentence-transformers sometimes creates subdirs like:
            # 'sentence-transformers_all-MiniLM-L6-v2' or just 'all-MiniLM-L6-v2'
            # We will check if a directory corresponding to the model name exists
            # and contains files.
            if expected_model_path_segment in item and os.path.isdir(os.path.join(self.CACHE_DIR, item)):
                model_specific_dir = os.path.join(self.CACHE_DIR, item)
                self.assertTrue(len(os.listdir(model_specific_dir)) > 0,
                                f"Model directory '{item}' in cache should not be empty.")
                model_found_in_cache = True
                break
        
        self.assertTrue(model_found_in_cache, 
                        f"Cache directory should contain a folder related to '{expected_model_path_segment}'.")

if __name__ == '__main__':
    unittest.main()
