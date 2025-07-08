import unittest
import os
import sys
import shutil
from PIL import Image
from io import BytesIO

# Add project root to sys.path to allow importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.image_embedder import ImageEmbedder


class TestImageEmbedder(unittest.TestCase):
    DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"
    EXPECTED_DIMENSION = 768
    CACHE_DIR = "./test_cache/image_models_cache"

    def setUp(self):
        # Ensure cache directory exists and is empty before each test that uses it
        if os.path.exists(self.CACHE_DIR):
            shutil.rmtree(self.CACHE_DIR)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def tearDown(self):
        # Clean up cache directory after tests
        if os.path.exists(self.CACHE_DIR):
            shutil.rmtree(self.CACHE_DIR)

    def test_image_model_loading(self):
        print(f"Current working directory: {os.getcwd()}")
        # Test with default model name, specify cache to avoid polluting default HF cache
        embedder = ImageEmbedder(
            model_name=self.DEFAULT_MODEL_NAME, model_cache_dir=self.CACHE_DIR
        )
        self.assertIsNotNone(embedder.model, "Model should be loaded.")
        self.assertIsNotNone(embedder.processor, "Processor should be loaded.")
        self.assertEqual(
            embedder.dimension,
            self.EXPECTED_DIMENSION,
            f"Dimension should be {self.EXPECTED_DIMENSION} for {self.DEFAULT_MODEL_NAME}",
        )

    def test_image_embedding_generation(self):
        # Test with default model name, specify cache
        embedder = ImageEmbedder(
            model_name=self.DEFAULT_MODEL_NAME, model_cache_dir=self.CACHE_DIR
        )

        # Create a dummy image
        dummy_pil_image = Image.new("RGB", (224, 224), color="red")

        # Convert to bytes
        img_byte_arr = BytesIO()
        dummy_pil_image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        embedding = embedder.get_embedding(image_bytes)

        self.assertIsInstance(embedding, list, "Embedding should be a list.")
        self.assertTrue(
            all(isinstance(x, float) for x in embedding),
            "All elements in embedding should be floats.",
        )
        self.assertEqual(
            len(embedding),
            self.EXPECTED_DIMENSION,
            f"Embedding length should be equal to dimension {self.EXPECTED_DIMENSION}.",
        )

    def test_model_caching_for_image_embedder(self):
        # Instantiate embedder, which should download and cache the model
        ImageEmbedder(
            model_name=self.DEFAULT_MODEL_NAME, model_cache_dir=self.CACHE_DIR
        )

        # Check if the cache directory is no longer empty or contains expected model files/folders
        # Hugging Face model cache structure can be complex (e.g. refs, blobs, snapshots).
        # A common pattern is a 'snapshots' directory containing subdirectories for models.
        # We'll check for non-emptiness and a 'snapshots' dir as a proxy.

        self.assertTrue(os.path.exists(self.CACHE_DIR), "Cache directory should exist.")
        self.assertTrue(
            len(os.listdir(self.CACHE_DIR)) > 0,
            "Cache directory should not be empty after model load.",
        )

        # More specific check: Hugging Face often creates a 'snapshots' folder
        # and then a subfolder with the model's commit hash.
        # For "google/vit-base-patch16-224", there isn't one specific file, but a directory.
        # Let's check for *any* subdirectory within snapshots, as hash names are unpredictable.
        snapshots_path = os.path.join(
            self.CACHE_DIR, "models--" + self.DEFAULT_MODEL_NAME.replace("/", "--")
        )
        # Alternative check for general Hugging Face structure
        # Check for any files in common locations like 'blobs' or 'snapshots'
        # This test assumes that *some* files will be downloaded directly into CACHE_DIR or subdirs like 'snapshots'
        # or a directory named like 'models--google--vit-base-patch16-224'.

        # A simple check that *something* related to the model is in the cache dir.
        # The exact structure can vary based on transformers version.
        # We look for a directory that starts with 'models--' followed by the model name.
        expected_model_dir_prefix = "models--" + self.DEFAULT_MODEL_NAME.replace(
            "/", "--"
        )
        found_model_dir = False
        for item in os.listdir(self.CACHE_DIR):
            if item.startswith(expected_model_dir_prefix):
                found_model_dir = True
                model_files_path = os.path.join(self.CACHE_DIR, item)
                self.assertTrue(
                    len(os.listdir(model_files_path)) > 0,
                    f"Model specific directory '{item}' should not be empty.",
                )
                break
        self.assertTrue(
            found_model_dir,
            f"Cache directory should contain a folder starting with '{expected_model_dir_prefix}'.",
        )


if __name__ == "__main__":
    unittest.main()
