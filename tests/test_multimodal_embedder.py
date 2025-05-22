import pytest
from PIL import Image
import numpy as np
import os
import requests # Added import for requests
from io import BytesIO # Added import for BytesIO
from app.models.multimodal_embedder import MultimodalEmbedder

# Create a dummy model cache directory for testing if it doesn't exist
MODEL_CACHE_DIR = "./test_model_cache"
if not os.path.exists(MODEL_CACHE_DIR):
    os.makedirs(MODEL_CACHE_DIR)

# Define a fixture for the MultimodalEmbedder instance
@pytest.fixture(scope="module")
def embedder():
    try:
        # Attempt to load the actual model for a more thorough test.
        # If this causes issues in CI/limited environments,
        # we might need to mock SentenceTransformer.
        instance = MultimodalEmbedder(model_cache_dir=MODEL_CACHE_DIR)
        if instance.model is None: # Should be caught by MultimodalEmbedder's _load_model
             pytest.skip("Failed to load multimodal model, skipping tests that require it.")
        return instance
    except Exception as e:
        # If model loading fails (e.g. no internet, specific platform issues),
        # skip tests that require the model.
        pytest.skip(f"Skipping multimodal embedder tests due to model loading error: {e}")


def test_multimodal_embedder_init(embedder):
    assert embedder is not None
    assert embedder.model_name == "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    assert embedder.model_type == "multimodal"
    assert embedder.model is not None, "Model should be loaded."
    assert embedder.dimension > 0, "Dimension should be set and greater than 0."

def test_multimodal_embedder_dimension_property(embedder):
    dim = embedder.dimension
    assert isinstance(dim, int)
    assert dim > 0
    # For "sentence-transformers/clip-ViT-B-32-multilingual-v1", the dimension is 512
    assert dim == 512, f"Expected dimension 512, got {dim}"

def test_multimodal_text_embedding(embedder):
    test_text = "Это тестовый текст для эмбеддинга."
    embedding = embedder.get_embedding(test_text)
    
    assert isinstance(embedding, list), "Embedding should be a list."
    assert len(embedding) == embedder.dimension, f"Embedding length {len(embedding)} should match model dimension {embedder.dimension}."
    assert all(isinstance(x, float) for x in embedding), "All elements in embedding should be floats."

def test_multimodal_image_embedding(embedder):
    # Create a dummy PIL image for testing
    # (width, height), color
    try:
        img = Image.new('RGB', (60, 30), color = 'red')
    except ImportError:
        pytest.skip("Pillow is not installed, skipping image embedding test.")

    embedding = embedder.get_embedding(img)
    
    assert isinstance(embedding, list), "Image embedding should be a list."
    assert len(embedding) == embedder.dimension, f"Image embedding length {len(embedding)} should match model dimension {embedder.dimension}."
    assert all(isinstance(x, float) for x in embedding), "All elements in image embedding should be floats."

def test_multimodal_embedding_consistency(embedder):
    # Test that the same text input produces very similar embeddings (allowing for minor float variations)
    # Note: For CLIP, the embeddings are generally deterministic.
    text1 = "Hello world"
    embedding1 = np.array(embedder.get_embedding(text1))
    embedding2 = np.array(embedder.get_embedding(text1))
    
    assert np.allclose(embedding1, embedding2, atol=1e-6), "Embeddings for the same text should be very close."

    # Test with a simple image
    try:
        img1 = Image.new('RGB', (100, 100), color = 'blue')
        img2 = Image.new('RGB', (100, 100), color = 'blue') # Identical image
    except ImportError:
        pytest.skip("Pillow is not installed, skipping image consistency test.")

    embedding_img1 = np.array(embedder.get_embedding(img1))
    embedding_img2 = np.array(embedder.get_embedding(img2))

    assert np.allclose(embedding_img1, embedding_img2, atol=1e-6), "Embeddings for the same image should be very close."

def test_load_image_from_url(embedder):
    # This tests the helper function; requires internet access to a reliable image URL
    # Using a public domain placeholder image
    image_url = "https://via.placeholder.com/150/FF0000/FFFFFF?Text=TestImage" # Red square
    try:
        pil_image = embedder._load_image_from_source(image_url)
        assert isinstance(pil_image, Image.Image)
        # Further check if embedding this image works
        embedding = embedder.get_embedding(pil_image)
        assert len(embedding) == embedder.dimension
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Skipping image URL test due to network issue or placeholder not available: {e}")
    except ValueError as e: # Catch errors if image loading/processing fails
        pytest.fail(f"Image loading from URL failed: {e}")


def test_load_image_from_bytes(embedder):
    try:
        # Create a dummy image, save to bytes, then load
        img = Image.new('RGB', (50, 50), color = 'green')
        # from io import BytesIO # Already imported at the top
        byte_io = BytesIO()
        img.save(byte_io, format='PNG')
        image_bytes = byte_io.getvalue()
        
        pil_image = embedder._load_image_from_source(image_bytes)
        assert isinstance(pil_image, Image.Image)
        # Further check if embedding this image works
        embedding = embedder.get_embedding(pil_image)
        assert len(embedding) == embedder.dimension
    except ImportError: # Should catch if Pillow is not there for Image.new or save
        pytest.skip("Pillow/BytesIO not available, skipping image bytes test.")
    except ValueError as e:
        pytest.fail(f"Image loading from bytes failed: {e}")


# Test for model loading failure (more complex to set up without specific mocking)
# One way is to provide an invalid model name and expect an error during __init__
def test_multimodal_embedder_invalid_model_name():
    with pytest.raises(Exception): # Expecting some form of error, could be more specific
        # This model path is intentionally incorrect
        MultimodalEmbedder(model_name="invalid-model-path/non-existent-model", model_cache_dir=MODEL_CACHE_DIR)
