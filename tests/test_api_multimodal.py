import pytest
from fastapi.testclient import TestClient
from main import app # Import your FastAPI application
import os
from io import BytesIO
from PIL import Image

# Fixture for the TestClient
@pytest.fixture(scope="module")
def client():
    # Ensure models are preloaded if your app's lifespan context does that
    # For TestClient, lifespan events are typically handled automatically if set up in FastAPI app
    with TestClient(app) as c:
        yield c

# Retrieve a valid API key from environment variables for testing
# Store it in .env or set it in the test environment
# Example: TEST_API_KEY="your_test_api_key"
API_KEY = os.getenv("TEST_API_KEY", "test_api_key_not_set") 
# Fallback if not set, but tests requiring auth might fail or be skipped.

# Helper to create headers
def get_auth_headers():
    if API_KEY == "test_api_key_not_set":
        # This allows running tests that don't strictly need a *valid* key for basic functionality checks,
        # but they might fail if the key is actually validated against a list.
        # For this service, any non-empty string might pass basic validation if VALID_API_KEYS is empty or not strictly checked.
        # print("Warning: TEST_API_KEY environment variable not set. Using a placeholder.") # Optional warning
        pass # Use the placeholder or skip if strict validation is in place
    return {"X-API-Key": API_KEY}


# Basic test to see if the endpoint is reachable
def test_multimodal_endpoint_exists(client):
    # A simple text payload
    payload = {"text": "test"}
    response = client.post("/v1/multimodal/embed", data=payload, headers=get_auth_headers())
    # Check if it's not a 404; actual success (200) or input error (400/422) means endpoint exists
    assert response.status_code != 404


def test_multimodal_embed_text(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping authenticated text embedding test.")
    
    payload = {"text": "Это тестовый текст для FastAPI."}
    response = client.post("/v1/multimodal/embed", data=payload, headers=get_auth_headers())
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model_name"] == "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    assert json_response["model_type"] == "multimodal"
    assert isinstance(json_response["embedding"], list)
    assert len(json_response["embedding"]) == 512 # Expected dimension for this model


def test_multimodal_embed_image_url(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping authenticated image URL embedding test.")

    # Using a public domain placeholder image URL
    image_url = "https://via.placeholder.com/150/0000FF/FFFFFF?Text=TestAPIImage" # Blue square
    payload = {"image_url": image_url}
    
    try:
        response = client.post("/v1/multimodal/embed", data=payload, headers=get_auth_headers())
    except Exception as e: # Catch potential network errors during test execution if placeholder is down
        pytest.skip(f"Skipping image URL test due to network issue or placeholder not available: {e}")
        return

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model_name"] == "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    assert json_response["model_type"] == "multimodal"
    assert isinstance(json_response["embedding"], list)
    assert len(json_response["embedding"]) == 512


def test_multimodal_embed_image_file(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping authenticated image file embedding test.")

    # Create a dummy image file for upload
    try:
        img = Image.new('RGB', (60, 30), color='green')
        byte_io = BytesIO()
        img.save(byte_io, format='PNG')
        byte_io.seek(0) # Reset stream position
        files = {'image_file': ('test_image.png', byte_io, 'image/png')}
    except ImportError:
        pytest.skip("Pillow is not installed, skipping image file upload test.")
        return

    response = client.post("/v1/multimodal/embed", files=files, headers=get_auth_headers())
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model_name"] == "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    assert json_response["model_type"] == "multimodal"
    assert isinstance(json_response["embedding"], list)
    assert len(json_response["embedding"]) == 512


def test_multimodal_embed_missing_input(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping missing input test.")
        
    # No data payload
    response = client.post("/v1/multimodal/embed", data={}, headers=get_auth_headers())
    assert response.status_code == 400 # Expecting Bad Request
    assert "Please provide exactly one of" in response.json()["detail"]


def test_multimodal_embed_multiple_inputs(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping multiple inputs test.")

    payload = {"text": "some text", "image_url": "http://example.com/image.jpg"}
    response = client.post("/v1/multimodal/embed", data=payload, headers=get_auth_headers())
    assert response.status_code == 400 # Expecting Bad Request
    assert "Please provide exactly one of" in response.json()["detail"]

    # Test with text and file
    try:
        img = Image.new('RGB', (10, 10), color='red')
        byte_io = BytesIO()
        img.save(byte_io, format='PNG')
        byte_io.seek(0)
        files = {'image_file': ('test.png', byte_io, 'image/png')}
        data = {"text": "some text"} # Sending data along with files
    except ImportError:
        pytest.skip("Pillow not installed, skipping part of multiple inputs test.")
        return

    response_with_file_and_text = client.post("/v1/multimodal/embed", files=files, data=data, headers=get_auth_headers())
    assert response_with_file_and_text.status_code == 400
    assert "Please provide exactly one of" in response_with_file_and_text.json()["detail"]


def test_multimodal_embed_invalid_image_url(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping invalid image URL test.")

    payload = {"image_url": "this_is_not_a_valid_url"}
    response = client.post("/v1/multimodal/embed", data=payload, headers=get_auth_headers())
    # This should be caught by Pydantic validation for HttpUrl before it hits the main logic
    assert response.status_code == 422 # Unprocessable Entity for Pydantic validation errors on Form data

def test_multimodal_embed_non_image_file(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping non-image file test.")

    byte_io = BytesIO(b"this is not an image")
    files = {'image_file': ('test_text.txt', byte_io, 'text/plain')}

    response = client.post("/v1/multimodal/embed", files=files, headers=get_auth_headers())
    assert response.status_code == 400 # Bad request due to invalid image content type
    assert "Invalid image file type" in response.json()["detail"]


def test_multimodal_no_api_key(client):
    payload = {"text": "test"}
    response = client.post("/v1/multimodal/embed", data=payload) # No headers
    assert response.status_code == 401 # Unauthorized or however your get_api_key handles it
