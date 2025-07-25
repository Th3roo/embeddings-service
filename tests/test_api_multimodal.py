import pytest
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI application
import os
from io import BytesIO
from PIL import Image


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


API_KEY = os.getenv("TEST_API_KEY", "test_api_key_not_set")


def get_auth_headers():
    if API_KEY == "test_api_key_not_set":
        pass
    return {"X-API-Key": API_KEY}


def test_multimodal_endpoint_exists(client):
    payload = {"text": "test"}
    response = client.post(
        "/v1/multimodal/embed", data=payload, headers=get_auth_headers()
    )
    assert response.status_code != 404


def test_multimodal_embed_text(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping authenticated text embedding test.")

    payload = {"text": "Это тестовый текст для FastAPI."}
    response = client.post(
        "/v1/multimodal/embed", data=payload, headers=get_auth_headers()
    )

    assert response.status_code == 200
    json_response = response.json()
    assert (
        json_response["model_name"]
        == "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )
    assert json_response["model_type"] == "multimodal"
    assert isinstance(json_response["embedding"], list)
    assert len(json_response["embedding"]) == 512


def test_multimodal_embed_image_url(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip(
            "TEST_API_KEY not set, skipping authenticated image URL embedding test."
        )

    image_url = "https://via.placeholder.com/150/0000FF/FFFFFF?Text=TestAPIImage"
    payload = {"image_url": image_url}

    try:
        response = client.post(
            "/v1/multimodal/embed", data=payload, headers=get_auth_headers()
        )
    except Exception as e:
        pytest.skip(
            f"Skipping image URL test due to network issue or placeholder not available: {e}"
        )
        return

    assert response.status_code == 200
    json_response = response.json()
    assert (
        json_response["model_name"]
        == "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )
    assert json_response["model_type"] == "multimodal"
    assert isinstance(json_response["embedding"], list)
    assert len(json_response["embedding"]) == 512


def test_multimodal_embed_image_file(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip(
            "TEST_API_KEY not set, skipping authenticated image file embedding test."
        )

    try:
        img = Image.new("RGB", (60, 30), color="green")
        byte_io = BytesIO()
        img.save(byte_io, format="PNG")
        byte_io.seek(0)
        files = {"image_file": ("test_image.png", byte_io, "image/png")}
    except ImportError:
        pytest.skip("Pillow is not installed, skipping image file upload test.")
        return

    response = client.post(
        "/v1/multimodal/embed", files=files, headers=get_auth_headers()
    )

    assert response.status_code == 200
    json_response = response.json()
    assert (
        json_response["model_name"]
        == "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )
    assert json_response["model_type"] == "multimodal"
    assert isinstance(json_response["embedding"], list)
    assert len(json_response["embedding"]) == 512


def test_multimodal_embed_missing_input(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping missing input test.")

    response = client.post("/v1/multimodal/embed", data={}, headers=get_auth_headers())
    assert response.status_code == 400
    assert "Please provide exactly one of" in response.json()["detail"]


def test_multimodal_embed_multiple_inputs(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping multiple inputs test.")

    payload = {"text": "some text", "image_url": "http://example.com/image.jpg"}
    response = client.post(
        "/v1/multimodal/embed", data=payload, headers=get_auth_headers()
    )
    assert response.status_code == 400
    assert "Please provide exactly one of" in response.json()["detail"]

    try:
        img = Image.new("RGB", (10, 10), color="red")
        byte_io = BytesIO()
        img.save(byte_io, format="PNG")
        byte_io.seek(0)
        files = {"image_file": ("test.png", byte_io, "image/png")}
        data = {"text": "some text"}
    except ImportError:
        pytest.skip("Pillow not installed, skipping part of multiple inputs test.")
        return

    response_with_file_and_text = client.post(
        "/v1/multimodal/embed", files=files, data=data, headers=get_auth_headers()
    )
    assert response_with_file_and_text.status_code == 400
    assert (
        "Please provide exactly one of" in response_with_file_and_text.json()["detail"]
    )


def test_multimodal_embed_invalid_image_url(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping invalid image URL test.")

    payload = {"image_url": "this_is_not_a_valid_url"}
    response = client.post(
        "/v1/multimodal/embed", data=payload, headers=get_auth_headers()
    )
    assert response.status_code == 422


def test_multimodal_embed_non_image_file(client):
    if API_KEY == "test_api_key_not_set":
        pytest.skip("TEST_API_KEY not set, skipping non-image file test.")

    byte_io = BytesIO(b"this is not an image")
    files = {"image_file": ("test_text.txt", byte_io, "text/plain")}

    response = client.post(
        "/v1/multimodal/embed", files=files, headers=get_auth_headers()
    )
    assert response.status_code == 400
    assert "Invalid image file type" in response.json()["detail"]


def test_multimodal_no_api_key(client):
    payload = {"text": "test"}
    response = client.post("/v1/multimodal/embed", data=payload)
    assert response.status_code == 401
