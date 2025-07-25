# Embeddings Service

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-blue)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple and efficient FastAPI service for generating text, image, and multimodal embeddings.

## Getting Started

### Prerequisites

*   Docker
*   Docker Compose

### Running the Service

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd embeddings-service
    ```

2.  **Set up your environment:**
    Copy the example environment file and add your API keys.
    ```bash
    cp example.env .env
    ```

3.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

The API will be available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

## API

The service provides the following main endpoints:

*   `POST /v1/embeddings/text`: Get embeddings for a string of text.
*   `POST /v1/embeddings/image/upload`: Upload an image file to get its embedding.
*   `POST /v1/embeddings/image/url`: Get the embedding of an image from a URL.
*   `POST /v1/multimodal/embed`: Get an embedding for text or an image using a multimodal model.
*   `GET /v1/models`: List the available models.

All endpoints require an `X-API-KEY` header for authentication.

## Configuration

Configuration is managed through environment variables in the `.env` file:

*   `VALID_API_KEYS`: A comma-separated list of valid API keys.
*   `MODEL_CACHE_DIR`: The directory to store downloaded models (defaults to `./model_cache`).

The `docker-compose.yml` file mounts the model cache directory to persist models between container restarts.
