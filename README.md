# Embeddings Service

This service provides an API for generating text and image embeddings using pre-trained models. It is built with FastAPI and designed to be run with Docker.

## Features

-   **Text Embeddings**: Utilizes Sentence Transformers (default model: `all-MiniLM-L6-v2`) to generate embeddings for text strings.
-   **Image Embeddings**: Employs Hugging Face Transformers (default model: `google/vit-base-patch16-224`) to generate embeddings for images (provided as URLs or byte content).
-   **API Key Authentication**: Protects endpoints with API key authentication.
-   **Model Caching**: Downloads and caches models locally to improve startup times and reduce reliance on internet connectivity after the initial download.
-   **Dockerized**: Easy to deploy and run using Docker and Docker Compose.

## Prerequisites

-   Docker
-   Docker Compose

## Setup and Running

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <directory_name_after_clone_typically_embeddings-service> 
    ```
    (Replace `<repository_url>` and `<directory_name_after_clone_typically_embeddings-service>` accordingly)

2.  **Environment Variables:**
    Create a `.env` file by copying `example.env`:
    ```bash
    cp example.env .env
    ```
    Edit `.env` to set your desired `VALID_API_KEYS` (comma-separated if multiple).

3.  **Build and Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    The service will be available at `http://localhost:8000`. The API documentation (Swagger UI) can be accessed at `http://localhost:8000/docs`.

## API Endpoints

-   `POST /v1/embeddings/text`: Generates text embeddings.
    -   **Request Body**:
        ```json
        {
          "input": "Your text string here"
        }
        ```
    -   **Headers**: `X-API-Key: <your_api_key>`
-   `POST /v1/embeddings/image`: Generates image embeddings.
    -   **Request Body**:
        ```json
        {
          "input": "url_or_bytes_representing_image"
        }
        ```
        (Input can be a publicly accessible image URL. For binary image data, ensure your client sends it appropriately, e.g., as part of a multipart/form-data request if the endpoint is adapted for it, or as raw bytes if the client and server are set up for that. The current server implementation expects URL or raw bytes directly if feasible by the HTTP client library used to call it.)
    -   **Headers**: `X-API-Key: <your_api_key>`
-   `GET /v1/models`: Lists available models (name, type, dimension, description).
-   `GET /health`: Health check endpoint.

## Model Caching

To enhance performance and reduce redundant downloads, the service implements model caching:

-   **How it works**: When a model is first requested (either text or image), it is downloaded from its source (e.g., Hugging Face Hub for `transformers` models, or Sentence Transformers community models) and stored locally. Subsequent requests for the same model will load it from the local cache.
-   **Benefits**:
    -   Significantly speeds up application startup and first request times after the initial download.
    -   Allows operation even without an internet connection once models are cached.
-   **Default Cache Location**: Models are cached by default in the `./model_cache` directory within the application's root directory (i.e., `/app/model_cache` inside the Docker container).
-   **Persistence**: The `docker-compose.yml` file is configured to mount this `./model_cache` directory from your host machine into the container (`./model_cache:/app/model_cache`). This ensures that your downloaded models persist even if you stop and restart the Docker container, saving you from re-downloading them.

## Testing

Unit tests are located in the `tests/` directory and can be run using:

```bash
python -m unittest discover tests
```
(Ensure you have the necessary dependencies installed in your environment if running tests outside of Docker, e.g., by creating a virtual environment and running `pip install -r requirements.txt`).

## Project Structure

```
.
├── app/                  # Main application code
│   ├── api/              # API endpoint definitions (routers)
│   │   └── v1/           # API version 1
│   ├── models/           # Embedding model logic (base, image, text)
│   └── core/             # Configuration, security, etc.
├── tests/                # Unit tests
├── .env                  # Local environment variables (gitignored)
├── example.env           # Example environment variables file
├── Dockerfile            # Instructions for building the Docker image
├── docker-compose.yml    # Docker Compose configuration for local development
├── main.py               # FastAPI application entry point
├── README.md             # This file
└── requirements.txt      # Python package dependencies
```