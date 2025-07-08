import logging
from fastapi import APIRouter, Depends, HTTPException
from app.schemas import (
    TextRequest,
    EmbeddingResponse,
)  # Using existing EmbeddingResponse
from app.models.text_embedder import TextEmbedder
from app.auth import get_api_key  # Assuming get_api_key is accessible here or will be

# Setup logger for this router
logger = logging.getLogger(__name__)
# Configure logging if it's not configured globally or if specific router logging is needed
# For simplicity, assuming global logging is configured in main.py

router = APIRouter()

# Instantiate TextEmbedder
# For now, each router manages its own embedder instance.
# This instance will use the default model name specified in TextEmbedder.
try:
    # The TextEmbedder's __init__ method loads the model by default.
    # model_cache_dir will use the default "./model_cache"
    text_embedder = TextEmbedder()
    logger.info(
        f"TextEmbedder loaded successfully with model: {text_embedder.model_name} and dimension: {text_embedder.dimension}"
    )
except Exception as e:
    logger.error(f"Failed to initialize TextEmbedder: {e}", exc_info=True)
    text_embedder = None  # Ensure it's None if loading fails


@router.post("/embeddings/text", response_model=EmbeddingResponse)
async def create_text_embedding_v1(
    request: TextRequest, api_key: str = Depends(get_api_key)
):
    """
    Creates an embedding for the given text using the pre-loaded text model.
    The `model_name` field in the request is currently ignored by this endpoint,
    as it uses a router-specific instance of TextEmbedder with its default model.
    """
    if text_embedder is None:
        logger.error("Text embedder is not available due to loading failure.")
        raise HTTPException(
            status_code=503, detail="Text embedding model is not available."
        )

    # The current TextRequest schema has an optional model_name.
    # Since this router uses a fixed instance of TextEmbedder, we log if a different model was requested.
    if request.model_name and request.model_name != text_embedder.model_name:
        logger.warning(
            f"Client requested text model '{request.model_name}', "
            f"but this endpoint uses a fixed instance of '{text_embedder.model_name}'. "
            "The requested model_name is ignored."
        )

    try:
        # We use the router's instance of text_embedder.
        # The check `if embedder.model_type != "text":` is inherently covered
        # because we directly instantiate TextEmbedder.

        embedding = text_embedder.get_embedding(request.text)

        return EmbeddingResponse(
            embedding=embedding,
            model_used=text_embedder.model_name,  # Use the actual model name from the embedder
            dim=text_embedder.dimension,  # Use the actual dimension
        )
    except ValueError as ve:
        logger.error(f"Validation error in text embedding: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        # This might occur if the model failed to load but wasn't caught by the initial check
        logger.error(f"Runtime error in text embedding: {re}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(
            f"Error processing text embedding with model {text_embedder.model_name}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing text embedding.",
        )
