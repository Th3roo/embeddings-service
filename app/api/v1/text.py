import logging
from fastapi import APIRouter, Depends, HTTPException
from app.schemas import (
    TextRequest,
    EmbeddingResponse,
)
from app.models.text_embedder import TextEmbedder
from app.auth import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

try:
    text_embedder = TextEmbedder()
    logger.info(
        f"TextEmbedder loaded successfully with model: {text_embedder.model_name} and dimension: {text_embedder.dimension}"
    )
except Exception as e:
    logger.error(f"Failed to initialize TextEmbedder: {e}", exc_info=True)
    text_embedder = None


@router.post("/embeddings/text", response_model=EmbeddingResponse)
async def create_text_embedding_v1(
    request: TextRequest, api_key: str = Depends(get_api_key)
):
    """
    Creates an embedding for the given text using the pre-loaded text model.
    The `model_name` field in the request is currently ignored by this endpoint.
    """
    if text_embedder is None:
        logger.error("Text embedder is not available due to loading failure.")
        raise HTTPException(
            status_code=503, detail="Text embedding model is not available."
        )

    if request.model_name and request.model_name != text_embedder.model_name:
        logger.warning(
            f"Client requested text model '{request.model_name}', "
            f"but this endpoint uses a fixed instance of '{text_embedder.model_name}'. "
            "The requested model_name is ignored."
        )

    try:
        embedding = text_embedder.get_embedding(request.text)

        return EmbeddingResponse(
            embedding=embedding,
            model_used=text_embedder.model_name,
            dim=text_embedder.dimension,
        )
    except ValueError as ve:
        logger.error(f"Validation error in text embedding: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
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
