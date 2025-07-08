import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Union, Optional
from pydantic import BaseModel, HttpUrl, Field
from PIL import (
    Image,
)  # Required for type hinting and potentially for operations if not fully encapsulated

from app.models.multimodal_embedder import MultimodalEmbedder
from app.auth import get_api_key  # For API key authentication
from app.schemas import (
    EmbeddingResponse,
)  # Assuming a generic EmbeddingResponse schema exists

# Setup logger for this router
logger = logging.getLogger(__name__)

router = APIRouter()

# Try to instantiate the MultimodalEmbedder
# This instance will be used by the endpoint
try:
    multimodal_embedder = MultimodalEmbedder()
    logger.info(
        f"MultimodalEmbedder loaded for /v1/multimodal/embed endpoint: {multimodal_embedder.model_name}"
    )
except Exception as e:
    logger.error(
        f"Failed to initialize MultimodalEmbedder for /v1/multimodal/embed endpoint: {e}",
        exc_info=True,
    )
    multimodal_embedder = None  # Set to None if loading fails

# This is a Pydantic model for the request body, but we are using Form inputs for this endpoint.
# It's good for documentation or if we switch to JSON body later for some fields.
# class MultimodalEmbeddingRequest(BaseModel):
# text: Optional[str] = None
# image_url: Optional[HttpUrl] = None
# If we want to support model selection in the future, add:
# model_name: Optional[str] = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


@router.post("/multimodal/embed", response_model=EmbeddingResponse)
async def get_multimodal_embedding_v1(
    text: Optional[str] = Form(None, description="Text to embed."),
    # Receive image_url as a string initially to handle potential empty string inputs
    image_url: Optional[str] = Form(None, description="URL of the image to embed."),
    image_file: Optional[UploadFile] = File(None, description="Image file to embed."),
    api_key: str = Depends(get_api_key),
):
    """
    Генерирует эмбеддинг для текста или изображения с использованием мультимодальной модели.

    Предоставьте либо 'text', либо 'image_url', либо 'image_file'.
    """
    if not multimodal_embedder:
        logger.error(
            "MultimodalEmbedder not loaded, cannot process request for multimodal embedding."
        )
        raise HTTPException(
            status_code=503, detail="Multimodal embedding model is not available."
        )

    # Determine which input was actually provided, treating None and empty strings as not provided
    is_text_provided = text is not None and text != ""
    is_image_url_provided = image_url is not None and image_url != ""
    is_image_file_provided = image_file is not None

    input_type_provided_count = sum(
        [is_text_provided, is_image_url_provided, is_image_file_provided]
    )

    if input_type_provided_count != 1:
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one of: 'text', 'image_url', or 'image_file'.",
        )

    try:
        embedding_input: Union[str, Image.Image]

        if is_text_provided:
            embedding_input = text
            logger.info(f"Processing multimodal embedding for text: {text[:50]}...")
        elif is_image_url_provided:
            # If image_url is provided, validate it as an HttpUrl
            try:
                valid_image_url = HttpUrl(image_url)
            except Exception:
                logger.error(f"Invalid URL format provided for image_url: {image_url}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid URL format provided for image_url: {image_url}",
                )

            pil_image = multimodal_embedder._load_image_from_source(
                str(valid_image_url)
            )
            embedding_input = pil_image
            logger.info(f"Processing multimodal embedding for image URL: {image_url}")
        elif is_image_file_provided:
            # Ensure content type is an image
            if not image_file.content_type or not image_file.content_type.startswith(
                "image/"
            ):
                # Remember to close the file in case of error before reading bytes
                await image_file.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file type: {image_file.content_type}. Please upload a valid image (e.g., JPEG, PNG).",
                )
            image_bytes = await image_file.read()
            await image_file.close()  # Close the file after reading
            # Use the helper from MultimodalEmbedder to load image from bytes
            pil_image = multimodal_embedder._load_image_from_source(image_bytes)
            embedding_input = pil_image
            logger.info(
                f"Processing multimodal embedding for uploaded image file: {image_file.filename} (type: {image_file.content_type})"
            )
        else:
            # This case should be caught by input_type_provided_count check, but as a fallback:
            raise HTTPException(
                status_code=400,
                detail="No input provided. Please provide 'text', 'image_url', or 'image_file'.",
            )

        embedding_vector = multimodal_embedder.get_embedding(embedding_input)

        logger.info(
            f"Successfully generated multimodal embedding of dimension {len(embedding_vector)} for model {multimodal_embedder.model_name}"
        )

        return EmbeddingResponse(
            model_name=multimodal_embedder.model_name,
            model_type="multimodal",  # This should match the model_type in MultimodalEmbedder
            embedding=embedding_vector,
            model_used=multimodal_embedder.model_name,
            dim=multimodal_embedder.dimension,
        )

    except ValueError as ve:
        logger.error(f"ValueError during multimodal embedding: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(
            f"RuntimeError during multimodal embedding (e.g., model not loaded): {re}",
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail=f"Model runtime error: {re}")
    except HTTPException as http_exc:  # Re-raise HTTPExceptions directly
        # Ensure file is closed even if HTTPException is re-raised from image_file processing
        if is_image_file_provided and image_file:
            try:
                await image_file.close()
            except Exception:
                pass  # Ignore errors during close if already closed or invalid object
        raise http_exc
    except Exception as e:
        logger.error(
            f"Unexpected error during multimodal embedding: {e}", exc_info=True
        )
        # Ensure file is closed on unexpected errors
        if is_image_file_provided and image_file:
            try:
                await image_file.close()
            except Exception:
                pass
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
