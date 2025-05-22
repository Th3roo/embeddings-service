import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Union, Optional
from pydantic import BaseModel, HttpUrl, Field
from PIL import Image # Required for type hinting and potentially for operations if not fully encapsulated

from app.models.multimodal_embedder import MultimodalEmbedder
from app.auth import get_api_key # For API key authentication
from app.schemas import EmbeddingResponse # Assuming a generic EmbeddingResponse schema exists

# Setup logger for this router
logger = logging.getLogger(__name__)

router = APIRouter()

# Try to instantiate the MultimodalEmbedder
# This instance will be used by the endpoint
try:
    multimodal_embedder = MultimodalEmbedder()
    logger.info(f"MultimodalEmbedder loaded for /v1/multimodal/embed endpoint: {multimodal_embedder.model_name}")
except Exception as e:
    logger.error(f"Failed to initialize MultimodalEmbedder for /v1/multimodal/embed endpoint: {e}", exc_info=True)
    multimodal_embedder = None # Set to None if loading fails

# This is a Pydantic model for the request body, but we are using Form inputs for this endpoint.
# It's good for documentation or if we switch to JSON body later for some fields.
# class MultimodalEmbeddingRequest(BaseModel):
# text: Optional[str] = None
# image_url: Optional[HttpUrl] = None
    # If we want to support model selection in the future, add:
    # model_name: Optional[str] = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

@router.post("/multimodal/embed", response_model=EmbeddingResponse, tags=["Multimodal_v1"])
async def get_multimodal_embedding_v1(
    text: Optional[str] = Form(None, description="Text to embed."),
    image_url: Optional[HttpUrl] = Form(None, description="URL of the image to embed."),
    image_file: Optional[UploadFile] = File(None, description="Image file to embed."),
    api_key: str = Depends(get_api_key)
):
    """
    Генерирует эмбеддинг для текста или изображения с использованием мультимодальной модели.
    
    Предоставьте либо 'text', либо 'image_url', либо 'image_file'.
    """
    if not multimodal_embedder:
        logger.error("MultimodalEmbedder not loaded, cannot process request for multimodal embedding.")
        raise HTTPException(status_code=503, detail="Multimodal embedding model is not available.")

    embedding_input: Union[str, Image.Image]
    input_type_provided = sum([bool(text), bool(str(image_url) if image_url else None), bool(image_file)])


    if input_type_provided != 1:
        raise HTTPException(status_code=400, detail="Please provide exactly one of: 'text', 'image_url', or 'image_file'.")

    try:
        if text:
            embedding_input = text
            logger.info(f"Processing multimodal embedding for text: {text[:50]}...")
        elif image_url:
            # Use the helper from MultimodalEmbedder to load image from URL
            # Ensure image_url is converted to string if it's a Pydantic HttpUrl object
            pil_image = multimodal_embedder._load_image_from_source(str(image_url))
            embedding_input = pil_image
            logger.info(f"Processing multimodal embedding for image URL: {image_url}")
        elif image_file:
            # Ensure content type is an image
            if not image_file.content_type or not image_file.content_type.startswith("image/"):
                 raise HTTPException(status_code=400, detail=f"Invalid image file type: {image_file.content_type}. Please upload a valid image (e.g., JPEG, PNG).")
            image_bytes = await image_file.read()
            # Use the helper from MultimodalEmbedder to load image from bytes
            pil_image = multimodal_embedder._load_image_from_source(image_bytes)
            embedding_input = pil_image
            logger.info(f"Processing multimodal embedding for uploaded image file: {image_file.filename} (type: {image_file.content_type})")
        else:
            # This case should be caught by input_type_provided check, but as a fallback:
            raise HTTPException(status_code=400, detail="No input provided. Please provide 'text', 'image_url', or 'image_file'.")

        embedding_vector = multimodal_embedder.get_embedding(embedding_input)
        
        logger.info(f"Successfully generated multimodal embedding of dimension {len(embedding_vector)} for model {multimodal_embedder.model_name}")

        return EmbeddingResponse(
            model_name=multimodal_embedder.model_name,
            model_type="multimodal", # This should match the model_type in MultimodalEmbedder
            embedding=embedding_vector,
        )

    except ValueError as ve: 
        logger.error(f"ValueError during multimodal embedding: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re: 
        logger.error(f"RuntimeError during multimodal embedding (e.g., model not loaded): {re}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Model runtime error: {re}")
    except HTTPException as http_exc: # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during multimodal embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
