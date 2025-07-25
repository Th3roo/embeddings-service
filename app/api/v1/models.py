import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.schemas import (
    ModelInfo,
    AvailableModelsResponse,
)
from app.models.text_embedder import TextEmbedder
from app.models.image_embedder import ImageEmbedder
from app.models.multimodal_embedder import MultimodalEmbedder
from app.auth import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

try:
    text_embedder_instance = TextEmbedder()
    logger.info(
        f"TextEmbedder loaded for models endpoint: {text_embedder_instance.model_name}"
    )
except Exception as e:
    logger.error(
        f"Failed to initialize TextEmbedder for models endpoint: {e}", exc_info=True
    )
    text_embedder_instance = None

try:
    image_embedder_instance = ImageEmbedder()
    logger.info(
        f"ImageEmbedder loaded for models endpoint: {image_embedder_instance.model_name}"
    )
except Exception as e:
    logger.error(
        f"Failed to initialize ImageEmbedder for models endpoint: {e}", exc_info=True
    )
    image_embedder_instance = None

try:
    multimodal_embedder_instance = MultimodalEmbedder()
    logger.info(
        f"MultimodalEmbedder loaded for models endpoint: {multimodal_embedder_instance.model_name}"
    )
except Exception as e:
    logger.error(
        f"Failed to initialize MultimodalEmbedder for models endpoint: {e}",
        exc_info=True,
    )
    multimodal_embedder_instance = None


@router.get("/models", response_model=AvailableModelsResponse)
async def list_available_models_v1(api_key: str = Depends(get_api_key)):
    """
    Returns a list of available (default loaded) models and their types.
    """
    models_info_list: List[ModelInfo] = []

    if text_embedder_instance:
        models_info_list.append(
            ModelInfo(
                model_name=text_embedder_instance.model_name,
                model_type="text",
                description=getattr(text_embedder_instance, "description", "N/A"),
            )
        )
    else:
        logger.warning("Text embedder instance not available for /models endpoint.")

    if image_embedder_instance:
        models_info_list.append(
            ModelInfo(
                model_name=image_embedder_instance.model_name,
                model_type="image",
                description=getattr(image_embedder_instance, "description", "N/A"),
            )
        )
    else:
        logger.warning("Image embedder instance not available for /models endpoint.")

    if multimodal_embedder_instance:
        models_info_list.append(
            ModelInfo(
                model_name=multimodal_embedder_instance.model_name,
                model_type="multimodal",
                description=getattr(multimodal_embedder_instance, "description", "N/A"),
            )
        )
    else:
        logger.warning(
            "Multimodal embedder instance not available for /models endpoint."
        )

    if not models_info_list:
        logger.error(
            "No models are available as both text and image embedders failed to load."
        )
        pass

    return AvailableModelsResponse(models=models_info_list)
