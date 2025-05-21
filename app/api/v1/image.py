import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from app.schemas import ImageUrlRequest, EmbeddingResponse # Using existing EmbeddingResponse
from app.models.image_embedder import ImageEmbedder
from app.auth import get_api_key

# Setup logger for this router
logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate ImageEmbedder
# This instance will use the default model name specified in ImageEmbedder.
try:
    # model_cache_dir will use the default "./model_cache"
    image_embedder = ImageEmbedder()
    logger.info(f"ImageEmbedder loaded successfully with model: {image_embedder.model_name} and dimension: {image_embedder.dimension}")
except Exception as e:
    logger.error(f"Failed to initialize ImageEmbedder: {e}", exc_info=True)
    image_embedder = None # Ensure it's None if loading fails


@router.post("/embeddings/image/upload", response_model=EmbeddingResponse, tags=["Embeddings_v1_Image"])
async def create_image_embedding_upload_v1(
    image_file: UploadFile = File(...),
    model_name: Optional[str] = Body(None, description="Опционально: имя модели для изображений. Если не указано, используется модель по умолчанию."),
    api_key: str = Depends(get_api_key)
):
    """
    Создает эмбеддинг для загруженного изображения.
    Параметр `model_name` в теле запроса на данный момент игнорируется этим эндпоинтом,
    так как он использует фиксированный экземпляр ImageEmbedder с моделью по умолчанию.
    """
    if image_embedder is None:
        logger.error("Image embedder is not available due to loading failure.")
        raise HTTPException(status_code=503, detail="Image embedding model is not available.")

    if model_name and model_name != image_embedder.model_name:
        logger.warning(
            f"Client requested image model '{model_name}', "
            f"but this endpoint uses a fixed instance of '{image_embedder.model_name}'. "
            "The requested model_name is ignored."
        )
    
    try:
        contents = await image_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="No image file content provided.")

        embedding = image_embedder.get_embedding(contents)
        return EmbeddingResponse(
            embedding=embedding,
            model_used=image_embedder.model_name,
            dim=image_embedder.dimension
        )
    except ValueError as ve:
        logger.error(f"Validation error in image upload embedding: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime error in image upload embedding: {re}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Error processing image upload embedding with model {image_embedder.model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing image upload.")
    finally:
        if 'image_file' in locals() and image_file:
             await image_file.close()


@router.post("/embeddings/image/url", response_model=EmbeddingResponse, tags=["Embeddings_v1_Image"])
async def create_image_embedding_url_v1(
    request: ImageUrlRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Создает эмбеддинг для изображения по URL.
    Параметр `model_name` в теле запроса на данный момент игнорируется этим эндпоинтом,
    так как он использует фиксированный экземпляр ImageEmbedder с моделью по умолчанию.
    """
    if image_embedder is None:
        logger.error("Image embedder is not available due to loading failure.")
        raise HTTPException(status_code=503, detail="Image embedding model is not available.")

    if request.model_name and request.model_name != image_embedder.model_name:
        logger.warning(
            f"Client requested image model '{request.model_name}', "
            f"but this endpoint uses a fixed instance of '{image_embedder.model_name}'. "
            "The requested model_name is ignored."
        )

    try:
        embedding = image_embedder.get_embedding(request.url)
        return EmbeddingResponse(
            embedding=embedding,
            model_used=image_embedder.model_name,
            dim=image_embedder.dimension
        )
    except ValueError as ve:
        logger.error(f"Validation error in image URL embedding: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime error in image URL embedding: {re}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Error processing image URL embedding with model {image_embedder.model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing image URL.")
