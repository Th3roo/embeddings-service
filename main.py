import logging # Import logging

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import traceback # Для детальных ошибок
from contextlib import asynccontextmanager # Import asynccontextmanager

from app.schemas import (
    TextRequest,
    ImageUrlRequest,
    EmbeddingResponse,
    AvailableModelsResponse,
    ModelInfo
    
)
from app.auth import get_api_key
from app import get_embedder_instance, get_default_text_model_name, get_default_image_model_name, get_available_models_info, preload_models

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- События жизненного цикла (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Действия при старте и завершении приложения, например, предзагрузка моделей."""
    logger.info("Application startup...")
    try:
        preload_models() # Предзагружаем все зарегистрированные модели
    except Exception as e:
        logger.error(f"Error during model preloading: {e}", exc_info=True)
    logger.info("Application ready.")
    yield # Приложение работает здесь
    # Здесь можно добавить код для завершения работы, если нужно
    logger.info("Application shutdown.")


# --- Приложение FastAPI ---
app = FastAPI(
    title="Embedding Service API",
    description="API для получения текстовых и графических эмбеддингов.",
    version="0.2.0",
    lifespan=lifespan, # Указываем функцию lifespan
)

# --- Обработчики ошибок ---
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    # Логирование ошибки
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "detail": str(exc)},
    )

# --- Эндпоинты ---
@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to the Embedding Service API!"}

@app.get("/health", tags=["General"])
async def health_check():
    return {"status": "ok"}

@app.get("/models", response_model=AvailableModelsResponse, tags=["Models"])
async def list_available_models(api_key: str = Depends(get_api_key)):
    """Возвращает список доступных моделей и их типы."""
    models_info_raw = get_available_models_info()
    models_info = [ModelInfo(**info) for info in models_info_raw]
    return AvailableModelsResponse(models=models_info)


@app.post("/embeddings/text", response_model=EmbeddingResponse, tags=["Embeddings"])
async def create_text_embedding(
    request: TextRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Создает эмбеддинг для заданного текста.
    Можно опционально указать `model_name` в теле запроса.
    Если `model_name` не указан, используется модель по умолчанию для текста.
    """
    model_name_to_use = request.model_name or get_default_text_model_name()
    try:
        embedder = get_embedder_instance(model_name_to_use)
        if embedder.model_type != "text":
            raise HTTPException(status_code=400, detail=f"Model {model_name_to_use} is not a text model.")
        
        embedding = embedder.get_embedding(request.text)
        return EmbeddingResponse(
            embedding=embedding,
            model_used=embedder.model_name,
            dim=embedder.dimension
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Error processing text embedding for model {model_name_to_use}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.post("/embeddings/image/upload", response_model=EmbeddingResponse, tags=["Embeddings"])
async def create_image_embedding_upload(
    image_file: UploadFile = File(...),
    model_name: Optional[str] = Body(None, description="Опционально: имя модели для изображений. Если не указано, используется модель по умолчанию."),
    api_key: str = Depends(get_api_key)
):
    """
    Создает эмбеддинг для загруженного изображения.
    Опционально можно передать `model_name` в теле form-data.
    """
    model_name_to_use = model_name or get_default_image_model_name()
    try:
        embedder = get_embedder_instance(model_name_to_use)
        if embedder.model_type != "image":
            raise HTTPException(status_code=400, detail=f"Model {model_name_to_use} is not an image model.")

        contents = await image_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="No image file content provided.")

        embedding = embedder.get_embedding(contents)
        return EmbeddingResponse(
            embedding=embedding,
            model_used=embedder.model_name,
            dim=embedder.dimension
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Error processing image upload embedding for model {model_name_to_use}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if 'image_file' in locals() and image_file:
             await image_file.close()


@app.post("/embeddings/image/url", response_model=EmbeddingResponse, tags=["Embeddings"])
async def create_image_embedding_url(
    request: ImageUrlRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Создает эмбеддинг для изображения по URL.
    Можно опционально указать `model_name` в теле запроса.
    """
    model_name_to_use = request.model_name or get_default_image_model_name()
    try:
        embedder = get_embedder_instance(model_name_to_use)
        if embedder.model_type != "image":
            raise HTTPException(status_code=400, detail=f"Model {model_name_to_use} is not an image model.")
        
        embedding = embedder.get_embedding(request.url)
        return EmbeddingResponse(
            embedding=embedding,
            model_used=embedder.model_name,
            dim=embedder.dimension
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Error processing image URL embedding for model {model_name_to_use}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

import uvicorn # Import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)