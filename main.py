import logging # Import logging

from fastapi import FastAPI, Depends, HTTPException # Removed UploadFile, File, Body as they are not used directly in main.py anymore
from fastapi.responses import JSONResponse
from typing import Optional # Removed List, Dict, Any as they are not used directly in main.py anymore
import traceback # Для детальных ошибок
from contextlib import asynccontextmanager # Import asynccontextmanager

# Schemas like TextRequest, EmbeddingResponse etc. are now used in the routers, not directly in main.py
# from app.schemas import (
#     TextRequest,
#     ImageUrlRequest,
#     EmbeddingResponse,
#     AvailableModelsResponse,
#     ModelInfo
# )
from app.auth import get_api_key # get_api_key is used by routers, so keep its origin in mind.
# The following app imports are related to the old way of loading models.
# They might be refactored or removed in future steps if preload_models and direct app access changes.
from app import get_embedder_instance, get_default_text_model_name, get_default_image_model_name, get_available_models_info, preload_models

# Import new routers
from app.api.v1 import text as api_text_v1
from app.api.v1 import image as api_image_v1
from app.api.v1 import models as api_models_v1


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

# Include V1 routers
app.include_router(api_text_v1.router, prefix="/v1", tags=["V1 - Text Embeddings"])
app.include_router(api_image_v1.router, prefix="/v1", tags=["V1 - Image Embeddings"])
app.include_router(api_models_v1.router, prefix="/v1", tags=["V1 - Models"])


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

# Old model and embedding endpoints are removed.
# They are now handled by the routers in app/api/v1/

import uvicorn # Import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)