import logging

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
)
from fastapi.responses import JSONResponse
from typing import (
    Optional,
)
import traceback
from contextlib import asynccontextmanager

from app.auth import (
    get_api_key,
)

from app import (
    get_embedder_instance,
    get_default_text_model_name,
    get_default_image_model_name,
    get_available_models_info,
    preload_models,
)

from app.api.v1 import text as api_text_v1
from app.api.v1 import image as api_image_v1
from app.api.v1 import models as api_models_v1
from app.api.v1 import multimodal as api_multimodal_v1


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Actions on application startup and shutdown, e.g. preloading models."""
    logger.info("Application startup...")
    try:
        preload_models()
    except Exception as e:
        logger.error(f"Error during model preloading: {e}", exc_info=True)
    logger.info("Application ready.")
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title="Embedding Service API",
    description="API for getting text and image embeddings.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(api_text_v1.router, prefix="/v1", tags=["V1 - Text Embeddings"])
app.include_router(api_image_v1.router, prefix="/v1", tags=["V1 - Image Embeddings"])
app.include_router(api_models_v1.router, prefix="/v1", tags=["V1 - Models"])
app.include_router(
    api_multimodal_v1.router, prefix="/v1", tags=["V1 - Multimodal Embeddings"]
)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "detail": str(exc)},
    )


@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to the Embedding Service API!"}


@app.get("/health", tags=["General"])
async def health_check():
    return {"status": "ok"}


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
